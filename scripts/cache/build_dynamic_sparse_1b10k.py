#!/usr/bin/env python3
"""Dynamic per-token top-K SAE cache, SPARSE (CSR-by-token) — Gemma-3-1B, 10k, BATCHED (bs=16).

10k variant of build_dynamic_sparse.py. Same construction and on-disk format (a single
sparse.pt payload: col/val/tok_ptr + labels/roles/lengths/splits + width), the format the
dynamic trainer (train_dynamic_gpu.py) and analysis loaders expect. Differences vs the 2k
original:
  * SRC/DATASET/model/SAE point at the 10k 1B pipeline
  * teacher-forced prefill is BATCHED (right-pad + attention_mask, slice each record's real
    tokens out of hidden_states) instead of one record at a time
  * project root derived from __file__ so it runs in this nested layout

Per token, per layer: keep that token's own top-K (=150) features (JumpReLU magnitude) from
the full 16,384 dictionary. Columns = union of features appearing in any TRAIN token's top-K
per layer, capped to CAP_PER_LAYER (=6000) most-frequent. Stored CSR-by-token so the cache is
a few tens of GB (sparse), not a dense total×width matrix.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer


# local imports
from important_scripts.paths import PROJECT_ROOT
from important_scripts.model.models import TOPICS

ROOT = PROJECT_ROOT

SRC = ROOT / "cache/sae500_10k_1b_prompt_response"  # lengths/labels/roles/split
OUT = ROOT / "cache/dyn150_sparse_10k_1b"
DATASET = ROOT / "runs/GPT_5.6_10k_2108/dataset.final.jsonl"
GEMMA = "google/gemma-3-1b-it"
REL = "gemma-scope-2-1b-it-res-all"
LAYERS = list(range(16, 26))
K = 150
CAP_PER_LAYER = 6000
BATCH = 8
DEVICE = "cuda"
TAG_RE = re.compile(r"</?(?:" + "|".join(re.escape(t) for t in TOPICS) + r")>")
strip = lambda s: TAG_RE.sub("", s)


def ids_for(rec, tok):
    """Tokenize one {prompt,response} record (tags stripped) via the chat template to an id list."""
    x = tok.apply_chat_template(
        [
            {"role": "user", "content": strip(rec["prompt"])},
            {"role": "assistant", "content": strip(rec["response"])},
        ],
        tokenize=True,
        add_generation_prompt=False,
    )
    if x and not isinstance(x[0], int):
        x = x[0].ids
    return list(x)


def main() -> None:
    """Batched teacher-forced prefill -> per-token top-K SAE features -> CSR-by-token sparse cache."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--cap", type=int, default=CAP_PER_LAYER)
    a = ap.parse_args()
    bs, cap = a.batch, a.cap

    torch.set_grad_enabled(False)
    OUT.mkdir(parents=True, exist_ok=True)
    lengths = np.load(SRC / "lengths.npy")
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    total = int(offsets[-1])
    train_idx = {int(i) for i in np.load(SRC / "split_indices.npz")["train"]}
    records = [json.loads(l)["record"] for l in DATASET.open()]
    assert len(records) == len(lengths), f"records={len(records)} != lengths={len(lengths)}"
    print(f"records={len(records)} tokens={total} K={K} cap={cap} bs={bs}", flush=True)

    tok = AutoTokenizer.from_pretrained(GEMMA)
    tok.padding_side = "right"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = (
        AutoModelForCausalLM.from_pretrained(
            GEMMA, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .to(DEVICE)
        .eval()
    )
    saes = []
    for L in LAYERS:
        s = SAE.from_pretrained(release=REL, sae_id=f"layer_{L}_width_16k_l0_small", device=DEVICE)
        s.eval()
        saes.append((L, s.W_enc.float(), s.b_enc.float(), s.b_dec.float(), s.threshold.float()))
    d_sae = saes[0][1].shape[1]

    # topk stores are DISK-backed memmaps. Measured nnz≈9.1e9 (mean ~1030/token), so the output
    # col/val is ~55 GiB; 80 GiB of in-RAM topk + that 55 GiB overflows the ~109 GiB cgroup cap.
    # So topk lives on disk (file created empty — every slot is overwritten during prefill, so NO
    # eager init write, which is what stalled an earlier attempt) and the CSR is assembled from it
    # in chunks below (anonymous RAM peak ~ output + one chunk ≈ 60 GiB; topk pages stay reclaimable).
    nL = len(LAYERS)
    tk_idx_path = OUT / "_topk_idx.dat"
    tk_val_path = OUT / "_topk_val.dat"
    topk_idx = np.memmap(tk_idx_path, mode="w+", dtype=np.int32, shape=(total, nL, K))
    topk_val = np.memmap(tk_val_path, mode="w+", dtype=np.float16, shape=(total, nL, K))
    freq = [np.zeros(d_sae, dtype=np.int64) for _ in LAYERS]

    t0 = time.time()
    with torch.inference_mode():
        for b0 in range(0, len(records), bs):
            batch = list(range(b0, min(b0 + bs, len(records))))
            batch_ids = []
            for i in batch:
                ii = ids_for(records[i], tok)
                if len(ii) != int(lengths[i]):
                    raise RuntimeError(f"record {i}: tok={len(ii)} cache={int(lengths[i])}")
                batch_ids.append(ii)
            maxlen = max(len(x) for x in batch_ids)
            inp = torch.full((len(batch), maxlen), pad_id, device=DEVICE, dtype=torch.long)
            am = torch.zeros((len(batch), maxlen), device=DEVICE, dtype=torch.long)
            for j, x in enumerate(batch_ids):
                inp[j, : len(x)] = torch.tensor(x, device=DEVICE)
                am[j, : len(x)] = 1
            out = model(
                input_ids=inp,
                attention_mask=am,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            for li, (L, W, b_enc, b_dec, thr) in enumerate(saes):
                hs = out.hidden_states[L + 1].float()  # (B,T,d)
                pre = (hs - b_dec) @ W + b_enc
                feat = pre * (pre > thr)  # JumpReLU
                vals, idx = torch.topk(feat, K, dim=-1)  # (B,T,K)
                zero = vals <= 0
                idx = idx.clone()
                idx[zero] = -1
                idx_c = idx.cpu().numpy()
                vals_c = vals.to(torch.float16).cpu().numpy()
                zero_c = zero.cpu().numpy()
                for j, i in enumerate(batch):
                    T = int(lengths[i])
                    lo, hi = int(offsets[i]), int(offsets[i + 1])
                    topk_idx[lo:hi, li, :] = idx_c[j, :T]
                    topk_val[lo:hi, li, :] = vals_c[j, :T]
                    if i in train_idx:
                        flat = idx_c[j, :T][~zero_c[j, :T]].reshape(-1)
                        if flat.size:
                            freq[li] += np.bincount(flat, minlength=d_sae)
            if (b0 // bs) % 40 == 0:
                done = min(b0 + bs, len(records))
                rate = done / (time.time() - t0)
                print(
                    f"  prefilled {done}/{len(records)}  {rate:.1f} rec/s "
                    f"eta {(len(records) - done) / max(rate, 1e-9) / 60:.1f} min",
                    flush=True,
                )
    del model, saes
    torch.cuda.empty_cache()
    print(f"prefill complete in {(time.time() - t0) / 60:.1f} min", flush=True)

    # union columns per layer (cap to most-frequent); also record natural (uncapped) width
    widths, natural_widths, luts, selected = [], [], [], {}
    for li, L in enumerate(LAYERS):
        present = np.where(freq[li] > 0)[0]
        natural_widths.append(len(present))
        keep = (
            present if len(present) <= cap else present[np.argsort(freq[li][present])[::-1][:cap]]
        )
        keep = np.sort(keep)
        lut = np.full(d_sae, -1, dtype=np.int64)
        lut[keep] = np.arange(len(keep))
        luts.append(lut)
        widths.append(len(keep))
        selected[f"layer_{L}"] = keep.astype(np.int64)
    layer_off = np.concatenate(([0], np.cumsum(widths)))
    W_total = int(layer_off[-1])
    sae_index = np.concatenate([selected[f"layer_{L}"] for L in LAYERS]).astype(np.int64)
    layer_cols = {
        int(L): (int(layer_off[li]), int(layer_off[li + 1])) for li, L in enumerate(LAYERS)
    }
    print(f"natural(uncapped) widths={natural_widths}", flush=True)
    print(
        f"kept widths={widths} TOTAL_WIDTH={W_total}  (cap hit on "
        f"{sum(n > cap for n in natural_widths)}/{len(LAYERS)} layers)",
        flush=True,
    )

    # Assemble CSR-by-token from the disk-backed topk in CHUNKS (int32 throughout): anonymous-RAM
    # peak stays ~ col+val+one chunk instead of a full (total,L,K) int64 index array. Two passes
    # over the finished topk: (A) count kept nnz/token -> tok_ptr, (B) fill preallocated col/val.
    CH = 400_000

    def map_layer(idx, li):
        """idx: (n,K) int32 topk feature ids (-1 = empty). Returns (mapped_global_col, ok_mask)."""
        valid = idx >= 0
        mapped = luts[li][np.where(valid, idx, 0)]  # lut[-1]->global col, else <0
        ok = valid & (mapped >= 0)
        return mapped, ok

    # pass A: kept nnz per token (after cap)
    nnz_per_tok = np.zeros(total, dtype=np.int64)
    fired = 0
    for c0 in range(0, total, CH):
        c1 = min(c0 + CH, total)
        idx_c = np.asarray(topk_idx[c0:c1])
        keep_ct = np.zeros(c1 - c0, dtype=np.int64)
        for li in range(nL):
            _, ok = map_layer(idx_c[:, li, :], li)
            fired += int((idx_c[:, li, :] >= 0).sum())
            keep_ct += ok.sum(axis=1)
        nnz_per_tok[c0:c1] = keep_ct
    total_nnz = int(nnz_per_tok.sum())
    tok_ptr = np.concatenate(([0], np.cumsum(nnz_per_tok)))
    kept = total_nnz
    print(
        f"kept {100 * kept / max(fired, 1):.2f}% of firing feats; nnz={total_nnz} "
        f"mean nnz/token={total_nnz / total:.0f}",
        flush=True,
    )

    # pass B: fill preallocated col/val (token-major, layer-then-k within a token)
    col = np.empty(total_nnz, dtype=np.int32)
    val = np.empty(total_nnz, dtype=np.float16)
    w = 0
    for c0 in range(0, total, CH):
        c1 = min(c0 + CH, total)
        n = c1 - c0
        idx_c = np.asarray(topk_idx[c0:c1])
        val_c = np.asarray(topk_val[c0:c1])
        gcol = np.full((n, nL, K), -1, dtype=np.int32)
        for li in range(nL):
            mapped, ok = map_layer(idx_c[:, li, :], li)
            gcol[:, li, :][ok] = (layer_off[li] + mapped[ok]).astype(np.int32)
        gflat = gcol.reshape(n, -1)
        vflat = val_c.reshape(n, -1)
        m = gflat >= 0
        cnt = int(m.sum())
        col[w : w + cnt] = gflat[m]
        val[w : w + cnt] = vflat[m]
        w += cnt
    assert w == total_nnz, f"filled {w} != nnz {total_nnz}"
    del topk_idx, topk_val  # release the memmaps
    for p in (tk_idx_path, tk_val_path):
        try:
            os.remove(p)
        except OSError:
            pass

    sp = np.load(SRC / "split_indices.npz")
    payload = {
        "col": torch.from_numpy(col),
        "val": torch.from_numpy(val),
        "tok_ptr": torch.from_numpy(tok_ptr),
        "labels": torch.from_numpy(np.load(SRC / "labels.npy").astype(np.int64)),
        "roles": torch.from_numpy(np.load(SRC / "role_ids.npy").astype(np.int64)),
        "lengths": torch.from_numpy(lengths.astype(np.int64)),
        "width": W_total,
        "widths": torch.tensor(widths, dtype=torch.long),
        "sae_index": torch.from_numpy(sae_index),
        "layer_cols": layer_cols,
        "layers": list(LAYERS),
        "train": torch.from_numpy(sp["train"].astype(np.int64)),
        "validation": torch.from_numpy(sp["validation"].astype(np.int64)),
        "test": torch.from_numpy(sp["test"].astype(np.int64)),
    }
    torch.save(payload, OUT / "sparse.pt")
    np.savez(OUT / "selected_features.npz", **selected)
    (OUT / "meta.json").write_text(
        json.dumps(
            {
                "width": W_total,
                "widths": widths,
                "natural_widths": natural_widths,
                "K": K,
                "cap": cap,
                "cap_hit_layers": int(sum(n > cap for n in natural_widths)),
                "kept_fraction": kept / fired,
                "nnz": int(col.size),
                "tokens": total,
                "batch": bs,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"wrote {OUT}/sparse.pt  ({os.path.getsize(OUT / 'sparse.pt') / 1e9:.2f} GB)  width={W_total}",
        flush=True,
    )


if __name__ == "__main__":
    main()
