#!/usr/bin/env python3
"""Dynamic per-token top-K SAE cache, SPARSE (CSR-by-token) — Gemma-3-4B, 10k — SHARDED build.

Rewrite of build_dynamic_sparse_4b10k.py that trades the single ~80 GiB preallocated topk
memmap for per-batch .pt shards + a single streaming merge. OUTPUT is byte-compatible with
finalize_dynamic_sparse.py (col.npy/val.npy/tok_ptr.npy + sidecars), read unchanged by the
dynamic trainer (train_dynamic_gpu_lowmem.py) and the analysis loaders.

Why sharded: the old two-stage flow's peak DISK was ~135 GB — the ~80 GB intermediate topk and
the ~55 GB final CSR coexist during finalize (topk deleted only at the very end). Here each
shard is deleted as its CSR is written, so peak disk collapses to ~the output size (~55-60 GB).
idx shards are int16 (feature ids < 16384 << 32767), halving their footprint vs int32. RAM
stays a few hundred MB throughout (per-batch on prefill, one shard at a time on merge) — well
under the ~116 GiB cgroup cap.

Phases:
  1. prefill (bs=16): forward -> per-layer top-K -> save OUT/_shards/shard_{b:05d}.pt holding
     {idx:int16 (n_tok,nL,K), val:fp16 (n_tok,nL,K)} with tokens in global record order. TRAIN-
     token feature frequency accumulated in RAM and checkpointed to OUT/_freq.npy so --merge-only
     can rebuild the CSR without re-running prefill (the merge is where the old run hit the wall).
  2. merge (one op): _freq -> per-layer union columns (cap) -> LUT; a count pass over the shards
     builds tok_ptr/total_nnz; a write pass streams the shards into col.npy/val.npy, deleting each
     shard right after it is consumed.

Flags: --batch (16), --cap (6000), --merge-only (skip prefill; reuse OUT/_shards + OUT/_freq.npy).
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

SRC = ROOT / "cache/sae500_10k_gemma4b_prompt_response"  # lengths/labels/roles/split
OUT = ROOT / "cache/dyn150_sparse_10k_4b"
SHARD_DIR = OUT / "_shards"
FREQ_PATH = OUT / "_freq.npy"
DATASET = ROOT / "runs/GPT_5.6_10k_2108/dataset.final.jsonl"
GEMMA = "google/gemma-3-4b-it"
REL = "gemma-scope-2-4b-it-res-all"
LAYERS = list(range(24, 34))
K = 150
CAP_PER_LAYER = 6000
BATCH = 16
DEVICE = "cuda"
CH_SHARDS = 1  # merge streams one shard at a time
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


# ----------------------------------------------------------------------------- prefill
def prefill(bs: int, lengths, offsets, total, train_idx, records, nL):
    """Stage 1: batched prefill writing per-batch topk .pt shards + accumulating TRAIN feature freq."""
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
    freq = np.zeros((nL, d_sae), dtype=np.int64)

    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with torch.inference_mode():
        for bi, b0 in enumerate(range(0, len(records), bs)):
            batch = list(range(b0, min(b0 + bs, len(records))))
            batch_ids = []
            for i in batch:
                ii = ids_for(records[i], tok)
                if len(ii) != int(lengths[i]):
                    raise RuntimeError(f"record {i}: tok={len(ii)} cache={int(lengths[i])}")
                batch_ids.append(ii)
            maxlen = max(len(x) for x in batch_ids)
            # per-record token offsets WITHIN this shard (global record order)
            Ts = [int(lengths[i]) for i in batch]
            wstart = np.concatenate(([0], np.cumsum(Ts)))
            n_tok = int(wstart[-1])
            shard_idx = np.empty((n_tok, nL, K), dtype=np.int16)
            shard_val = np.empty((n_tok, nL, K), dtype=np.float16)

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
                    T = Ts[j]
                    a, b = int(wstart[j]), int(wstart[j + 1])
                    shard_idx[a:b, li, :] = idx_c[j, :T].astype(np.int16)
                    shard_val[a:b, li, :] = vals_c[j, :T]
                    if i in train_idx:
                        flat = idx_c[j, :T][~zero_c[j, :T]].reshape(-1)
                        if flat.size:
                            freq[li] += np.bincount(flat, minlength=d_sae)
            torch.save(
                {"idx": torch.from_numpy(shard_idx), "val": torch.from_numpy(shard_val)},
                SHARD_DIR / f"shard_{bi:05d}.pt",
            )
            if bi % 40 == 0:
                done = min(b0 + bs, len(records))
                rate = done / (time.time() - t0)
                print(
                    f"  prefilled {done}/{len(records)}  {rate:.1f} rec/s "
                    f"eta {(len(records) - done) / max(rate, 1e-9) / 60:.1f} min",
                    flush=True,
                )
    del model, saes
    torch.cuda.empty_cache()
    np.save(FREQ_PATH, freq)
    print(
        f"prefill complete in {(time.time() - t0) / 60:.1f} min; freq -> {FREQ_PATH.name}",
        flush=True,
    )
    return freq


# ----------------------------------------------------------------------------- merge
def merge(cap: int, lengths, offsets, total, nL, bs):
    """Stage 2: pick per-layer union columns (freq-capped) and stream the shards into the CSR cache."""
    freq = np.load(FREQ_PATH)
    shards = sorted(SHARD_DIR.glob("shard_*.pt"))
    if not shards:
        raise RuntimeError(f"no shards in {SHARD_DIR}")

    # union columns per layer (cap to most-frequent); record natural (uncapped) width
    widths, natural, luts, selected = [], [], [], {}
    for li, L in enumerate(LAYERS):
        present = np.where(freq[li] > 0)[0]
        natural.append(len(present))
        keep = (
            present if len(present) <= cap else present[np.argsort(freq[li][present])[::-1][:cap]]
        )
        keep = np.sort(keep)
        lut = np.full(freq.shape[1], -1, dtype=np.int64)
        lut[keep] = np.arange(len(keep))
        luts.append(lut)
        widths.append(len(keep))
        selected[f"layer_{L}"] = keep.astype(np.int64)
    layer_off = np.concatenate(([0], np.cumsum(widths)))
    W_total = int(layer_off[-1])
    sae_index = np.concatenate([selected[f"layer_{L}"] for L in LAYERS]).astype(np.int64)
    layer_cols = {
        int(L): [int(layer_off[li]), int(layer_off[li + 1])] for li, L in enumerate(LAYERS)
    }
    print(
        f"natural widths={natural}\nkept widths={widths} W_total={W_total} "
        f"(cap hit {sum(n > cap for n in natural)}/{nL})",
        flush=True,
    )

    def map_layer(idx, li):
        """Map raw SAE ids (-1 = empty) to union-column indices; returns (mapped, keep_mask)."""
        valid = idx >= 0
        mapped = luts[li][np.where(valid, idx, 0)]
        return mapped, valid & (mapped >= 0)

    # pass A: kept nnz per token -> tok_ptr  (shards kept for pass B)
    nnz_per_tok = np.zeros(total, dtype=np.int64)
    p = 0
    fired = 0
    t0 = time.time()
    for si, sp_path in enumerate(shards):
        d = torch.load(sp_path, map_location="cpu")
        idx_s = d["idx"].numpy()
        n = idx_s.shape[0]
        kc = np.zeros(n, dtype=np.int64)
        for li in range(nL):
            _, ok = map_layer(idx_s[:, li, :], li)
            fired += int((idx_s[:, li, :] >= 0).sum())
            kc += ok.sum(axis=1)
        nnz_per_tok[p : p + n] = kc
        p += n
    assert p == total, f"pass-A tokens {p} != {total}"
    total_nnz = int(nnz_per_tok.sum())
    tok_ptr = np.concatenate(([0], np.cumsum(nnz_per_tok)))
    print(
        f"[nnz] total_nnz={total_nnz} mean/token={total_nnz / total:.0f} "
        f"kept={100 * total_nnz / max(fired, 1):.2f}%  {time.time() - t0:.0f}s",
        flush=True,
    )

    # pass B: stream shards -> col/val on-disk memmaps; delete each shard as consumed
    col = np.lib.format.open_memmap(OUT / "col.npy", mode="w+", dtype=np.int32, shape=(total_nnz,))
    val = np.lib.format.open_memmap(
        OUT / "val.npy", mode="w+", dtype=np.float16, shape=(total_nnz,)
    )
    w = 0
    for si, sp_path in enumerate(shards):
        d = torch.load(sp_path, map_location="cpu")
        idx_s = d["idx"].numpy()
        val_s = d["val"].numpy()
        n = idx_s.shape[0]
        g = np.full((n, nL, K), -1, dtype=np.int32)
        for li in range(nL):
            mapped, ok = map_layer(idx_s[:, li, :], li)
            g[:, li, :][ok] = (layer_off[li] + mapped[ok]).astype(np.int32)
        gflat = g.reshape(n, -1)
        vflat = val_s.reshape(n, -1)
        m = gflat >= 0
        cnt = int(m.sum())
        col[w : w + cnt] = gflat[m]
        val[w : w + cnt] = vflat[m]
        w += cnt
        os.remove(sp_path)  # <-- shard freed immediately
    assert w == total_nnz, f"{w} != {total_nnz}"
    col.flush()
    val.flush()
    del col, val
    print(f"[csr] wrote col.npy/val.npy {time.time() - t0:.0f}s", flush=True)

    # sidecars (identical set/format to finalize_dynamic_sparse.py)
    sp = np.load(SRC / "split_indices.npz")
    np.save(OUT / "tok_ptr.npy", tok_ptr)
    np.save(OUT / "labels.npy", np.load(SRC / "labels.npy").astype(np.int64))
    np.save(OUT / "roles.npy", np.load(SRC / "role_ids.npy").astype(np.int64))
    np.save(OUT / "lengths.npy", lengths.astype(np.int64))
    np.savez(OUT / "split_indices.npz", **{k: sp[k] for k in sp.files})
    np.savez(OUT / "selected_features.npz", **selected)
    np.save(OUT / "sae_index.npy", sae_index)
    (OUT / "meta.json").write_text(
        json.dumps(
            {
                "width": W_total,
                "widths": widths,
                "natural_widths": natural,
                "K": K,
                "cap": cap,
                "layer_cols": layer_cols,
                "layers": LAYERS,
                "nnz": total_nnz,
                "tokens": total,
                "mean_nnz_per_token": total_nnz / total,
                "batch": bs,
            },
            indent=2,
        )
        + "\n"
    )
    # tidy: drop the freq checkpoint + empty shard dir
    try:
        os.remove(FREQ_PATH)
    except OSError:
        pass
    try:
        SHARD_DIR.rmdir()
    except OSError:
        pass
    sz = sum((OUT / f).stat().st_size for f in ("col.npy", "val.npy", "tok_ptr.npy")) / 1e9
    print(
        f"DONE width={W_total} nnz={total_nnz} cache≈{sz:.1f} GB  {time.time() - t0:.0f}s",
        flush=True,
    )


def main() -> None:
    """Drive the two-stage sharded build: prefill -> shards, then merge -> CSR sparse cache."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--cap", type=int, default=CAP_PER_LAYER)
    ap.add_argument(
        "--merge-only",
        action="store_true",
        help="skip prefill; rebuild CSR from existing OUT/_shards + OUT/_freq.npy",
    )
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    OUT.mkdir(parents=True, exist_ok=True)

    lengths = np.load(SRC / "lengths.npy")
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    total = int(offsets[-1])
    nL = len(LAYERS)
    train_idx = {int(i) for i in np.load(SRC / "split_indices.npz")["train"]}
    records = [json.loads(l)["record"] for l in DATASET.open()]
    assert len(records) == len(lengths), f"records={len(records)} != lengths={len(lengths)}"
    print(
        f"records={len(records)} tokens={total} K={K} cap={a.cap} bs={a.batch} "
        f"merge_only={a.merge_only}",
        flush=True,
    )

    if not a.merge_only:
        prefill(a.batch, lengths, offsets, total, train_idx, records, nL)
    merge(a.cap, lengths, offsets, total, nL, a.batch)


if __name__ == "__main__":
    main()
