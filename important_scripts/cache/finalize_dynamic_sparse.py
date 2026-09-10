#!/usr/bin/env python3
"""Streaming CSR finalize for a dynamic sparse cache — writes the output straight to DISK.

Given the on-disk topk stores from a build's prefill (`_topk_idx.dat`, `_topk_val.dat`, shape
(total, L, K)) and the stage-1 SRC cache (lengths/labels/roles/split), this:

  1. recomputes per-layer feature frequency over TRAIN tokens by streaming the topk in chunks
     (only tiny integer counts kept in RAM),
  2. picks each layer's union columns (cap to the most-frequent CAP),
  3. streams the topk again and writes the CSR-by-token `col`/`val`/`tok_ptr` directly into
     on-disk .npy memmaps — so nothing large (neither the 80 GiB topk nor the ~55 GiB output)
     is ever resident in RAM. Peak RAM ~ one chunk (a few GiB).

Output (a dir the low-mem trainer mmaps, so N parallel trainers share one physical copy):
  col.npy (int32), val.npy (fp16), tok_ptr.npy (int64), labels.npy, roles.npy, lengths.npy,
  split_indices.npz, selected_features.npz, meta.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

# local imports
from important_scripts.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT


CH = 400_000  # tokens per streaming chunk


def main() -> None:
    """Stream the on-disk topk stores into a CSR-by-token sparse cache (col/val/tok_ptr + sidecars)."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dir",
        required=True,
        help="cache dir holding _topk_idx.dat/_topk_val.dat; output written here",
    )
    ap.add_argument(
        "--src", required=True, help="stage-1 cache dir (lengths/labels/roles/split/metadata)"
    )
    ap.add_argument("--layers", required=True, help="e.g. 16-25 or 24-33")
    ap.add_argument("--k", type=int, default=150)
    ap.add_argument("--cap", type=int, default=6000)
    ap.add_argument("--d-sae", type=int, default=16384)
    a = ap.parse_args()
    lo, hi = a.layers.split("-")
    layers = list(range(int(lo), int(hi) + 1))
    nL, K, cap, d_sae = len(layers), a.k, a.cap, a.d_sae
    D = ROOT / a.dir
    SRC = ROOT / a.src
    t0 = time.time()

    lengths = np.load(SRC / "lengths.npy")
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    total = int(offsets[-1])
    sp = np.load(SRC / "split_indices.npz")
    train_recs = sp["train"].astype(np.int64)
    # per-token train mask (8.85M bool ≈ 9 MB)
    is_train = np.zeros(total, dtype=bool)
    for r in train_recs:
        is_train[int(offsets[r]) : int(offsets[r + 1])] = True

    topk_idx = np.memmap(D / "_topk_idx.dat", mode="r", dtype=np.int32, shape=(total, nL, K))
    topk_val = np.memmap(D / "_topk_val.dat", mode="r", dtype=np.float16, shape=(total, nL, K))
    print(f"total_tokens={total} layers={layers} K={K} cap={cap}", flush=True)

    # ---- pass 1: per-layer frequency over TRAIN tokens ----
    freq = np.zeros((nL, d_sae), dtype=np.int64)
    for c0 in range(0, total, CH):
        c1 = min(c0 + CH, total)
        idx_c = np.asarray(topk_idx[c0:c1])  # (n,L,K)
        tr = is_train[c0:c1]
        if tr.any():
            for li in range(nL):
                col = idx_c[tr, li, :]
                v = col[col >= 0]
                if v.size:
                    freq[li] += np.bincount(v, minlength=d_sae)
    print(f"[freq] done {time.time() - t0:.0f}s", flush=True)

    # ---- union columns per layer (cap to most-frequent) ----
    widths, natural, luts, selected = [], [], [], {}
    for li, L in enumerate(layers):
        present = np.where(freq[li] > 0)[0]
        natural.append(len(present))
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
    sae_index = np.concatenate([selected[f"layer_{L}"] for L in layers]).astype(np.int64)
    layer_cols = {
        int(L): [int(layer_off[li]), int(layer_off[li + 1])] for li, L in enumerate(layers)
    }
    print(
        f"natural widths={natural}\nkept widths={widths} W_total={W_total} "
        f"(cap hit {sum(n > cap for n in natural)}/{nL})",
        flush=True,
    )

    def map_layer(idx: np.ndarray, li: int) -> tuple[np.ndarray, np.ndarray]:
        """Map raw SAE feature ids to this layer's union-column indices.

        Returns (mapped_columns, keep_mask); keep_mask is False where the feature was padding
        (idx < 0) or was dropped by the per-layer frequency cap (lut == -1).
        """
        valid = idx >= 0
        mapped = luts[li][np.where(valid, idx, 0)]
        return mapped, valid & (mapped >= 0)

    # ---- pass 2: nnz per token -> tok_ptr ----
    nnz_per_tok = np.zeros(total, dtype=np.int64)
    fired = 0
    for c0 in range(0, total, CH):
        c1 = min(c0 + CH, total)
        idx_c = np.asarray(topk_idx[c0:c1])
        kc = np.zeros(c1 - c0, dtype=np.int64)
        for li in range(nL):
            _, ok = map_layer(idx_c[:, li, :], li)
            fired += int((idx_c[:, li, :] >= 0).sum())
            kc += ok.sum(axis=1)
        nnz_per_tok[c0:c1] = kc
    total_nnz = int(nnz_per_tok.sum())
    tok_ptr = np.concatenate(([0], np.cumsum(nnz_per_tok)))
    print(
        f"[nnz] total_nnz={total_nnz} mean/token={total_nnz / total:.0f} kept={100 * total_nnz / max(fired, 1):.2f}% "
        f"{time.time() - t0:.0f}s",
        flush=True,
    )

    # ---- pass 3: stream col/val straight to on-disk .npy memmaps ----
    col = np.lib.format.open_memmap(D / "col.npy", mode="w+", dtype=np.int32, shape=(total_nnz,))
    val = np.lib.format.open_memmap(D / "val.npy", mode="w+", dtype=np.float16, shape=(total_nnz,))
    w = 0
    for c0 in range(0, total, CH):
        c1 = min(c0 + CH, total)
        n = c1 - c0
        idx_c = np.asarray(topk_idx[c0:c1])
        val_c = np.asarray(topk_val[c0:c1])
        g = np.full((n, nL, K), -1, dtype=np.int32)
        for li in range(nL):
            mapped, ok = map_layer(idx_c[:, li, :], li)
            g[:, li, :][ok] = (layer_off[li] + mapped[ok]).astype(np.int32)
        gflat = g.reshape(n, -1)
        vflat = val_c.reshape(n, -1)
        m = gflat >= 0
        cnt = int(m.sum())
        col[w : w + cnt] = gflat[m]
        val[w : w + cnt] = vflat[m]
        w += cnt
    assert w == total_nnz, f"{w} != {total_nnz}"
    col.flush()
    val.flush()
    del col, val
    print(f"[csr] wrote col.npy/val.npy {time.time() - t0:.0f}s", flush=True)

    # ---- small sidecars ----
    np.save(D / "tok_ptr.npy", tok_ptr)
    np.save(D / "labels.npy", np.load(SRC / "labels.npy").astype(np.int64))
    np.save(D / "roles.npy", np.load(SRC / "role_ids.npy").astype(np.int64))
    np.save(D / "lengths.npy", lengths.astype(np.int64))
    np.savez(D / "split_indices.npz", **{k: sp[k] for k in sp.files})
    np.savez(D / "selected_features.npz", **selected)
    np.save(D / "sae_index.npy", sae_index)
    (D / "meta.json").write_text(
        json.dumps(
            {
                "width": W_total,
                "widths": widths,
                "natural_widths": natural,
                "K": K,
                "cap": cap,
                "layer_cols": layer_cols,
                "layers": layers,
                "nnz": total_nnz,
                "tokens": total,
                "mean_nnz_per_token": total_nnz / total,
            },
            indent=2,
        )
        + "\n"
    )
    # drop the big temp topk stores
    for p in (D / "_topk_idx.dat", D / "_topk_val.dat"):
        try:
            os.remove(p)
        except OSError:
            pass
    sz = sum((D / f).stat().st_size for f in ("col.npy", "val.npy", "tok_ptr.npy")) / 1e9
    print(
        f"DONE width={W_total} nnz={total_nnz} cache≈{sz:.1f} GB  {time.time() - t0:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
