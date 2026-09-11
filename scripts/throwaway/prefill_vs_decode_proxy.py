#!/usr/bin/env python3
"""Prefill (GPT response, in-distribution) vs Decode (Gemma response, cross-style) on the SAME
proxy metrics, per detector, by #prompt-topics. Proxies use the prompt's topic set (the only
thing decode has), so the prefill->decode gap isolates the cross-style + generation shift.

Prefill predictions are computed here over the static SAE-1500 test split (teacher-forced GPT
features already in the cache); decode numbers are read from the per_record logs.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "modeling")
from models import TOPICS, build_model

ROOT = Path("/workspace/scope")
CACHE = ROOT / "cache/sae1500_10k_1b_pr"
DET = {"GRU": "gru_lr3e4", "Transformer": "transformer_wide", "ConvNeXt": "convnext_window505"}
BASE = ROOT / "results/gemma1b_baselines_sae150_10k_pr"
DATASET = ROOT / "runs/GPT_5.6_10k_2108/dataset.final.jsonl"
DECODE_STATIC = ROOT / "ablation/gemma_arm/decode_1b_static_pr/per_record.jsonl"
RESP, PAD, TAU = 2, -100, 0.10
DEV = "cuda"


def proxies(pred, true):
    """Compute the prompt-topic proxy metrics for one set of per-token predictions."""
    pred = np.asarray(pred)
    n = len(pred)
    on = np.isin(pred, true)
    shares = [float((pred == t).mean()) for t in true]
    return {
        "n_true": len(true),
        "purity": float(on.mean()),
        "maj": (Counter(pred.tolist()).most_common(1)[0][0] in true),
        "cover": sum(s >= TAU for s in shares) / len(true),
        "full": (all(s >= TAU for s in shares)),
    }


def agg(rows):
    """Aggregate proxy metrics across records."""
    import numpy as np

    def by(k):
        out = {}
        for tc in [1, 2, 3, 4, "ALL"]:
            g = [r for r in rows if (tc == "ALL" or r["n_true"] == tc)]
            out[tc] = np.mean([r[k] for r in g]) if g else float("nan")
        return out

    return {k: by(k) for k in ["purity", "maj", "cover", "full"]}


def prefill_rows():
    """Build prefill (GPT-response) prediction rows from the static SAE-1500 test cache."""
    feats = np.load(CACHE / "features.npy", mmap_mode="r")
    labels = np.load(CACHE / "labels.npy", mmap_mode="r")
    roles = np.load(CACHE / "role_ids.npy", mmap_mode="r")
    lengths = np.load(CACHE / "lengths.npy")
    off = np.concatenate(([0], np.cumsum(lengths)))
    test = np.load(CACHE / "split_indices.npz")["test"]
    ds = [json.loads(l) for l in DATASET.open()]
    dets = {}
    for name, sub in DET.items():
        p = torch.load(BASE / sub / "checkpoint_best.pt", map_location="cpu", weights_only=False)
        m = build_model(p["architecture"], p["model"])
        m.load_state_dict(p["state_dict"])
        dets[name] = m.to(DEV).eval()
    rows = {name: [] for name in DET}
    with torch.inference_mode():
        for i in test:
            i = int(i)
            a, b = int(off[i]), int(off[i + 1])
            role = np.asarray(roles[a:b])
            keep = role == RESP
            if keep.sum() == 0:
                continue
            topics = (
                [TOPICS.index(t) for t in eval(ds[i]["record"]["topics"])]
                if isinstance(ds[i]["record"]["topics"], str)
                else [TOPICS.index(t) for t in ds[i]["record"]["topics"]]
            )
            x = torch.from_numpy(np.asarray(feats[a:b]).astype(np.float32))[None].to(DEV)
            for name, m in dets.items():
                pred = m(x)[0].argmax(-1).cpu().numpy()[keep]
                rows[name].append(proxies(pred, topics))
    return rows


def decode_rows():
    """Read decode (Gemma-response) prediction rows from the per_record logs."""
    recs = [json.loads(l) for l in DECODE_STATIC.open()]
    rows = {name: [] for name in DET}
    for r in recs:
        true = [TOPICS.index(t) for t in r["topics"]]
        for name in DET:
            if name in r["pred"]:
                rows[name].append(proxies(r["pred"][name], true))
    return rows


def main():
    """Compare prefill vs decode on the same proxy metrics, split by prompt-topic count."""
    pf = prefill_rows()
    dc = decode_rows()
    for name in DET:
        A, B = agg(pf[name]), agg(dc[name])
        print(f"\n===== {name}:  PREFILL (GPT) vs DECODE (Gemma), by #topics =====")
        print(f"{'metric':>9} | " + " ".join(f"{tc:>6}" for tc in [1, 2, 3, 4, "ALL"]))
        for k, lbl in [
            ("purity", "purity"),
            ("maj", "domAcc"),
            ("cover", "coverage"),
            ("full", "fullCov"),
        ]:
            pr = "  ".join(
                f"{A[k][tc]:.3f}" if not np.isnan(A[k][tc]) else "  -  "
                for tc in [1, 2, 3, 4, "ALL"]
            )
            dd = "  ".join(
                f"{B[k][tc]:.3f}" if not np.isnan(B[k][tc]) else "  -  "
                for tc in [1, 2, 3, 4, "ALL"]
            )
            print(f"{lbl:>9} P {pr}")
            print(f"{'':>9} D {dd}")


if __name__ == "__main__":
    main()
