#!/usr/bin/env python3
"""Compare decode scores between two per_record.jsonl files (e.g. 1000-subset vs full-1500 test).

Uses the exact scoring convention shared by the PR decode scripts (decode_raw_1b_pr.py,
exp_decode_dynamic.py): per detector, single-topic assumed-label token accuracy, and
multi-topic set recall / precision. Prints each metric for both files and the delta in
percentage POINTS, and flags whether any |delta| exceeds a threshold (default 2.0 pp).

  python compare_decode_1000_1500.py A.jsonl B.jsonl [--threshold 2.0]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path("/workspace/scope/modeling")))
from models import TOPICS


def score(path):
    """Score one per_record.jsonl with the shared decode convention (assumed-label accuracy + set recall/precision)."""
    recs = [json.loads(l) for l in open(path)]
    dets = sorted({k for r in recs for k in r.get("pred", {})})
    n_single = sum(r["single_topic"] for r in recs)
    n_multi = len(recs) - n_single
    out = {"_n": len(recs), "_n_single": n_single, "_n_multi": n_multi}
    for name in dets:
        single_acc, two_rec, two_prec = [], [], []
        for r in recs:
            if not r["pred"].get(name):
                continue
            pred = np.asarray(r["pred"][name])
            if r["single_topic"]:
                single_acc.append(float((pred == r["assumed_label_id"]).mean()))
            else:
                truth = {TOPICS.index(t) for t in r["topics"]}
                pset = {int(v) for v in pred}
                two_rec.append(len(pset & truth) / len(truth))
                two_prec.append(len(pset & truth) / max(1, len(pset)))
        out[name] = {
            "single_acc": float(np.mean(single_acc)) if single_acc else None,
            "two_recall": float(np.mean(two_rec)) if two_rec else None,
            "two_prec": float(np.mean(two_prec)) if two_prec else None,
        }
    return out, dets


def main():
    """Compare decode metrics between two per_record.jsonl files and flag large deltas."""
    ap = argparse.ArgumentParser()
    ap.add_argument("file_a", help="baseline per_record.jsonl (e.g. @1000)")
    ap.add_argument("file_b", help="comparison per_record.jsonl (e.g. @1500)")
    ap.add_argument(
        "--threshold", type=float, default=2.0, help="pp threshold for 'across the board'"
    )
    a = ap.parse_args()
    A, dets = score(a.file_a)
    B, _ = score(a.file_b)

    print(f"A = {a.file_a}   n={A['_n']} (single {A['_n_single']} / multi {A['_n_multi']})")
    print(f"B = {a.file_b}   n={B['_n']} (single {B['_n_single']} / multi {B['_n_multi']})")
    print(f"threshold = {a.threshold:.1f} pp\n")
    hdr = f"{'detector':12s} {'metric':11s} {'A':>8s} {'B':>8s} {'Δpp':>8s}"
    print(hdr)
    print("-" * len(hdr))
    max_abs = 0.0
    any_over = False
    for name in dets:
        for m in ("single_acc", "two_recall", "two_prec"):
            va, vb = A[name][m], B[name][m]
            if va is None or vb is None:
                print(f"{name:12s} {m:11s} {'n/a':>8s} {'n/a':>8s} {'--':>8s}")
                continue
            d = (vb - va) * 100.0
            max_abs = max(max_abs, abs(d))
            flag = "  <-- >thr" if abs(d) > a.threshold else ""
            if abs(d) > a.threshold:
                any_over = True
            print(f"{name:12s} {m:11s} {va * 100:8.2f} {vb * 100:8.2f} {d:+8.2f}{flag}")
    print("-" * len(hdr))
    print(f"max |Δ| = {max_abs:.2f} pp")
    verdict = (
        "REDO at 1500 (a metric moved > threshold)"
        if any_over
        else f"KEEP 1000 (all metrics within {a.threshold:.1f} pp)"
    )
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
