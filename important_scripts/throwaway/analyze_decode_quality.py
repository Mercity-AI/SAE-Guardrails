#!/usr/bin/env python3
"""Decode cross-style analysis: GPT-trained detectors tested on GEMMA-generated responses.

Central question: can a detector trained purely on GPT's writing still land the topics when
Gemma writes them (different style)? We report, broken down by #prompt-topics (1/2/3/4):
  - purity        : share of response tokens labelled as one of the prompt's topics (on-topic mass)
  - dominant_acc  : majority response label is one of the prompt topics
  - coverage      : mean fraction of the prompt's topics that get >=10% of tokens
                    (a JOINT gen x detector proxy: needs Gemma to write it AND the detector to see it)
  - full_cover    : fraction of records where EVERY prompt topic reaches >=10%
  - segments      : mean # contiguous topic runs >=20 tokens (does the detector track topic switches?)
  - switch_rate   : label changes / tokens (lower = stays on course)
  - hit_cap       : fraction whose generation length hit the token cap (truncated)

Usage: python scripts/analyze_decode_quality.py [static.jsonl dynamic.jsonl]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, "modeling")
from models import TOPICS

DEFAULTS = [
    ("STATIC SAE-150", "ablation/gemma_arm/decode_1b_static_pr/per_record.jsonl"),
    ("DYNAMIC-150", "results/dyn150_10k_1b_pr/decode_dynamic_per_record.jsonl"),
]
TAU = 0.10  # a topic counts as "covered" if it holds >=10% of response tokens
MINRUN = 20  # a contiguous run of >=20 tokens counts as a topic "segment"


def rec_stats(rec, fam, cap):
    """Per-record on-topic purity / dominant-topic / coverage stats for one detector."""
    true = [TOPICS.index(t) for t in rec["topics"]]
    pred = np.asarray(rec["pred"][fam])
    n = len(pred)
    if n == 0:
        return None
    on = np.isin(pred, true)
    shares = {t: float((pred == t).mean()) for t in true}
    covered = sum(s >= TAU for s in shares.values())
    # segments: contiguous runs >= MINRUN tokens
    runs = []
    for x in pred:
        if runs and runs[-1][0] == x:
            runs[-1][1] += 1
        else:
            runs.append([x, 1])
    segs = sum(1 for _, c in runs if c >= MINRUN)
    switches = int((pred[1:] != pred[:-1]).sum())
    gen = rec.get("generated_tokens", n)
    return {
        "n_true": len(true),
        "purity": float(on.mean()),
        "maj_in_true": (Counter(pred.tolist()).most_common(1)[0][0] in true),
        "coverage": covered / len(true),
        "full_cover": (covered == len(true)),
        "segments": segs,
        "switch_rate": switches / n,
        "hit_cap": (gen >= cap),
    }


def report(name, path):
    """Aggregate and print the cross-style decode-quality table, broken down by prompt-topic count."""
    recs = [json.loads(l) for l in open(path)]
    fams = list(recs[0]["pred"].keys())
    cap = max(r.get("generated_tokens", 0) for r in recs)  # infer cap from the run
    print(
        f"\n{'=' * 100}\n{name}  ({len(recs)} recs, inferred cap {cap}, "
        f"hit-cap {np.mean([r.get('generated_tokens', 0) >= cap for r in recs]) * 100:.1f}%)\n{'=' * 100}"
    )
    for fam in fams:
        S = [s for r in recs if (s := rec_stats(r, fam, cap))]
        print(f"\n-- {fam} --")
        print(
            f"{'#topics':>7} {'n':>5} {'purity':>7} {'domAcc':>7} {'coverage':>9} "
            f"{'fullCov':>8} {'segs':>6} {'switch%':>8} {'hitCap%':>8}"
        )
        for k in [1, 2, 3, 4]:
            g = [s for s in S if s["n_true"] == k]
            if not g:
                continue
            print(
                f"{k:>7} {len(g):>5} {np.mean([s['purity'] for s in g]):>7.3f} "
                f"{np.mean([s['maj_in_true'] for s in g]):>7.3f} "
                f"{np.mean([s['coverage'] for s in g]):>9.3f} "
                f"{np.mean([s['full_cover'] for s in g]):>8.3f} "
                f"{np.mean([s['segments'] for s in g]):>6.2f} "
                f"{np.mean([s['switch_rate'] for s in g]) * 100:>8.2f} "
                f"{np.mean([s['hit_cap'] for s in g]) * 100:>8.1f}"
            )
        allS = S
        print(
            f"{'ALL':>7} {len(allS):>5} {np.mean([s['purity'] for s in allS]):>7.3f} "
            f"{np.mean([s['maj_in_true'] for s in allS]):>7.3f} "
            f"{np.mean([s['coverage'] for s in allS]):>9.3f} "
            f"{np.mean([s['full_cover'] for s in allS]):>8.3f} "
            f"{np.mean([s['segments'] for s in allS]):>6.2f} "
            f"{np.mean([s['switch_rate'] for s in allS]) * 100:>8.2f} "
            f"{np.mean([s['hit_cap'] for s in allS]) * 100:>8.1f}"
        )


def main():
    """Run the GPT-trained-detector-on-Gemma-text decode-quality analysis."""
    args = sys.argv[1:]
    pairs = (
        DEFAULTS
        if not args
        else list(zip([f"ARM{i}" for i in range(len(args))], args, strict=False))
    )
    for name, path in pairs:
        if Path(path).exists():
            report(name, path)
        else:
            print(f"[skip] {path} missing")


if __name__ == "__main__":
    main()
