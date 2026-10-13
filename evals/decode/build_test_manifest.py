#!/usr/bin/env python3
"""Build a LEAKAGE-FREE decode manifest drawn only from the canonical TEST split.

The original ablation/gemma_arm/manifest_4b_10k.jsonl was sampled from the whole 10k dataset:
725 train / 145 val / 130 test. Every decode arm that used it scored detectors on prompts the
detectors were trained on. This rebuilds the manifest using ONLY the held-out test records, so
the decode ablation is honest.

Same schema/convention as before: {index, topics, single_topic, assumed_label_id, prompt},
assumed_label_id = TOPICS.index(topics[0]), prompt keeps its <Topic>..</Topic> tags. Same target
composition (200 single-topic + 800 multi-topic = 1000), sampled deterministically (seed 42).
Index == line number in runs/GPT_5.6_10k_2108/dataset.final.jsonl, and the canonical split is
identical across all 10k caches (verified), so the same manifest is clean for 1B and 4B.

    python evals/decode/build_test_manifest.py [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

# local imports
from common.paths import PROJECT_ROOT
from models import TOPICS

ROOT = PROJECT_ROOT

DATASET = ROOT / "runs/GPT_5.6_10k_2108/dataset.final.jsonl"
SPLIT = ROOT / "cache/sae1500_10k_gemma4b_prompt_response/split_indices.npz"  # canonical, cache-invariant
OUT = ROOT / "ablation/gemma_arm/manifest_4b_10k_test.jsonl"
N_SINGLE, N_MULTI, SEED = 200, 800, 42


def _resolve(path: str) -> Path:
    """Resolve a path against the data root unless it is already absolute."""
    return Path(path) if Path(path).is_absolute() else ROOT / path


def main() -> None:
    """Sample 200 single- + 800 multi-topic TEST records (seed 42) into the leakage-free manifest."""
    global DATASET, SPLIT, OUT, N_SINGLE, N_MULTI, SEED
    parser = argparse.ArgumentParser(
        description="Build the leakage-free decode manifest from the test split."
    )
    parser.add_argument("--dataset", default=str(DATASET), help="dataset.final.jsonl")
    parser.add_argument("--split", default=str(SPLIT), help="any 10k cache's split_indices.npz")
    parser.add_argument("--out", default=str(OUT), help="manifest to write")
    parser.add_argument("--n-single", type=int, default=N_SINGLE, help="single-topic records to sample")
    parser.add_argument("--n-multi", type=int, default=N_MULTI, help="multi-topic records to sample")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    DATASET, SPLIT, OUT = _resolve(args.dataset), _resolve(args.split), _resolve(args.out)
    N_SINGLE, N_MULTI, SEED = args.n_single, args.n_multi, args.seed

    dataset = [json.loads(line) for line in DATASET.open()]
    test_idx = sorted(int(i) for i in np.load(SPLIT)["test"])

    single, multi = [], []
    for i in test_idx:
        topics = dataset[i]["record"]["topics"]
        (single if len(topics) == 1 else multi).append(i)
    print(f"test split: {len(test_idx)} records | single={len(single)} multi={len(multi)}")
    if len(single) < N_SINGLE or len(multi) < N_MULTI:
        raise SystemExit(
            f"not enough test records: need {N_SINGLE}/{N_MULTI}, have {len(single)}/{len(multi)}"
        )

    rng = np.random.default_rng(SEED)
    pick_single = sorted(rng.choice(single, N_SINGLE, replace=False).tolist())
    pick_multi = sorted(rng.choice(multi, N_MULTI, replace=False).tolist())

    records = []
    for i in sorted(pick_single + pick_multi):
        rec = dataset[i]["record"]
        topics = rec["topics"]
        records.append(
            {
                "index": i,
                "topics": topics,
                "single_topic": len(topics) == 1,
                "assumed_label_id": TOPICS.index(topics[0]),
                "prompt": rec["prompt"],
            }
        )

    # sanity: every index is in the held-out test set
    test_set = set(test_idx)
    assert all(r["index"] in test_set for r in records), "leak: a manifest index is not in test"

    with OUT.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    from collections import Counter

    tc = Counter(len(r["topics"]) for r in records)
    print(f"wrote {OUT}")
    print(
        f"  {len(records)} records | single={sum(r['single_topic'] for r in records)} "
        f"multi={sum(not r['single_topic'] for r in records)} | topics-per-record={dict(sorted(tc.items()))}"
    )
    print("  ALL indices in canonical TEST split — no train/val leakage.")


if __name__ == "__main__":
    main()
