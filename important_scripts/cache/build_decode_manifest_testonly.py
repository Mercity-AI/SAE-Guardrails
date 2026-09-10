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

  python build_decode_manifest_testonly.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# local imports
from important_scripts.paths import PROJECT_ROOT
from important_scripts.model.models import TOPICS

ROOT = PROJECT_ROOT

DATASET = ROOT / "runs/GPT_5.6_10k_2108/dataset.final.jsonl"
SPLIT = (
    ROOT / "cache/sae1500_10k_gemma4b_prompt_response/split_indices.npz"
)  # canonical, cache-invariant
OUT = ROOT / "ablation/gemma_arm/manifest_4b_10k_test.jsonl"
N_SINGLE, N_MULTI, SEED = 200, 800, 42


def main() -> None:
    """Sample 200 single- + 800 multi-topic TEST records (seed 42) into the leakage-free manifest."""
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
