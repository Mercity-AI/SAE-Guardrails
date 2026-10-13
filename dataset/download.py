#!/usr/bin/env python3
"""Rebuild the generation run's ``dataset.final.jsonl`` from the published Hub dataset.

Every cache builder and decode script reads ``runs/GPT_5.6_10k_2108/dataset.final.jsonl``: one
Simula row per line, with the tagged conversation under ``"record"``, in the original generation
order. The canonical split is a permutation of those line numbers, so order matters. The Hub
dataset keeps each record's original line number in ``row``; this script joins the three splits,
sorts on it and writes the rows back in Simula's shape.

    python dataset/download.py
    python dataset/download.py --out runs/GPT_5.6_10k_2108/dataset.final.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

# local imports
from common.data import split_indices

REPO = "Mercity/topic-shift-10k"
RECORD_KEYS = ("prompt", "response", "topics")
ROW_KEYS = (
    "id",
    "taxonomy_mix",
    "strategy_id",
    "meta_prompt",
    "generator_model",
    "critic_verdicts",
    "created_at",
)


def main() -> None:
    """Download all three splits, restore generation order and check the split before writing."""
    parser = argparse.ArgumentParser(description="Rebuild dataset.final.jsonl from the Hub dataset.")
    parser.add_argument("--out", type=Path, default=Path("runs/GPT_5.6_10k_2108/dataset.final.jsonl"))
    parser.add_argument("--token", default=None, help="Hugging Face token (defaults to the logged-in one)")
    args = parser.parse_args()

    splits = load_dataset(REPO, token=args.token)
    rows = sorted(
        (r for name in ("train", "validation", "test") for r in splits[name]), key=lambda r: r["row"]
    )
    assert [r["row"] for r in rows] == list(range(len(rows))), "row numbers are not a contiguous 0..N-1"

    # The published splits must be exactly the canonical permutation of these line numbers.
    train_idx, val_idx, test_idx = split_indices(len(rows))
    for name, idx in (("train", train_idx), ("validation", val_idx), ("test", test_idx)):
        assert sorted(splits[name]["row"]) == sorted(int(i) for i in idx), f"{name} split does not match"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            line = {k: r[k] for k in ROW_KEYS} | {"record": {k: r[k] for k in RECORD_KEYS}}
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows):,} records to {args.out} (split checked against split_indices)")


if __name__ == "__main__":
    main()
