#!/usr/bin/env python3
"""Add prompt/response role IDs to an existing prompt+response SAE cache."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT.parent / "scripts" / "prefill-mvp"
sys.path.insert(0, str(LEGACY))
import train_gru_prefill as base  # noqa: E402


def build_role_ids(dataset: Path, cache: Path) -> Path:
    base.DATASET_PATH = str(dataset)
    tokenizer = AutoTokenizer.from_pretrained(base.MODEL_NAME)
    examples = base.load_and_parse_dataset(tokenizer)
    lengths = np.load(cache / "lengths.npy")
    labels = np.load(cache / "labels.npy", mmap_mode="r")
    if len(examples) != len(lengths):
        raise ValueError(f"parsed {len(examples)} records but cache has {len(lengths)}")
    marker = tokenizer("<start_of_turn>model\n", add_special_tokens=False)["input_ids"]
    role_rows = []
    offset = 0
    for index, (example, length) in enumerate(zip(examples, lengths)):
        token_ids = example["input_ids"]
        if len(token_ids) != int(length):
            raise ValueError(f"record {index}: token length differs from cache")
        matches = [
            position
            for position in range(len(token_ids) - len(marker) + 1)
            if token_ids[position : position + len(marker)] == marker
        ]
        if not matches:
            raise ValueError(f"record {index}: assistant marker not found")
        assistant_start = matches[-1] + len(marker)
        roles = np.ones(int(length), dtype=np.int8)
        roles[assistant_start:] = 2
        # Template/neutral tokens retain their surrounding role. Targets and
        # metrics still use labels.npy to exclude them from topic supervision.
        row_labels = labels[offset : offset + int(length)]
        if not np.any((row_labels != base.PAD_LABEL) & (roles == 1)):
            raise ValueError(f"record {index}: no labeled prompt tokens")
        if not np.any((row_labels != base.PAD_LABEL) & (roles == 2)):
            raise ValueError(f"record {index}: no labeled response tokens")
        role_rows.append(roles)
        offset += int(length)
    output = cache / "role_ids.npy"
    np.save(output, np.concatenate(role_rows))
    print(f"wrote {output} ({offset:,} tokens; 1=prompt, 2=response)")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    build_role_ids(args.dataset, args.cache)
