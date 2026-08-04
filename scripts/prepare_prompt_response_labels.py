#!/usr/bin/env python3
"""Reuse train-selected SAE features while retaining tagged prompt/response labels."""
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "scripts" / "prefill-mvp"))
import train_gru_prefill as base
DEFAULT_SOURCE = ROOT / "cache" / "sae500_1500_train_selected"
DEFAULT_OUT = ROOT / "cache" / "sae500_1500_prompt_response_train_selected"
DEFAULT_DATASET = ROOT / "runs" / "1500" / "dataset.final.jsonl"

def link(source, destination):
    if not destination.exists():
        os.link(source, destination)
    elif source.stat().st_ino != destination.stat().st_ino:
        raise RuntimeError(f"Refusing to replace {destination}")

def main(dataset=DEFAULT_DATASET, source=DEFAULT_SOURCE, output=DEFAULT_OUT):
    base.DATASET_PATH = str(dataset)
    tok = AutoTokenizer.from_pretrained(base.MODEL_NAME)
    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    examples = base.load_and_parse_dataset(tok)
    cached_lengths = np.load(source / "lengths.npy")
    parsed_lengths = np.asarray([len(x["input_ids"]) for x in examples], dtype=np.int32)
    if not np.array_equal(cached_lengths, parsed_lengths):
        raise RuntimeError("Parsed lengths do not match cached features")
    neutral = base.LABEL2ID[base.NEUTRAL_LABEL]
    labels = []
    for example in examples:
        values = np.asarray(example["labels"], dtype=np.int16)
        values[values == neutral] = base.PAD_LABEL
        labels.append(values)
    output.mkdir(parents=True, exist_ok=True)
    for name in (
        "features.npy",
        "lengths.npy",
        "selected_features.npz",
        "split_indices.npz",
    ):
        link(source / name, output / name)
    np.save(output / "labels.npy", np.concatenate(labels).astype(np.int16))
    metadata = json.loads((source / "metadata.json").read_text())
    metadata["dataset"] = str(Path(dataset).resolve())
    metadata["supervision"] = "prompt and response topic tokens; template/neutral masked"
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    old = np.load(source / "labels.npy", mmap_mode="r")
    new = np.load(output / "labels.npy", mmap_mode="r")
    old_n = int((old != base.PAD_LABEL).sum())
    new_n = int((new != base.PAD_LABEL).sum())
    print(f"examples: {len(examples)}")
    print(f"response-only labeled tokens: {old_n}")
    print(f"prompt+response labeled tokens: {new_n}")
    print(f"added prompt tokens: {new_n-old_n}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    main(args.dataset, args.source, args.output)
