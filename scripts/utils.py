"""Shared data, split, prediction, and metric utilities for model training."""

import numpy as np

# These are defined alongside the model code for checkpoint compatibility and
# re-exported here as the stable training interface.
from models import (
    PAD,
    TOPICS,
    boundary_positions,
    boundary_report,
    build_model,
    collate,
    flatten_labeled,
    load_data,
    load_split,
    metrics,
    predict_sequences,
    split_indices,
)

__all__ = [
    "PAD",
    "TOPICS",
    "boundary_positions",
    "boundary_report",
    "build_model",
    "collate",
    "flatten_labeled",
    "load_data",
    "load_split",
    "metrics",
    "predict_sequences",
    "split_indices",
]


def response_topic_labels_only(
    examples, tokenizer, labels, *, neutral_label_id: int, pad_label: int = PAD
):
    """Mask prompt/template/neutral tokens while preserving token coordinates."""
    marker = tokenizer("<start_of_turn>model\n", add_special_tokens=False)["input_ids"]
    if not marker:
        raise ValueError("assistant marker tokenization is empty")
    cleaned = []
    for example, values in zip(examples, labels):
        values = np.asarray(values).copy()
        token_ids = example["input_ids"]
        matches = [
            index
            for index in range(len(token_ids) - len(marker) + 1)
            if token_ids[index : index + len(marker)] == marker
        ]
        if not matches:
            raise ValueError("assistant marker not found")
        values[: matches[-1] + len(marker)] = pad_label
        values[values == neutral_label_id] = pad_label
        cleaned.append(values)
    return cleaned


__all__.append("response_topic_labels_only")
