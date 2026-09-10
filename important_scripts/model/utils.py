"""Shared cache, data, prediction, and metric utilities for model training."""

import numpy as np
from pathlib import Path

# local imports
from important_scripts.model.decoder_utils import (
    CAUSAL_DECODERS,
    OFFLINE_DECODERS,
    decode_hysteresis,
    decode_logits,
    decode_minimum_duration,
    decode_raw,
    decode_transition_penalty,
    decode_viterbi,
)

# These are defined alongside the model code for checkpoint compatibility and
# re-exported here as the stable training interface.
from important_scripts.model.models import (
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
    seed_everything,
    split_indices,
    topic_overlap,
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
    "seed_everything",
    "split_indices",
    "topic_overlap",
    "CAUSAL_DECODERS",
    "OFFLINE_DECODERS",
    "decode_hysteresis",
    "decode_logits",
    "decode_minimum_duration",
    "decode_raw",
    "decode_transition_penalty",
    "decode_viterbi",
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


def cache_status(cache: Path, require_roles: bool = False) -> dict[str, bool]:
    """Return the presence of every file required by a packed feature cache."""
    names = ["features.npy", "labels.npy", "lengths.npy", "split_indices.npz"]
    if require_roles:
        names.append("role_ids.npy")
    return {name: (cache / name).exists() for name in names}


def require_cache(cache: Path, require_roles: bool = False) -> None:
    """Validate a cache and log explicitly when the existing cache is reused."""
    status = cache_status(cache, require_roles=require_roles)
    missing = sorted(name for name, exists in status.items() if not exists)
    if missing:
        raise FileNotFoundError(f"cache is incomplete at {cache}; missing {missing}")
    print(f"Using existing cache: {cache}", flush=True)


def load_cached_partitions(cache: Path):
    """Load an existing cache and return its persisted dataset partitions."""
    require_cache(cache)
    data = load_data(cache)
    return load_split(cache, data)


__all__.extend(["cache_status", "require_cache", "load_cached_partitions"])
