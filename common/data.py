"""Data plumbing shared by every stage: seeding, the canonical split, packed-cache loading, batching.

The packed feature caches hold one row per token; ``lengths.npy`` splits them into records and
``split_indices.npz`` holds the persisted record-level 70/15/15 split.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

# local imports
from models import PAD


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG used by the pipeline and optionally require deterministic ops."""
    # PYTHONHASHSEED is also inherited by any child Python processes. The current
    # process does not use unordered hash iteration for experiment decisions.
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        # Required by deterministic CUDA matrix multiplications on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def split_indices(record_count: int, seed: int = 42):
    """Return the canonical deterministic 70/15/15 response-level split."""
    order = np.random.RandomState(seed).permutation(record_count)
    train_end = int(0.70 * record_count)
    validation_end = train_end + int(0.15 * record_count)
    return order[:train_end], order[train_end:validation_end], order[validation_end:]


def load_data(cache: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return per-record feature and label arrays from a packed cache."""
    features = np.load(cache / "features.npy", mmap_mode="r")
    labels = np.load(cache / "labels.npy", mmap_mode="r")
    lengths = np.load(cache / "lengths.npy")
    offsets = np.r_[0, np.cumsum(lengths)]
    return [
        (
            features[offsets[index] : offsets[index + 1]],
            labels[offsets[index] : offsets[index + 1]].astype(np.int64),
        )
        for index in range(len(lengths))
    ]


def load_split(cache: Path, data: list[tuple[np.ndarray, np.ndarray]]):
    """Apply persisted train, validation, and test indices to cached records."""
    split_path = cache / "split_indices.npz"
    if not split_path.exists():
        raise FileNotFoundError(
            f"{split_path} is required; corrected runs never synthesize a validation-only split"
        )
    split = np.load(split_path)
    expected = {"train", "validation", "test"}
    if set(split.files) != expected:
        raise ValueError(f"split file must contain exactly {sorted(expected)}")
    partitions = {name: [data[index] for index in split[name]] for name in ("train", "validation", "test")}
    all_indices = np.concatenate([split[name] for name in partitions])
    if len(all_indices) != len(data) or len(np.unique(all_indices)) != len(data):
        raise ValueError("split indices must cover every record exactly once")
    return partitions


def collate(rows, input_features: int = 500):
    """Right-pad variable-length feature and label sequences into one batch."""
    length = max(len(features) for features, _ in rows)
    first_features, first_labels = rows[0]
    device = first_features.device if isinstance(first_features, torch.Tensor) else "cpu"
    x_batch = torch.zeros(len(rows), length, input_features, device=device)
    y_batch = torch.full((len(rows), length), PAD, dtype=torch.long, device=device)
    for index, (features, labels) in enumerate(rows):
        if isinstance(features, torch.Tensor):
            feature_tensor = features
        else:
            feature_tensor = torch.from_numpy(np.array(features, dtype=np.float32, copy=True))
        if isinstance(labels, torch.Tensor):
            label_tensor = labels
        else:
            label_tensor = torch.from_numpy(labels)
        x_batch[index, : len(features)] = feature_tensor
        y_batch[index, : len(labels)] = label_tensor
    return x_batch, y_batch


def response_topic_labels_only(examples, tokenizer, labels, *, neutral_label_id: int, pad_label: int = PAD):
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
