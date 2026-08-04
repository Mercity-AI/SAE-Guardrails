"""Shared causal TCN models, data utilities, and original-coordinate metrics."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score

TOPICS = [
    "Enterprise documents",
    "General news & content",
    "Customer service",
    "Legal",
    "Financial",
    "HR & people operations",
    "Healthcare",
]
PAD = -100


def split_indices(record_count: int, seed: int = 42):
    """Return the canonical deterministic 70/15/15 response-level split."""
    order = np.random.RandomState(seed).permutation(record_count)
    train_end = int(0.70 * record_count)
    validation_end = train_end + int(0.15 * record_count)
    return order[:train_end], order[train_end:validation_end], order[validation_end:]


def load_data(cache: Path) -> list[tuple[np.ndarray, np.ndarray]]:
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
    split_path = cache / "split_indices.npz"
    if not split_path.exists():
        raise FileNotFoundError(
            f"{split_path} is required; corrected runs never synthesize a validation-only split"
        )
    split = np.load(split_path)
    expected = {"train", "validation", "test"}
    if set(split.files) != expected:
        raise ValueError(f"split file must contain exactly {sorted(expected)}")
    partitions = {
        name: [data[index] for index in split[name]]
        for name in ("train", "validation", "test")
    }
    all_indices = np.concatenate([split[name] for name in partitions])
    if len(all_indices) != len(data) or len(np.unique(all_indices)) != len(data):
        raise ValueError("split indices must cover every record exactly once")
    return partitions


def collate(rows, input_features: int = 500):
    length = max(len(features) for features, _ in rows)
    x_batch = torch.zeros(len(rows), length, input_features)
    y_batch = torch.full((len(rows), length), PAD, dtype=torch.long)
    for index, (features, labels) in enumerate(rows):
        x_batch[index, : len(features)] = torch.from_numpy(
            np.asarray(features, dtype=np.float32)
        )
        y_batch[index, : len(labels)] = torch.from_numpy(labels)
    return x_batch, y_batch


class CausalChannelNorm(nn.Module):
    """Normalize channels at each token, never across time or padding positions."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, values):
        return self.norm(values.transpose(1, 2)).transpose(1, 2)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, kernel: int, dropout: float):
        super().__init__()
        self.padding = (kernel - 1) * dilation
        self.conv = nn.Conv1d(
            channels, channels, kernel, padding=self.padding, dilation=dilation
        )
        self.norm = CausalChannelNorm(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, values):
        convolved = self.conv(values)
        if self.padding:
            convolved = convolved[..., : -self.padding]
        return values + self.drop(torch.relu(self.norm(convolved)))


class BaselineTCN(nn.Module):
    def __init__(
        self,
        input_features: int = 500,
        classes: int = len(TOPICS),
        channels: int = 128,
        depth: int = 6,
        kernel: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input = nn.Conv1d(input_features, channels, 1)
        self.blocks = nn.Sequential(
            *(ResidualBlock(channels, 2**index, kernel, dropout) for index in range(depth))
        )
        self.head = nn.Conv1d(channels, classes, 1)

    def forward(self, values):
        hidden = self.input(values.transpose(1, 2))
        return self.head(self.blocks(hidden)).transpose(1, 2)


class GatedBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, kernel: int, dropout: float):
        super().__init__()
        self.padding = (kernel - 1) * dilation
        self.conv = nn.Conv1d(
            channels, 2 * channels, kernel, padding=self.padding, dilation=dilation
        )
        self.norm = CausalChannelNorm(2 * channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, values):
        convolved = self.conv(values)
        if self.padding:
            convolved = convolved[..., : -self.padding]
        value, gate = self.norm(convolved).chunk(2, dim=1)
        return values + self.drop(torch.tanh(value) * torch.sigmoid(gate))


class GatedTCN(nn.Module):
    def __init__(
        self,
        input_features: int = 500,
        classes: int = len(TOPICS),
        channels: int = 256,
        depth: int = 6,
        kernel: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input = nn.Conv1d(input_features, channels, 1)
        self.blocks = nn.Sequential(
            *(GatedBlock(channels, 2**index, kernel, dropout) for index in range(depth))
        )
        self.head = nn.Conv1d(channels, classes, 1)

    def forward(self, values):
        hidden = self.input(values.transpose(1, 2))
        return self.head(self.blocks(hidden)).transpose(1, 2)


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        dilation: int,
        kernels: list[int],
        dropout: float,
    ):
        super().__init__()
        if channels < len(kernels):
            raise ValueError("channels must be at least the number of multi-scale kernels")
        branch_channels = channels // len(kernels)
        self.paddings = [(kernel - 1) * dilation for kernel in kernels]
        self.branches = nn.ModuleList(
            nn.Conv1d(
                channels,
                branch_channels,
                kernel,
                padding=padding,
                dilation=dilation,
            )
            for kernel, padding in zip(kernels, self.paddings)
        )
        self.merge = nn.Conv1d(branch_channels * len(kernels), channels, 1)
        self.norm = CausalChannelNorm(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, values):
        branches = []
        for conv, padding in zip(self.branches, self.paddings):
            branch = conv(values)
            if padding:
                branch = branch[..., :-padding]
            branches.append(branch)
        merged = self.merge(torch.cat(branches, dim=1))
        return values + self.drop(torch.relu(self.norm(merged)))


class MultiScaleTCN(nn.Module):
    def __init__(
        self,
        input_features: int = 500,
        classes: int = len(TOPICS),
        channels: int = 192,
        depth: int = 6,
        kernels: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        kernels = kernels or [3, 5, 9]
        self.input = nn.Conv1d(input_features, channels, 1)
        self.blocks = nn.Sequential(
            *(
                MultiScaleBlock(channels, 2**index, kernels, dropout)
                for index in range(depth)
            )
        )
        self.head = nn.Conv1d(channels, classes, 1)

    def forward(self, values):
        hidden = self.input(values.transpose(1, 2))
        return self.head(self.blocks(hidden)).transpose(1, 2)


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        dilation: int,
        kernel: int,
        expansion: int,
        dropout: float,
        layer_scale: float,
    ):
        super().__init__()
        self.padding = (kernel - 1) * dilation
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel,
            padding=self.padding,
            dilation=dilation,
            groups=channels,
        )
        self.norm = CausalChannelNorm(channels)
        self.expand = nn.Conv1d(channels, expansion * channels, 1)
        self.contract = nn.Conv1d(expansion * channels, channels, 1)
        self.drop = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.full((1, channels, 1), layer_scale))

    def forward(self, values):
        hidden = self.depthwise(values)
        if self.padding:
            hidden = hidden[..., : -self.padding]
        hidden = self.contract(torch.nn.functional.gelu(self.expand(self.norm(hidden))))
        return values + self.drop(self.scale * hidden)


class ConvNeXtTCN(nn.Module):
    def __init__(
        self,
        input_features: int = 500,
        classes: int = len(TOPICS),
        channels: int = 256,
        depth: int = 6,
        kernel: int = 7,
        expansion: int = 4,
        dropout: float = 0.1,
        layer_scale: float = 0.01,
    ):
        super().__init__()
        self.input = nn.Conv1d(input_features, channels, 1)
        self.blocks = nn.Sequential(
            *(
                ConvNeXtBlock(
                    channels,
                    2**index,
                    kernel,
                    expansion,
                    dropout,
                    layer_scale,
                )
                for index in range(depth)
            )
        )
        self.head = nn.Conv1d(channels, classes, 1)

    def forward(self, values):
        hidden = self.input(values.transpose(1, 2))
        return self.head(self.blocks(hidden)).transpose(1, 2)


class JointConvNeXtTCN(nn.Module):
    """ConvNeXt TCN with a shared causal trunk and topic/boundary heads.

    The boundary head predicts whether the current token begins a new topic
    segment. Both heads see exactly the same causal hidden representation.
    """

    def __init__(
        self,
        input_features: int = 500,
        classes: int = len(TOPICS),
        channels: int = 256,
        depth: int = 6,
        kernel: int = 7,
        expansion: int = 4,
        dropout: float = 0.1,
        layer_scale: float = 0.01,
    ):
        super().__init__()
        self.input = nn.Conv1d(input_features, channels, 1)
        self.blocks = nn.Sequential(
            *(
                ConvNeXtBlock(
                    channels, 2**index, kernel, expansion, dropout, layer_scale
                )
                for index in range(depth)
            )
        )
        self.topic_head = nn.Conv1d(channels, classes, 1)
        self.boundary_head = nn.Conv1d(channels, 1, 1)

    def forward(self, values):
        hidden = self.blocks(self.input(values.transpose(1, 2)))
        topics = self.topic_head(hidden).transpose(1, 2)
        boundaries = self.boundary_head(hidden).squeeze(1)
        return topics, boundaries


class CausalTransformer(nn.Module):
    """Causal token classifier over the same cached SAE activation sequences."""

    def __init__(
        self,
        input_features: int = 500,
        classes: int = len(TOPICS),
        hidden_width: int = 128,
        heads: int = 4,
        depth: int = 2,
        feedforward_width: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_width = hidden_width
        self.project = nn.Linear(input_features, hidden_width)
        layer = nn.TransformerEncoderLayer(
            hidden_width,
            heads,
            dim_feedforward=feedforward_width,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, depth, nn.LayerNorm(hidden_width)
        )
        self.head = nn.Linear(hidden_width, classes)

    def forward(self, values):
        length = values.shape[1]
        mask = torch.ones(
            length, length, dtype=torch.bool, device=values.device
        ).triu(1)
        hidden = self.project(values) * math.sqrt(self.hidden_width)
        return self.head(self.encoder(hidden, mask=mask, is_causal=True))


MODEL_FACTORIES: dict[str, Callable[..., nn.Module]] = {
    "baseline": BaselineTCN,
    "gated": GatedTCN,
    "multiscale": MultiScaleTCN,
    "convnext": ConvNeXtTCN,
    "joint_convnext": JointConvNeXtTCN,
    "transformer": CausalTransformer,
}


def build_model(architecture: str, model_config: dict) -> nn.Module:
    try:
        factory = MODEL_FACTORIES[architecture]
    except KeyError as exc:
        raise ValueError(f"unknown TCN architecture: {architecture}") from exc
    return factory(**model_config)


@torch.inference_mode()
def predict_sequences(model, rows, device="cuda"):
    model.eval()
    result = []
    for features, labels in rows:
        tensor = torch.from_numpy(np.asarray(features, dtype=np.float32))[None].to(device)
        logits = model(tensor)[0]
        result.append(
            {
                "labels": labels.copy(),
                "predictions": logits.argmax(-1).cpu().numpy(),
            }
        )
    return result


def flatten_labeled(rows):
    truth, predicted = [], []
    for row in rows:
        keep = row["labels"] != PAD
        truth.append(row["labels"][keep])
        predicted.append(row["predictions"][keep])
    return np.concatenate(truth), np.concatenate(predicted)


def boundary_positions(labels):
    """Return transitions in original token coordinates, retaining masked gaps."""
    positions = np.flatnonzero(labels != PAD)
    if len(positions) < 2:
        return np.empty(0, dtype=np.int64)
    values = labels[positions]
    return positions[1:][values[1:] != values[:-1]]


def boundary_report(rows, tolerance: int):
    matched = predicted = true = 0
    errors = []
    for row in rows:
        labels = row["labels"]
        keep = labels != PAD
        predicted_full = np.full_like(labels, PAD)
        predicted_full[keep] = row["predictions"][keep]
        true_positions = boundary_positions(labels)
        predicted_positions = boundary_positions(predicted_full)
        true += len(true_positions)
        predicted += len(predicted_positions)
        unused = set(map(int, predicted_positions))
        for position in true_positions:
            candidates = [
                candidate
                for candidate in unused
                if abs(candidate - int(position)) <= tolerance
            ]
            if candidates:
                candidate = min(candidates, key=lambda value: abs(value - int(position)))
                unused.remove(candidate)
                matched += 1
                errors.append(candidate - int(position))
    precision = matched / predicted if predicted else 0.0
    recall = matched / true if true else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "matched": matched,
        "predicted": predicted,
        "true": true,
        "median_absolute_error": float(np.median(np.abs(errors))) if errors else None,
    }


def metrics(rows, tolerances: list[int] | None = None):
    tolerances = tolerances or [5, 10]
    truth, predicted = flatten_labeled(rows)
    report = classification_report(
        truth,
        predicted,
        labels=range(len(TOPICS)),
        target_names=TOPICS,
        output_dict=True,
        zero_division=0,
    )
    result = {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "per_topic": {topic: report[topic] for topic in TOPICS},
    }
    for tolerance in tolerances:
        result[f"boundary_at_{tolerance}"] = boundary_report(rows, tolerance)
    return result
