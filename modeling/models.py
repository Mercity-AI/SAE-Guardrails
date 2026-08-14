"""Shared causal TCN models, data utilities, and original-coordinate metrics."""
from __future__ import annotations

import math
import os
import random
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
    partitions = {
        name: [data[index] for index in split[name]]
        for name in ("train", "validation", "test")
    }
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
            feature_tensor = torch.from_numpy(
                np.array(features, dtype=np.float32, copy=True)
            )
        if isinstance(labels, torch.Tensor):
            label_tensor = labels
        else:
            label_tensor = torch.from_numpy(labels)
        x_batch[index, : len(features)] = feature_tensor
        y_batch[index, : len(labels)] = label_tensor
    return x_batch, y_batch


class CausalChannelNorm(nn.Module):
    """Normalize channels at each token, never across time or padding positions."""

    def __init__(self, channels: int):
        """Create normalization over the channel dimension only."""
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, values):
        """Normalize each token independently across channels."""
        return self.norm(values.transpose(1, 2)).transpose(1, 2)


class ResidualBlock(nn.Module):
    """Plain residual left-causal convolution block."""

    def __init__(self, channels: int, dilation: int, kernel: int, dropout: float):
        """Create one dilated residual convolution block."""
        super().__init__()
        self.padding = (kernel - 1) * dilation
        self.conv = nn.Conv1d(
            channels, channels, kernel, padding=self.padding, dilation=dilation
        )
        self.norm = CausalChannelNorm(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, values):
        """Apply the causal convolution and residual update."""
        convolved = self.conv(values)
        if self.padding:
            convolved = convolved[..., : -self.padding]
        return values + self.drop(torch.relu(self.norm(convolved)))


class BaselineTCN(nn.Module):
    """Plain causal temporal convolutional topic classifier."""

    def __init__(
        self,
        input_features: int = 500,
        classes: int = len(TOPICS),
        channels: int = 128,
        depth: int = 6,
        kernel: int = 3,
        dropout: float = 0.1,
    ):
        """Create the configured plain TCN stack."""
        super().__init__()
        self.input = nn.Conv1d(input_features, channels, 1)
        self.blocks = nn.Sequential(
            *(ResidualBlock(channels, 2**index, kernel, dropout) for index in range(depth))
        )
        self.head = nn.Conv1d(channels, classes, 1)

    def forward(self, values):
        """Return per-token topic logits."""
        hidden = self.input(values.transpose(1, 2))
        return self.head(self.blocks(hidden)).transpose(1, 2)


class GatedBlock(nn.Module):
    """Residual causal block with value and gate branches."""

    def __init__(self, channels: int, dilation: int, kernel: int, dropout: float):
        """Create one gated dilated convolution block."""
        super().__init__()
        self.padding = (kernel - 1) * dilation
        self.conv = nn.Conv1d(
            channels, 2 * channels, kernel, padding=self.padding, dilation=dilation
        )
        self.norm = CausalChannelNorm(2 * channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, values):
        """Apply the gated causal convolution and residual update."""
        convolved = self.conv(values)
        if self.padding:
            convolved = convolved[..., : -self.padding]
        value, gate = self.norm(convolved).chunk(2, dim=1)
        return values + self.drop(torch.tanh(value) * torch.sigmoid(gate))


class GatedTCN(nn.Module):
    """Gated causal temporal convolutional topic classifier."""

    def __init__(
        self,
        input_features: int = 500,
        classes: int = len(TOPICS),
        channels: int = 256,
        depth: int = 6,
        kernel: int = 3,
        dropout: float = 0.1,
    ):
        """Create the configured gated TCN stack."""
        super().__init__()
        self.input = nn.Conv1d(input_features, channels, 1)
        self.blocks = nn.Sequential(
            *(GatedBlock(channels, 2**index, kernel, dropout) for index in range(depth))
        )
        self.head = nn.Conv1d(channels, classes, 1)

    def forward(self, values):
        """Return per-token topic logits."""
        hidden = self.input(values.transpose(1, 2))
        return self.head(self.blocks(hidden)).transpose(1, 2)


class MultiScaleBlock(nn.Module):
    """Residual causal block with several parallel kernel widths."""

    def __init__(
        self,
        channels: int,
        dilation: int,
        kernels: list[int],
        dropout: float,
    ):
        """Create parallel dilated convolution branches."""
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
        """Merge the causal branches and apply the residual update."""
        branches = []
        for conv, padding in zip(self.branches, self.paddings):
            branch = conv(values)
            if padding:
                branch = branch[..., :-padding]
            branches.append(branch)
        merged = self.merge(torch.cat(branches, dim=1))
        return values + self.drop(torch.relu(self.norm(merged)))


class MultiScaleTCN(nn.Module):
    """Multi-scale causal temporal convolutional topic classifier."""

    def __init__(
        self,
        input_features: int = 500,
        classes: int = len(TOPICS),
        channels: int = 192,
        depth: int = 6,
        kernels: list[int] | None = None,
        dropout: float = 0.1,
    ):
        """Create the configured multi-scale TCN stack."""
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
        """Return per-token topic logits."""
        hidden = self.input(values.transpose(1, 2))
        return self.head(self.blocks(hidden)).transpose(1, 2)


class ConvNeXtBlock(nn.Module):
    """Left-causal ConvNeXt-style temporal block."""

    def __init__(
        self,
        channels: int,
        dilation: int,
        kernel: int,
        expansion: int,
        dropout: float,
        layer_scale: float,
    ):
        """Create one depthwise causal ConvNeXt block."""
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
        """Apply the ConvNeXt transformation and residual update."""
        hidden = self.depthwise(values)
        if self.padding:
            hidden = hidden[..., : -self.padding]
        hidden = self.contract(torch.nn.functional.gelu(self.expand(self.norm(hidden))))
        return values + self.drop(self.scale * hidden)


class ConvNeXtTCN(nn.Module):
    """ConvNeXt-style causal temporal topic classifier."""

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
        """Create the configured ConvNeXt TCN stack."""
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
        """Return per-token topic logits."""
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
        """Create a ConvNeXt trunk with topic and boundary heads."""
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
        """Return per-token topic logits and boundary logits."""
        hidden = self.blocks(self.input(values.transpose(1, 2)))
        topics = self.topic_head(hidden).transpose(1, 2)
        boundaries = self.boundary_head(hidden).squeeze(1)
        return topics, boundaries


class CausalGRU(nn.Module):
    """Causal token classifier built from PyTorch's native GRU."""

    def __init__(
        self,
        input_features: int = 500,
        classes: int = len(TOPICS),
        hidden_width: int = 256,
        depth: int = 3,
        dropout: float = 0.1,
    ):
        """Create a unidirectional GRU topic classifier."""
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_features,
            hidden_size=hidden_width,
            num_layers=depth,
            batch_first=True,
            dropout=dropout if depth > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Linear(hidden_width, classes)

    def forward(self, values):
        """Return topic logits from chronological recurrent states."""
        hidden, _ = self.gru(values)
        return self.head(hidden)


class JointCausalGRU(nn.Module):
    """Causal GRU trunk with matched topic and boundary heads."""

    def __init__(
        self,
        input_features: int = 500,
        classes: int = len(TOPICS),
        hidden_width: int = 256,
        depth: int = 3,
        dropout: float = 0.1,
    ):
        """Create a unidirectional GRU with topic and boundary heads."""
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_features,
            hidden_size=hidden_width,
            num_layers=depth,
            batch_first=True,
            dropout=dropout if depth > 1 else 0.0,
            bidirectional=False,
        )
        self.topic_head = nn.Linear(hidden_width, classes)
        self.boundary_head = nn.Linear(hidden_width, 1)

    def forward(self, values):
        """Return per-token topic logits and boundary logits."""
        hidden, _ = self.gru(values)
        return self.topic_head(hidden), self.boundary_head(hidden).squeeze(-1)


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
        """Create a strictly masked causal Transformer topic classifier."""
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
        """Return topic logits without attending to later tokens."""
        length = values.shape[1]
        mask = torch.ones(
            length, length, dtype=torch.bool, device=values.device
        ).triu(1)
        hidden = self.project(values) * math.sqrt(self.hidden_width)
        return self.head(self.encoder(hidden, mask=mask, is_causal=True))


class JointCausalTransformer(nn.Module):
    """Strictly causal Transformer trunk with topic and boundary heads."""

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
        """Create a causal Transformer with topic and boundary heads."""
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
        self.encoder = nn.TransformerEncoder(layer, depth, nn.LayerNorm(hidden_width))
        self.topic_head = nn.Linear(hidden_width, classes)
        self.boundary_head = nn.Linear(hidden_width, 1)

    def forward(self, values):
        """Return causal per-token topic and boundary logits."""
        length = values.shape[1]
        mask = torch.ones(length, length, dtype=torch.bool, device=values.device).triu(1)
        hidden = self.project(values) * math.sqrt(self.hidden_width)
        hidden = self.encoder(hidden, mask=mask, is_causal=True)
        return self.topic_head(hidden), self.boundary_head(hidden).squeeze(-1)


MODEL_FACTORIES: dict[str, Callable[..., nn.Module]] = {
    "baseline": BaselineTCN,
    "gated": GatedTCN,
    "multiscale": MultiScaleTCN,
    "convnext": ConvNeXtTCN,
    "joint_convnext": JointConvNeXtTCN,
    "joint_gru": JointCausalGRU,
    "joint_transformer": JointCausalTransformer,
    "gru": CausalGRU,
    "transformer": CausalTransformer,
}


def build_model(architecture: str, model_config: dict) -> nn.Module:
    """Construct a registered architecture from saved model settings."""
    try:
        factory = MODEL_FACTORIES[architecture]
    except KeyError as exc:
        raise ValueError(f"unknown TCN architecture: {architecture}") from exc
    return factory(**model_config)


@torch.inference_mode()
def predict_sequences(model, rows, device="cuda", batch_size: int = 1, decoder=None):
    """Run batched causal decoding and restore original sequence lengths."""
    from decoder_utils import decode_logits

    model.eval()
    result = []
    for start in range(0, len(rows), batch_size):
        selected = rows[start : start + batch_size]
        tensors, _ = collate(selected, input_features=selected[0][0].shape[-1])
        logits = model(tensors.to(device)).float().cpu().numpy()
        for index, (features, labels) in enumerate(selected):
            label_values = (
                labels.detach().cpu().numpy()
                if isinstance(labels, torch.Tensor)
                else labels.copy()
            )
            result.append(
                {
                    "labels": label_values,
                    "predictions": decode_logits(logits[index, : len(features)], decoder),
                }
            )
    return result


def flatten_labeled(rows):
    """Flatten only supervised labels and matching predictions across records."""
    truth, predicted = [], []
    for row in rows:
        keep = row["labels"] != PAD
        truth.append(row["labels"][keep])
        predicted.append(row["predictions"][keep])
    return np.concatenate(truth), np.concatenate(predicted)


def topic_overlap(truth, predicted):
    """Pooled token-wise one-vs-rest Dice/IoU for every topic class."""
    truth = np.asarray(truth)
    predicted = np.asarray(predicted)
    per_topic = {}
    for class_id, topic in enumerate(TOPICS):
        true_mask = truth == class_id
        predicted_mask = predicted == class_id
        intersection = int(np.count_nonzero(true_mask & predicted_mask))
        true_tokens = int(np.count_nonzero(true_mask))
        predicted_tokens = int(np.count_nonzero(predicted_mask))
        dice_denominator = true_tokens + predicted_tokens
        union = true_tokens + predicted_tokens - intersection
        per_topic[topic] = {
            "dice": 2.0 * intersection / dice_denominator if dice_denominator else 0.0,
            "iou": intersection / union if union else 0.0,
            "intersection_tokens": intersection,
            "true_tokens": true_tokens,
            "predicted_tokens": predicted_tokens,
        }
    return {
        "definition": "pooled labeled-token one-vs-rest overlap; macro is unweighted across topics",
        "macro_dice": float(np.mean([value["dice"] for value in per_topic.values()])),
        "macro_iou": float(np.mean([value["iou"] for value in per_topic.values()])),
        "per_topic": per_topic,
    }


def boundary_positions(labels):
    """Return transitions in original token coordinates, retaining masked gaps."""
    positions = np.flatnonzero(labels != PAD)
    if len(positions) < 2:
        return np.empty(0, dtype=np.int64)
    values = labels[positions]
    return positions[1:][values[1:] != values[:-1]]


def boundary_report(rows, tolerance: int):
    """Calculate one-to-one tolerant boundary precision, recall, and F1."""
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


def topic_detection_latency(rows, confirmation_tokens: int = 3):
    """Measure how far into each true topic span a correct prediction appears.

    Latency is one-based tokens consumed: an immediately correct prediction has latency 1.
    ``confirmation_tokens`` requires that many consecutive correct predictions, shortened to
    the span length for spans shorter than the requested confirmation window. Statistics over
    latency are conditional on detection; detection rate and misses are reported separately.
    """
    observations = {topic: [] for topic in TOPICS}
    for row in rows:
        keep = row["labels"] != PAD
        truth = np.asarray(row["labels"])[keep]
        predicted = np.asarray(row["predictions"])[keep]
        start = 0
        while start < len(truth):
            stop = start + 1
            while stop < len(truth) and truth[stop] == truth[start]:
                stop += 1
            class_id = int(truth[start])
            span_predictions = predicted[start:stop]
            span_length = stop - start
            required = min(int(confirmation_tokens), span_length)
            consumed = None
            for offset in range(span_length - required + 1):
                if np.all(span_predictions[offset : offset + required] == class_id):
                    consumed = offset + required
                    break
            observations[TOPICS[class_id]].append((span_length, consumed))
            start = stop

    def summarize(values):
        """Summarize detection coverage and conditional latency values."""
        detected = [(length, consumed) for length, consumed in values if consumed is not None]
        tokens = np.asarray([consumed for _, consumed in detected], dtype=np.float64)
        fractions = np.asarray(
            [consumed / length for length, consumed in detected], dtype=np.float64
        )
        spans = len(values)
        found = len(detected)
        return {
            "spans": spans,
            "detected_spans": found,
            "missed_spans": spans - found,
            "detection_rate": found / spans if spans else 0.0,
            "mean_tokens_consumed": float(tokens.mean()) if found else None,
            "median_tokens_consumed": float(np.median(tokens)) if found else None,
            "p90_tokens_consumed": float(np.percentile(tokens, 90)) if found else None,
            "mean_fraction_consumed": float(fractions.mean()) if found else None,
            "median_fraction_consumed": float(np.median(fractions)) if found else None,
        }

    pooled = [value for values in observations.values() for value in values]
    return {
        "definition": (
            "one-based labeled topic tokens consumed from each true span start until the first "
            f"run of {confirmation_tokens} correct predictions; for shorter spans the required "
            "run equals span length; latency summaries are conditional on detection"
        ),
        "confirmation_tokens": int(confirmation_tokens),
        "overall": summarize(pooled),
        "per_topic": {topic: summarize(values) for topic, values in observations.items()},
    }


def metrics(rows, tolerances: list[int] | None = None):
    """Calculate token, overlap, boundary, and acquisition metrics."""
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
        "topic_overlap": topic_overlap(truth, predicted),
        "per_topic": {topic: report[topic] for topic in TOPICS},
        "topic_detection_first_correct": topic_detection_latency(rows, confirmation_tokens=1),
        "topic_detection_confirmed_3": topic_detection_latency(rows, confirmation_tokens=3),
    }
    for tolerance in tolerances:
        result[f"boundary_at_{tolerance}"] = boundary_report(rows, tolerance)
    return result
