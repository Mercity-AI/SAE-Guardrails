"""The three canonical detectors and the data/metric helpers they are trained with.

Only the architectures this study reports are registered: a recurrent detector (``gru``), a
causal transformer (``transformer``) and a causal convolutional stack (``convnext``). Earlier
ablations also carried plain, gated and multi-scale temporal convolution variants and three
joint topic+boundary heads. Those were deleted, not moved: checkpoints saved under those names
cannot be rebuilt from this tree, and the runs that used them are finished and reported.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn

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


class CausalChannelNorm(nn.Module):
    """Normalize channels at each token, never across time or padding positions."""

    def __init__(self, channels: int):
        """Create normalization over the channel dimension only."""
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, values):
        """Normalize each token independently across channels."""
        return self.norm(values.transpose(1, 2)).transpose(1, 2)


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
        self.encoder = nn.TransformerEncoder(layer, depth, nn.LayerNorm(hidden_width))
        self.head = nn.Linear(hidden_width, classes)

    def forward(self, values):
        """Return topic logits without attending to later tokens."""
        length = values.shape[1]
        mask = torch.ones(length, length, dtype=torch.bool, device=values.device).triu(1)
        hidden = self.project(values) * math.sqrt(self.hidden_width)
        return self.head(self.encoder(hidden, mask=mask, is_causal=True))


MODEL_FACTORIES: dict[str, Callable[..., nn.Module]] = {
    "gru": CausalGRU,
    "transformer": CausalTransformer,
    "convnext": ConvNeXtTCN,
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
    # local import: decoder_utils imports nothing from here, but this keeps the module pair
    # free of an import cycle at module load time.
    from common.data import collate
    from common.decoding import decode_logits

    model.eval()
    result = []
    for start in range(0, len(rows), batch_size):
        selected = rows[start : start + batch_size]
        tensors, _ = collate(selected, input_features=selected[0][0].shape[-1])
        logits = model(tensors.to(device)).float().cpu().numpy()
        for index, (features, labels) in enumerate(selected):
            label_values = (
                labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else labels.copy()
            )
            result.append(
                {
                    "labels": label_values,
                    "predictions": decode_logits(logits[index, : len(features)], decoder),
                }
            )
    return result
