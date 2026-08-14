#!/usr/bin/env python3
"""Load and inspect saved model checkpoints without changing their format."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from models import build_model


def load_checkpoint(checkpoint_path: Path, device: str = "cpu"):
    """Restore a saved model and return it together with its checkpoint metadata."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {"architecture", "model", "state_dict"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"checkpoint is missing fields: {sorted(missing)}")
    model = build_model(checkpoint["architecture"], checkpoint["model"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


def describe_checkpoint(checkpoint_path: Path, checkpoint: dict) -> None:
    """Print a short human-readable checkpoint description."""
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Architecture: {checkpoint['architecture']}")
    print(f"Epoch: {checkpoint.get('epoch', 'not recorded')}")
    print(f"Selection: {checkpoint.get('selection', 'not recorded')}")
    print(f"Model settings: {checkpoint['model']}")


def main() -> None:
    """Load a checkpoint from the command line and print its metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    _, checkpoint = load_checkpoint(arguments.checkpoint, arguments.device)
    describe_checkpoint(arguments.checkpoint, checkpoint)


if __name__ == "__main__":
    main()
