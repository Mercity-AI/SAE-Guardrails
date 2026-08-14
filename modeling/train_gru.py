#!/usr/bin/env python3
"""Train ordinary or joint causal GRU models from saved YAML configurations."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

import training
import training_joint


ARCHITECTURES = {"gru", "joint_gru"}


def run_config(config_path: Path) -> None:
    """Run the GRU model declared by one configuration file."""
    config = yaml.safe_load(config_path.read_text())
    if "models" in config:
        training.run(config_path, ARCHITECTURES)
        return
    architecture = config.get("model", {}).get("architecture")
    if architecture not in ARCHITECTURES:
        raise ValueError(f"configuration does not describe a GRU: {architecture}")
    training_joint.train_joint_model(config_path)


def main() -> None:
    """Parse configuration paths and train each requested GRU experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, action="append", required=True)
    arguments = parser.parse_args()
    for config_path in arguments.config:
        run_config(config_path)


if __name__ == "__main__":
    main()
