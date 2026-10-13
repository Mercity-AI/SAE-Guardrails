#!/usr/bin/env python3
"""Run static detector baselines from an explicit experiment configuration.

Every detector the YAML names is trained against one cache load:

    python training/train_static.py training/configs/gemma1b_baselines_sae150_10k_pr.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

# local imports
from training import static_engine as training


def main() -> None:
    """Train every detector named in a static experiment config (one cache load for all of them)."""
    parser = argparse.ArgumentParser(description="Train the static detectors named in one config.")
    parser.add_argument("config", type=Path, help="YAML experiment configuration")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="write here instead of the config's experiment.output_dir (keeps reruns apart)",
    )
    args = parser.parse_args()
    training.run(args.config.resolve(), output_dir_override=args.output_dir)


if __name__ == "__main__":
    main()
