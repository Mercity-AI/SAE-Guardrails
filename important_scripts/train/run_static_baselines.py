#!/usr/bin/env python3
"""Run static detector baselines from an explicit experiment configuration."""

from __future__ import annotations

import argparse
from pathlib import Path


# local imports
from important_scripts.model import training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="YAML experiment configuration")
    args = parser.parse_args()
    training.run(args.config.resolve())


if __name__ == "__main__":
    main()
