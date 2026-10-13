#!/usr/bin/env python3
"""Derive a dynamic cache's prompt+response (``_pr``) copy from its base cache.

The dynamic cache stores tokens in the same order as the static cache of the same backbone
(byte-equal lengths and roles), so it reuses the static ``_pr`` labels, as int64 because the
dynamic analysis indexes with them. The heavy sparse arrays are symlinked, not copied. Run
``add_prompt_labels.py`` for the same backbone first.

    python preprocessing/add_dynamic_prompt_labels.py --model-size 1b
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

# local imports
from common.paths import PROJECT_ROOT

PROFILES = {  # base dynamic cache, its _pr copy, the static _pr cache holding the labels
    "1b": ("cache/dyn150_sparse_10k_1b", "cache/dyn150_sparse_10k_1b_pr", "cache/sae1500_10k_1b_pr"),
    "4b": ("cache/dyn150_sparse_10k_4b", "cache/dyn150_sparse_10k_4b_pr", "cache/sae1500_10k_gemma4b_pr"),
}
LINKED = (
    "col.npy",
    "val.npy",
    "tok_ptr.npy",
    "lengths.npy",
    "roles.npy",
    "split_indices.npz",
    "sae_index.npy",
    "selected_features.npz",
    "meta.json",
)


def main() -> None:
    """Symlink the base cache's arrays into the _pr folder and write its int64 labels."""
    parser = argparse.ArgumentParser(description="Derive a dynamic _pr cache from its base cache.")
    parser.add_argument("--model-size", choices=sorted(PROFILES), required=True)
    args = parser.parse_args()
    src, out, labels = (PROJECT_ROOT / p for p in PROFILES[args.model_size])

    out.mkdir(parents=True, exist_ok=True)
    for name in LINKED:
        link = out / name
        link.unlink(missing_ok=True)
        link.symlink_to((src / name).resolve())
    np.save(out / "labels.npy", np.load(labels / "labels.npy").astype(np.int64))
    print(f"wrote {out} (arrays linked from {src}, labels from {labels})")


if __name__ == "__main__":
    main()
