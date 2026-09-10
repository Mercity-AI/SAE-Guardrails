#!/usr/bin/env python3
"""Build a PROMPT+RESPONSE-supervised copy of the 1B sae1500 cache (non-destructive).

Mirrors relabel_4b_prompt_response.py but for the 1B backbone and WITHOUT clobbering the original
response-only cache: it creates cache/sae1500_10k_1b_pr/ that symlinks features/lengths/roles/
split/selection from the response-only cache and writes a fresh prompt+response labels.npy there.
Uses the identical labeler (label_record_pr) as the 4B relabel so the convention matches exactly.

  python relabel_1b_prompt_response.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


# local imports
from important_scripts.paths import PROJECT_ROOT
from important_scripts.model.models import PAD
from important_scripts.cache.relabel_pr_labels import (
    label_record_pr,
)

ROOT = PROJECT_ROOT

MODEL_ID = "google/gemma-3-1b-it"
DATASET = ROOT / "runs/GPT_5.6_10k_2108/dataset.final.jsonl"
SRC = ROOT / "cache/sae1500_10k_1b_prompt_response"
OUT = ROOT / "cache/sae1500_10k_1b_pr"
LINK = [
    "features.npy",
    "lengths.npy",
    "role_ids.npy",
    "split_indices.npz",
    "selected_features.npz",
    "metadata.json",
]


def main() -> None:
    """Relabel the 1B response-only cache to prompt+response and write the non-destructive PR copy."""
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    recs = [json.loads(line)["record"] for line in DATASET.open()]
    lengths = np.load(SRC / "lengths.npy")
    roles_ref = np.load(SRC / "role_ids.npy")

    labels_all, roles_all = [], []
    for k, r in enumerate(recs):
        full, roles, labels = label_record_pr(r, tok)
        if len(full) != int(lengths[k]):
            raise ValueError(
                f"record {k}: token count {len(full)} != cached length {int(lengths[k])}"
            )
        labels_all.append(np.asarray(labels, np.int64))
        roles_all.append(np.asarray(roles, np.int8))
        if (k + 1) % 2000 == 0:
            print(f"  relabeled {k + 1}/{len(recs)}", flush=True)
    labels = np.concatenate(labels_all)
    roles = np.concatenate(roles_all)
    if not np.array_equal(roles.astype(np.int8), roles_ref.astype(np.int8)):
        raise ValueError("recomputed roles differ from cached role_ids.npy -- tokenization drift")

    r1, r2 = roles == 1, roles == 2
    print(
        f"prompt labeled: {int((labels[r1] != PAD).sum())}/{int(r1.sum())} | "
        f"response labeled: {int((labels[r2] != PAD).sum())}/{int(r2.sum())} | "
        f"topics present: {sorted(set(labels.tolist()))}",
        flush=True,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    for name in LINK:
        dst = OUT / name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(SRC / name)
    np.save(OUT / "labels.npy", labels.astype(np.int16))
    meta = (
        json.loads((SRC / "metadata.json").read_text()) if (SRC / "metadata.json").exists() else {}
    )
    meta["supervision"] = "prompt_and_response_topic_tokens (relabeled from response-only)"
    (OUT / "metadata.json").unlink(missing_ok=True)  # replace the symlinked meta with a real file
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {OUT}/labels.npy (+ symlinks); response-only cache untouched", flush=True)


if __name__ == "__main__":
    main()
