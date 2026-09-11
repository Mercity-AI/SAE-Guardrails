#!/usr/bin/env python3
"""Build a PROMPT+RESPONSE-supervised copy of a static sae1500 cache (non-destructive), 1B or 4B.

Consolidates relabel_1b_prompt_response.py and relabel_4b_pr_nondestructive.py, which differed only
by MODEL_ID / SRC / OUT. Creates cache/<out>/ that symlinks features/lengths/roles/split/selection
from the response-only base cache and writes a fresh prompt+response labels.npy there, using the
shared labeler (label_record_pr) so the convention matches the caches exactly. The base
response-only cache is never modified.

  python -m important_scripts.cache.relabel_pr --model-size {1b,4b}
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from transformers import AutoTokenizer

from important_scripts.cache.relabel_pr_labels import label_record_pr
from important_scripts.model.models import PAD
from important_scripts.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
DATASET = ROOT / "runs/GPT_5.6_10k_2108/dataset.final.jsonl"
LINK = [
    "features.npy",
    "lengths.npy",
    "role_ids.npy",
    "split_indices.npz",
    "selected_features.npz",
    "metadata.json",
]

# Only these three values differed between the 1B and 4B relabel twins.
PROFILES = {
    "1b": {
        "model_id": "google/gemma-3-1b-it",
        "src": "cache/sae1500_10k_1b_prompt_response",
        "out": "cache/sae1500_10k_1b_pr",
    },
    "4b": {
        "model_id": "google/gemma-3-4b-it",
        "src": "cache/sae1500_10k_gemma4b_prompt_response",
        "out": "cache/sae1500_10k_gemma4b_pr",
    },
}


def main() -> None:
    """Relabel one backbone's response-only cache to prompt+response; write the non-destructive PR copy."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-size", choices=sorted(PROFILES), required=True)
    args = ap.parse_args()
    profile = PROFILES[args.model_size]
    model_id = profile["model_id"]
    src = ROOT / profile["src"]
    out = ROOT / profile["out"]

    tok = AutoTokenizer.from_pretrained(model_id)
    recs = [json.loads(line)["record"] for line in DATASET.open()]
    lengths = np.load(src / "lengths.npy")
    roles_ref = np.load(src / "role_ids.npy")

    labels_all, roles_all = [], []
    for k, r in enumerate(recs):
        full, roles, labels = label_record_pr(r, tok)
        if len(full) != int(lengths[k]):
            raise ValueError(f"record {k}: token count {len(full)} != cached length {int(lengths[k])}")
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

    out.mkdir(parents=True, exist_ok=True)
    for name in LINK:
        dst = out / name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src / name)
    np.save(out / "labels.npy", labels.astype(np.int16))
    meta = json.loads((src / "metadata.json").read_text()) if (src / "metadata.json").exists() else {}
    meta["supervision"] = "prompt_and_response_topic_tokens (relabeled from response-only)"
    (out / "metadata.json").unlink(missing_ok=True)  # replace the symlinked meta with a real file
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out}/labels.npy (+ symlinks); response-only cache untouched", flush=True)


if __name__ == "__main__":
    main()
