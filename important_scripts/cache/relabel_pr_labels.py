#!/usr/bin/env python3
"""Rewrite the 4B sae1500 cache labels to PROMPT+RESPONSE topic supervision.

The maintained `label_record` labels response spans only (prompt -> PAD). The baseline
configs (gemma4b_baselines.yaml / sae1500_*.yaml) use `prompt_and_response_topic_tokens`:
tokens inside a <topic>...</topic> span in EITHER the prompt or the response are supervised;
template/neutral tokens stay PAD. This reproduces that convention for the 10k 4B cache,
patching labels.npy and gpu_cache.pt in place. Features/lengths/roles/split are untouched.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer

# local imports
from important_scripts.paths import PROJECT_ROOT
from important_scripts.model.models import PAD
from important_scripts.cache.label_records import _ids, span_char_ranges, strip_tags

ROOT = PROJECT_ROOT

MODEL_ID = "google/gemma-3-4b-it"
DATASET = ROOT / "runs/GPT_5.6_10k_2108/dataset.final.jsonl"
CACHE = ROOT / "cache/sae1500_10k_gemma4b_prompt_response"


def label_record_pr(rec, tok):
    """input_ids, roles, labels with BOTH prompt and response topic spans supervised."""
    prompt, response = strip_tags(rec["prompt"]), strip_tags(rec["response"])
    full = _ids(tok.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
        tokenize=True, add_generation_prompt=False))
    prefix = _ids(tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True))
    prefix_str = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    nprefix, T = len(prefix), len(full)
    roles = [1] * nprefix + [2] * (T - nprefix)
    labels = [PAD] * T

    templated = tok.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
        tokenize=False)
    enc = tok(templated, add_special_tokens=False, return_offsets_mapping=True)
    ids2, offs = enc["input_ids"], enc["offset_mapping"]
    shift = 1 if (len(ids2) == T + 1 and ids2[1:] == list(full)) else 0

    def fill(text_tagged, content, lo):
        ranges = span_char_ranges(text_tagged)
        if not ranges:
            return
        starts = {a for a, b, t in ranges}
        first = ranges[0][0]
        def topic_at(ch):
            for a, b, t in ranges:
                if a <= ch < b:
                    return t
            return ranges[-1][2]
        hi = lo + len(content)
        for k in range(len(ids2)):
            idx = k - shift
            if idx < 0 or idx >= T:
                continue
            s, _ = offs[k]
            if lo <= s < hi:
                rel = s - lo
                labels[idx] = PAD if (rel in starts and rel != first) else topic_at(rel)

    # response content begins at len(prefix_str); prompt content begins where it appears
    fill(rec["response"], response, len(prefix_str))
    p_lo = templated.find(prompt)
    if p_lo >= 0 and prompt:
        fill(rec["prompt"], prompt, p_lo)
    return full, roles, labels


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    recs = [json.loads(l)["record"] for l in DATASET.open()]
    lengths = np.load(CACHE / "lengths.npy")
    roles_ref = np.load(CACHE / "role_ids.npy")
    offsets = np.r_[0, np.cumsum(lengths)]

    labels_all, roles_all = [], []
    for k, r in enumerate(recs):
        full, roles, labels = label_record_pr(r, tok)
        if len(full) != int(lengths[k]):
            raise ValueError(f"record {k}: token count {len(full)} != cached length {int(lengths[k])}")
        labels_all.append(np.asarray(labels, np.int64)); roles_all.append(np.asarray(roles, np.int8))
        if (k + 1) % 2000 == 0:
            print(f"  relabeled {k+1}/{len(recs)}", flush=True)
    labels = np.concatenate(labels_all)
    roles = np.concatenate(roles_all)
    if not np.array_equal(roles.astype(np.int8), roles_ref.astype(np.int8)):
        raise ValueError("recomputed roles differ from cached role_ids.npy")

    r1, r2 = roles == 1, roles == 2
    print(f"prompt labeled: {int((labels[r1]!=PAD).sum())}/{int(r1.sum())} | "
          f"response labeled: {int((labels[r2]!=PAD).sum())}/{int(r2.sum())} | "
          f"topics present: {sorted(set(labels.tolist()))}", flush=True)

    np.save(CACHE / "labels.npy", labels.astype(np.int16))
    gc = torch.load(CACHE / "gpu_cache.pt", weights_only=False, map_location="cpu")
    gc["labels"] = torch.from_numpy(labels.astype(np.int64))       # full prompt+response supervision
    torch.save(gc, CACHE / "gpu_cache.pt")
    meta = json.loads((CACHE / "metadata.json").read_text())
    meta["supervision"] = "prompt_and_response_topic_tokens (relabeled to match baseline configs)"
    (CACHE / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("patched labels.npy + gpu_cache.pt", flush=True)


if __name__ == "__main__":
    main()
