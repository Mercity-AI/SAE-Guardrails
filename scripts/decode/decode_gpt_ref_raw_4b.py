#!/usr/bin/env python3
"""GPT reference control arm for the 4B raw-hidden decode ablation (10k).

The decode ablation scores the raw detectors on GEMMA's own greedy generations against *assumed*
(prompt) labels, because generated text has no per-token ground truth. This script runs the SAME
raw detectors on GPT's reference responses for the SAME 1000 manifest prompts -- and GPT text
carries real per-token topic labels (from the <Topic>..</Topic> tags, materialised in the cache's
labels.npy). So this gives the TRUE-label ceiling: the Gemma-minus-GPT gap isolates decode
style-shift from label noise.

Join: manifest `index` == line number in runs/GPT_5.6_10k_2108/dataset.final.jsonl. The GPT
response is teacher-forced exactly as the 4B cache builder did (verified: reconstructed length ==
cache lengths[index]), so the cache's labels/role_ids align token-for-token. Raw features are
captured via forward hooks on backbone.layers[L] -- identical to decode_raw_4b.py / the trainer.

Leakage note: only the manifest prompts in the cache TEST split were held out from detector
training; the rest were in train/val, so the detector has seen those exact GPT tokens. Metrics
are reported BOTH over all 1000 and over the held-out subset (the honest ceiling).

  python decode_gpt_ref_raw_4b.py --outdir ablation/nosae_stream_4b_10k
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from important_scripts.decode.decode_raw_4b_pr import (
    DETECTORS,
    LAYERS,
    MODEL_ID,
    find_backbone,
    load_det,
)

# local imports
from important_scripts.paths import PROJECT_ROOT
from important_scripts.model.models import PAD, TOPICS, seed_everything

ROOT = PROJECT_ROOT

DATASET = ROOT / "runs/GPT_5.6_10k_2108/dataset.final.jsonl"
# Test-split-only manifest (see build_decode_manifest_testonly.py). With it, every record is
# held out from detector training, so the GPT control arm has no train leakage.
MANIFEST = ROOT / "ablation/gemma_arm/manifest_4b_10k_test.jsonl"
CACHE = ROOT / "cache/sae500_10k_gemma4b_prompt_response"  # labels/role_ids/lengths/split
DEVICE = "cuda"
RESPONSE_ROLE = 2
_TAG_RE = re.compile(r"</?(?:" + "|".join(re.escape(t) for t in TOPICS) + r")>")


def strip_tags(text: str) -> str:
    """Remove every <Topic>/</Topic> marker from the given string."""
    return _TAG_RE.sub("", text)


def teacher_force(prompt: str, response: str, tokenizer):
    """Rebuild the cache builder's exact [prompt+response] input_ids (tags stripped)."""
    ids = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": strip_tags(prompt)},
            {"role": "assistant", "content": strip_tags(response)},
        ],
        tokenize=True,
        add_generation_prompt=False,
    )
    if ids and not isinstance(ids[0], int):
        ids = ids[0].ids
    return list(ids)


@torch.inference_mode()
def extract_raw(full_ids, gemma, captured):
    """Extract the concatenated raw residual-stream features (no SAE) for the given token sequences."""
    tokens = torch.tensor([list(full_ids)], device=DEVICE)
    captured.clear()
    gemma(input_ids=tokens, use_cache=False, return_dict=True)
    missing = set(LAYERS) - set(captured)
    if missing:
        raise RuntimeError(f"missing hooked layers: {missing}")
    return torch.cat([captured[L][0].float() for L in LAYERS], dim=-1)  # (T, 25600) on GPU


def iou_macro(true, pred):
    """Pooled one-vs-rest IoU, macro over the topics actually present in `true`."""
    true = np.asarray(true)
    pred = np.asarray(pred)
    ious = []
    for c in range(len(TOPICS)):
        t = true == c
        if not t.any():
            continue
        p = pred == c
        inter = (t & p).sum()
        union = (t | p).sum()
        ious.append(inter / union if union else 0.0)
    return float(np.mean(ious)) if ious else None


def main():
    """Generate responses, extract features once, and score every detector on the held-out decode set."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="ablation/nosae_stream_4b_10k")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    outdir = ROOT / a.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    seed_everything(a.seed, deterministic=True)

    labels = np.load(CACHE / "labels.npy")
    role_ids = np.load(CACHE / "role_ids.npy")
    lengths = np.load(CACHE / "lengths.npy")
    offsets = np.r_[0, np.cumsum(lengths)]
    test_set = {int(i) for i in np.load(CACHE / "split_indices.npz")["test"]}

    dataset = [json.loads(l) for l in DATASET.open()]
    manifest = [json.loads(l) for l in MANIFEST.open()]
    if a.limit:
        manifest = manifest[: a.limit]

    attn = "flash_attention_2"
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        gemma = (
            AutoModelForCausalLM.from_pretrained(
                MODEL_ID, dtype=torch.bfloat16, attn_implementation=attn
            )
            .to(DEVICE)
            .eval()
        )
    except Exception as e:
        print(f"flash_attention_2 unavailable ({str(e)[:60]}); using sdpa", flush=True)
        attn = "sdpa"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        gemma = (
            AutoModelForCausalLM.from_pretrained(
                MODEL_ID, dtype=torch.bfloat16, attn_implementation=attn
            )
            .to(DEVICE)
            .eval()
        )
    print(f"attn={attn}", flush=True)

    backbone = find_backbone(gemma)
    captured: dict[int, torch.Tensor] = {}
    for L in LAYERS:
        backbone.layers[L].register_forward_hook(
            lambda _m, _i, o, L=L: captured.__setitem__(L, o[0] if isinstance(o, tuple) else o)
        )
    dets = {name: load_det(p, DEVICE) for name, p in DETECTORS.items()}

    per_record, t0 = [], time.time()
    for n, row in enumerate(manifest, 1):
        idx = int(row["index"])
        rec = dataset[idx]["record"]
        full_ids = teacher_force(rec["prompt"], rec["response"], tokenizer)
        lo, hi = int(offsets[idx]), int(offsets[idx + 1])
        if hi - lo != len(full_ids):
            raise RuntimeError(f"idx {idx}: tokenization drift {hi - lo} vs {len(full_ids)}")
        y = labels[lo:hi]
        role = role_ids[lo:hi]
        resp_mask = (role == RESPONSE_ROLE) & (y != PAD)
        out = {
            "index": idx,
            "topics": row["topics"],
            "single_topic": row["single_topic"],
            "assumed_label_id": row["assumed_label_id"],
            "held_out": idx in test_set,
            "response_tokens": int(resp_mask.sum()),
            "pred": {},
        }
        if resp_mask.any():
            x = extract_raw(full_ids, gemma, captured)[None]
            true = y[resp_mask].astype(int)
            with torch.inference_mode():
                for name, m in dets.items():
                    pred_full = m(x)[0].argmax(-1).cpu().numpy()
                    pred = pred_full[resp_mask].astype(int)
                    out["pred"][name] = pred.tolist()
            out["true_labels"] = true.tolist()
        per_record.append(out)
        if n % 50 == 0 or n == len(manifest):
            print(f"{n}/{len(manifest)} scored [{(time.time() - t0) / 60:.1f}m]", flush=True)

    # ---- aggregate: true-label acc / macro-IoU on response tokens; assumed acc for single ----
    def summarise(records):
        res = {}
        for name in DETECTORS:
            true_all, pred_all, single_assumed = [], [], []
            for r in records:
                if not r["pred"].get(name):
                    continue
                pred = np.asarray(r["pred"][name])
                true = np.asarray(r["true_labels"])
                true_all.append(true)
                pred_all.append(pred)
                if r["single_topic"]:
                    single_assumed.append(float((pred == r["assumed_label_id"]).mean()))
            if not true_all:
                res[name] = None
                continue
            T = np.concatenate(true_all)
            P = np.concatenate(pred_all)
            res[name] = {
                "true_label_token_acc": float((T == P).mean()),
                "true_label_macro_iou": iou_macro(T, P),
                "single_assumed_acc_mean": float(np.mean(single_assumed))
                if single_assumed
                else None,
                "n_records": len(true_all),
            }
        return res

    held = [r for r in per_record if r["held_out"]]
    metrics = {
        "arm": "gpt-reference-raw",
        "model": MODEL_ID,
        "attn": attn,
        "test_records": len(per_record),
        "n_single": sum(r["single_topic"] for r in per_record),
        "n_held_out": len(held),
        "all_1000": summarise(per_record),
        "held_out_only": summarise(held),
    }
    (outdir / "decode_gpt_ref_raw.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with (outdir / "decode_gpt_ref_raw_per_record.jsonl").open("w") as f:
        for r in per_record:
            f.write(json.dumps(r) + "\n")
    print(
        json.dumps(
            {"all_1000": metrics["all_1000"], "held_out_only": metrics["held_out_only"]}, indent=2
        )
    )
    print(f"\nwrote {outdir / 'decode_gpt_ref_raw.json'}\nGPT REF ARM DONE", flush=True)


if __name__ == "__main__":
    main()
