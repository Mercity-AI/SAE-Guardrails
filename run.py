#!/usr/bin/env python3
"""Score the published detectors on the test split, straight from the Hugging Face Hub.

Downloads the 1,500 test records and each detector, runs every record through Gemma 3 once
(teacher-forced), rebuilds the features each detector reads and prints test macro-F1 and overlap
next to the numbers recorded at training time. Needs a GPU and access to the gated Gemma 3 weights.

    python run.py --model-size 1b
    python run.py --model-size 4b --arms dynamic,raw --limit 50
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer

# local imports
from common.data import seed_everything
from common.labels import label_record, label_record_pr
from common.metrics import metrics
from models import build_model
from preprocessing.build_static_cache import find_backbone, jumprelu

DEVICE = "cuda"
DATASET = "Mercity/topic-shift-10k"
REPOS = {
    "summation": "Mercity/online-guardrails-sae-summation",
    "fstatistic": "Mercity/online-guardrails-sae-fstatistic",
    "grouplasso": "Mercity/online-guardrails-sae-grouplasso",
    "dynamic": "Mercity/online-guardrails-sae-dynamic",
    "raw": "Mercity/online-guardrails-raw-hidden",
}
BACKBONES = {  # model, SAE release, the ten layers the detectors read
    "1b": ("google/gemma-3-1b-it", "gemma-scope-2-1b-it-res-all", list(range(16, 26))),
    "4b": ("google/gemma-3-4b-it", "gemma-scope-2-4b-it-res-all", list(range(24, 34))),
}
# F-statistic and group lasso were labelled by their cache builder; everything else by label_record_pr.
LABELERS = {
    "fstatistic": partial(label_record, supervise_prompt=True),
    "grouplasso": partial(label_record, supervise_prompt=True),
}


def load_detector(arm: str, size: str, layers: list[int]) -> dict:
    """Download one published detector, its recorded test scores and the SAE columns it reads."""

    def get(name: str) -> str:
        return hf_hub_download(REPOS[arm], f"gemma-3-{size}/{name}")

    checkpoint = torch.load(get("checkpoint_best.pt"), map_location="cpu", weights_only=False)
    model = build_model(checkpoint["architecture"], checkpoint["model"])
    model.load_state_dict(checkpoint["state_dict"])
    detector = {
        "model": model.to(DEVICE).eval(),
        "architecture": checkpoint["architecture"],
        "width": int(checkpoint["model"]["input_features"]),
        "recorded": json.loads(Path(get("results.json")).read_text())["test"],
        "rows": [],
    }
    if arm != "raw":
        selected = np.load(get("selected_features.npz"))
        detector["columns"] = {L: torch.as_tensor(selected[f"layer_{L}"], device=DEVICE) for L in layers}
    if arm == "dynamic":
        detector["k"] = int(json.loads(Path(get("meta.json")).read_text())["K"])
    return detector


def load_saes(release: str, layers: list[int]) -> dict:
    """Gemma Scope parameters per layer, as the (W_enc, b_enc, b_dec, threshold) that jumprelu takes."""
    saes = {}
    for L in layers:
        sae = SAE.from_pretrained(release=release, sae_id=f"layer_{L}_width_16k_l0_small", device=DEVICE)
        sae = sae[0] if isinstance(sae, tuple) else sae
        saes[L] = tuple(p.float() for p in (sae.W_enc, sae.b_enc, sae.b_dec, sae.threshold))
    return saes


def features(arm: str, detector: dict, hidden: dict, saes: dict, layers: list[int]) -> torch.Tensor:
    """Turn one record's hidden states into the (tokens, width) input this detector was trained on."""
    if arm == "raw":
        return torch.cat([hidden[L] for L in layers], dim=-1)
    if arm != "dynamic":  # static: the fixed selected SAE columns of every layer
        columns = []
        for L in layers:
            W, b_enc, b_dec, threshold = saes[L]
            c = detector["columns"][L]
            columns.append(jumprelu(hidden[L], W[:, c], b_enc[c], b_dec, threshold[c]))
        return torch.cat(columns, dim=-1)
    # dynamic: each token's top-K features per layer, placed in its training-vocabulary column
    tokens = hidden[layers[0]].shape[0]
    x = torch.zeros(tokens, detector["width"], device=DEVICE)
    offset = 0
    for L in layers:
        values, ids = torch.topk(jumprelu(hidden[L], *saes[L]), detector["k"], dim=-1)
        vocab = detector["columns"][L]
        lookup = torch.full((saes[L][0].shape[1],), -1, dtype=torch.long, device=DEVICE)
        lookup[vocab] = torch.arange(len(vocab), device=DEVICE)
        column = lookup[ids]
        keep = (values > 0) & (column >= 0)
        rows = torch.arange(tokens, device=DEVICE)[:, None].expand_as(ids)
        x[rows[keep], offset + column[keep]] = values[keep].half().float()  # the cache stored fp16
        offset += len(vocab)
    return x


@torch.inference_mode()
def evaluate(size: str, arms: list[str], records: list[dict], attention: str) -> dict:
    """Score every requested detector on one backbone, with one teacher-forced pass per record."""
    model_id, release, layers = BACKBONES[size]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    gemma = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation=attention
    )
    backbone = find_backbone(gemma.to(DEVICE).eval())
    saes = load_saes(release, layers) if arms != ["raw"] else {}
    detectors = {arm: load_detector(arm, size, layers) for arm in arms}

    # Hook the layer outputs, as the cache builders did (the last entry of hidden_states is normed).
    hidden: dict[int, torch.Tensor] = {}
    for L in layers:
        backbone.layers[L].register_forward_hook(
            lambda _m, _i, out, L=L: hidden.__setitem__(
                L, (out[0] if isinstance(out, tuple) else out)[0].float()
            )
        )

    for i, record in enumerate(records, 1):
        labelled = {arm: LABELERS.get(arm, label_record_pr)(record, tokenizer) for arm in arms}
        ids = next(iter(labelled.values()))[0]
        backbone(input_ids=torch.tensor([list(ids)], device=DEVICE), use_cache=False)
        for arm, detector in detectors.items():
            x = features(arm, detector, hidden, saes, layers)[None]
            # the dynamic Transformer and ConvNeXt were trained under bf16 autocast
            bf16 = arm == "dynamic" and detector["architecture"] != "gru"
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                predictions = detector["model"](x.bfloat16() if bf16 else x).argmax(-1)[0]
            labels = np.asarray(labelled[arm][2], dtype=np.int64)
            detector["rows"].append({"labels": labels, "predictions": predictions.cpu().numpy()})
        if i % 100 == 0:
            print(f"  [{size}] {i}/{len(records)}", flush=True)

    report = {}
    for arm, detector in detectors.items():
        result = metrics(detector["rows"])
        report[arm] = {
            "detector": detector["architecture"],
            "macro_f1": result["macro_f1"],
            "macro_iou": result["topic_overlap"]["macro_iou"],
            "recorded_macro_f1": detector["recorded"]["macro_f1"],
            "recorded_macro_iou": detector["recorded"]["topic_overlap"]["macro_iou"],
        }
    return report


def main() -> None:
    """Evaluate the requested backbones and detectors, print a table and save it as json."""
    parser = argparse.ArgumentParser(description="Score the published detectors on the test split.")
    parser.add_argument("--model-size", choices=["1b", "4b", "both"], default="both")
    parser.add_argument("--arms", default=",".join(REPOS), help=f"comma-separated subset of {list(REPOS)}")
    parser.add_argument("--limit", type=int, default=0, help="score only the first N test records")
    parser.add_argument("--attention", default="sdpa", help="Gemma attention implementation")
    parser.add_argument("--out", type=Path, default=Path("results/hf_detector_eval.json"))
    args = parser.parse_args()

    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    if unknown := sorted(set(arms) - set(REPOS)):
        raise SystemExit(f"unknown arms {unknown}; choose from {list(REPOS)}")
    seed_everything(42, deterministic=False)
    records = list(load_dataset(DATASET, split="test"))[: args.limit or None]
    print(f"{len(records)} test records from {DATASET}", flush=True)

    sizes = ["1b", "4b"] if args.model_size == "both" else [args.model_size]
    report = {size: evaluate(size, arms, records, args.attention) for size in sizes}

    print(f"\n{'size':4} {'arm':11} {'detector':11} {'F1':>7} {'recorded':>8} {'IoU':>7} {'recorded':>8}")
    for size, rows in report.items():
        for arm, r in rows.items():
            print(
                f"{size:4} {arm:11} {r['detector']:11} {100 * r['macro_f1']:7.2f} {100 * r['recorded_macro_f1']:8.2f}"
                f" {100 * r['macro_iou']:7.2f} {100 * r['recorded_macro_iou']:8.2f}"
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
