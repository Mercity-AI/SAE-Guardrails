#!/usr/bin/env python3
"""Raw-activation layer knockout: which residual layer does the raw (no-SAE) detector rely on?

The SAE layer-knockout reads a cached feature matrix, but the raw arm is never cached (it is
streamed). So this streams the base model over the test split, builds the raw feature vector
(concatenation of the ten residual layers) once per record with a single forward, then for each
detector scores the intact vector and each single-layer-zeroed variant. Scoring is two-sided
(every labelled topic token), matching the baselines and the SAE knockout convention.

  python evals/knockout/knockout_raw.py --model google/gemma-3-1b-it --layers 16-25 \
      --ckpt-dir ablation/nosae_stream_1b_10k_pr/raw --pr-labels cache/sae1500_10k_1b_pr \
      --split cache/sae1500_10k_1b_pr --dataset runs/GPT_5.6_10k_2108/dataset.final.jsonl \
      --out results/analysis_raw_1b10k_pr
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

# local imports
from common.labels import label_record
from common.metrics import topic_overlap
from common.paths import PROJECT_ROOT
from models import PAD, build_model

ROOT = PROJECT_ROOT

DEVICE = "cuda"


def find_backbone(model):
    """Return the text transformer that owns ``.layers``, across the Gemma 3 model classes."""
    for c in (
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(model, "language_model", None),
        getattr(model, "model", None),
    ):
        if c is not None and hasattr(c, "layers"):
            return c
    raise TypeError("could not locate the text backbone")


def main(a):
    """Score each raw detector on the test split intact and with each layer zeroed, and save the drops."""
    lo, hi = a.layers.split("-")
    layers = list(range(int(lo), int(hi) + 1))
    nL = len(layers)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    recs = [json.loads(l)["record"] for l in (ROOT / a.dataset).open()]
    ids_all, lengths = [], []
    for r in recs:
        ii, _, _ = label_record(r, tok)
        ids_all.append(ii)
        lengths.append(len(ii))
    lengths = np.asarray(lengths, np.int64)

    pr = np.load(ROOT / a.pr_labels / "labels.npy").astype(np.int64)
    if len(pr) != int(lengths.sum()):
        raise SystemExit(f"pr-labels {len(pr)} != tokens {int(lengths.sum())} (tokenisation drift)")
    offs = np.concatenate([[0], np.cumsum(lengths)])
    labels_all = [pr[offs[k] : offs[k + 1]] for k in range(len(recs))]
    test = np.load(ROOT / a.split / "split_indices.npz")["test"]
    print(f"records={len(recs)} test={len(test)} layers={layers}", flush=True)

    model = (
        AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation=a.attn)
        .to(DEVICE)
        .eval()
    )
    backbone = find_backbone(model)
    H = int(getattr(model.config, "text_config", model.config).hidden_size)
    captured: dict[int, torch.Tensor] = {}
    for L in layers:
        backbone.layers[L].register_forward_hook(
            lambda _m, _i, o, L=L: captured.__setitem__(L, o[0] if isinstance(o, tuple) else o)
        )

    dets = {}
    for name, sub in [("GRU", "gru"), ("Transformer", "transformer"), ("ConvNeXt", "convnext")]:
        p = torch.load(ROOT / a.ckpt_dir / sub / "checkpoint_best.pt", map_location="cpu", weights_only=False)
        m = build_model(p["architecture"], p["model"])
        m.load_state_dict(p["state_dict"])
        dets[name] = m.to(DEVICE).eval()

    variants = ["baseline"] + [f"L{L}" for L in layers]
    Tr = {d: {v: [] for v in variants} for d in dets}
    Pr = {d: {v: [] for v in variants} for d in dets}

    t0 = time.time()
    with torch.inference_mode():
        for n, r in enumerate(test, 1):
            r = int(r)
            toks = torch.tensor([ids_all[r]], device=DEVICE)
            captured.clear()
            model(input_ids=toks, use_cache=False, return_dict=True)
            raw = torch.cat([captured[L][0].float() for L in layers], dim=-1)[None]  # (1,T,H*nL)
            lab = labels_all[r]
            keep = lab != PAD
            if keep.sum() == 0:
                continue
            for d, m in dets.items():
                base = m(raw)[0].argmax(-1).cpu().numpy()
                Tr[d]["baseline"].append(lab[keep])
                Pr[d]["baseline"].append(base[keep])
                for li, L in enumerate(layers):
                    masked = raw.clone()
                    masked[:, :, li * H : (li + 1) * H] = 0.0
                    pl = m(masked)[0].argmax(-1).cpu().numpy()
                    Tr[d][f"L{L}"].append(lab[keep])
                    Pr[d][f"L{L}"].append(pl[keep])
            if n % 200 == 0:
                print(f"  {n}/{len(test)} ({(time.time() - t0) / n * 1000:.0f} ms/rec)", flush=True)

    def score(d, v):
        """Macro-IoU and macro-F1 of detector ``d`` under variant ``v`` (intact or one layer zeroed)."""
        tr = np.concatenate(Tr[d][v])
        pr_ = np.concatenate(Pr[d][v])
        ov = topic_overlap(tr, pr_)
        return {
            "macro_iou": ov["macro_iou"],
            "macro_f1": float(f1_score(tr, pr_, average="macro", zero_division=0)),
            "scored_tokens": int(len(tr)),
        }

    result = {}
    for d in dets:
        b = score(d, "baseline")
        lyr = {}
        for L in layers:
            s = score(d, f"L{L}")
            lyr[L] = {"macro_iou": s["macro_iou"], "d_macro_iou": s["macro_iou"] - b["macro_iou"]}
        top = min(lyr, key=lambda L: lyr[L]["d_macro_iou"])
        result[d] = {
            "baseline": b,
            "layers": lyr,
            "top_layer": top,
            "d_macro_iou_at_top": lyr[top]["d_macro_iou"],
        }
        print(
            f"{d:12s} base IoU={b['macro_iou']:.3f} (scored {b['scored_tokens']:,}) "
            f"top layer {top} dIoU={lyr[top]['d_macro_iou']:+.3f}",
            flush=True,
        )

    json.dump(result, open(out / "raw_layer_knockout.json", "w"), indent=2)
    print(f"wrote {out}/raw_layer_knockout.json", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--layers", required=True)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--pr-labels", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--dataset", default="runs/GPT_5.6_10k_2108/dataset.final.jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--attn", default="flash_attention_2")
    main(p.parse_args())
