#!/usr/bin/env python3
"""Experiment 1: which layers'/features' SAE signal drives the topic prediction?

Three complementary views, all on the fixed SAE-150 arm, response tokens, test split,
for GRU / Transformer / ConvNeXt:

  A. Layer knockout. Zero each layer's 150-column block and re-score. The drop in
     macro-IoU / per-topic-IoU vs the intact baseline measures how much that layer
     (SAE) contributes to the decision.

  B. Gradient x input attribution. For every response token, attribute its true-topic
     logit back onto the input features (grad * value), aggregate |attr| per layer and
     per (topic, layer). This is the fine-grained "which layer feeds which topic".

  C. Attribution-guided feature knockout. Within the single most-important layer (by A),
     zero the top-k attributed features vs the bottom-k, sweeping k, to confirm the
     attribution actually locates the load-bearing features.

Serves both backbones: the layer range and cache layout come from KNOCKOUT_CACHE's metadata via
analysis_common_static. Pick detectors with --families; point at specific checkpoints with
KNOCKOUT_CKPT_DIR plus optional per-family KNOCKOUT_CKPT_<FAMILY> overrides.
Results MERGE into any existing output json, so a later GRU-only run adds to an earlier
Transformer+ConvNeXt run rather than replacing it. Merged from the 1B/4B twins; see MERGE.md.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

# local imports
from important_scripts.analysis import analysis_common_static as A
from important_scripts.model.models import PAD, TOPICS

# The loader owns the profile: the cache named in the environment fixes the backbone, the layer
# range and the column layout, and ROOT is the data root it resolved.
ROOT = A.ROOT
OUT = ROOT / A._required("KNOCKOUT_OUT", "results/analysis_sae150_1b10k_pr")
OUT.mkdir(parents=True, exist_ok=True)
RESP = A.RESP


# ---------- A. layer knockout ----------
def layer_knockout(model, C, baseline) -> dict:
    """Zero each layer's feature block, re-score, and report the macro-IoU / per-topic drop."""
    res = {"baseline": baseline, "layers": {}}
    print(
        f"    [knockout] baseline IoU={baseline['macro_iou']:.4f}; zeroing each of {len(C['layers'])} layers",
        flush=True,
    )
    for i, L in enumerate(C["layers"]):
        t = time.time()
        s = A.score(model, C, mask=A.zero_layer_mask(C, L))
        print(
            f"    [knockout] layer {L} ({i + 1}/{len(C['layers'])}) ΔIoU={s['macro_iou'] - baseline['macro_iou']:+.4f} ({time.time() - t:.1f}s)",
            flush=True,
        )
        res["layers"][L] = {
            "macro_f1": s["macro_f1"],
            "macro_iou": s["macro_iou"],
            "acc": s["acc"],
            "d_macro_iou": s["macro_iou"] - baseline["macro_iou"],
            "d_macro_f1": s["macro_f1"] - baseline["macro_f1"],
            "per_topic_iou": {t: s["per_topic"][t]["iou"] for t in TOPICS},
            "d_per_topic_iou": {
                t: s["per_topic"][t]["iou"] - baseline["per_topic"][t]["iou"] for t in TOPICS
            },
        }
    return res


# ---------- B. gradient x input attribution ----------
def attribution(model, C) -> dict:
    """Grad x input attribution of each layer/feature to the topic predictions."""
    W = C["width"]
    nL = len(C["layers"])
    # Explicit per-layer column ranges: group-sparse caches have unequal layer widths, so a
    # reshape into (nL, per_layer) blocks would mis-assign every column after the first short layer.
    ranges = [C["layer_cols"][L] for L in C["layers"]]
    # accumulators
    per_layer = np.zeros(nL)  # sum |attr| by layer
    per_topic_layer = np.zeros((len(TOPICS), nL))  # sum |attr| by (topic, layer)
    per_col_abs = np.zeros(W)  # sum |attr| by feature column
    # Signed companions. Taking |attr| before summing destroys the completeness property that
    # makes the per-layer total meaningful (it then tracks activation magnitude, not importance),
    # so the signed sum is accumulated alongside and reported as its own field. The abs fields
    # stay as they are, because every earlier arm is reported on them.
    per_layer_signed = np.zeros(nL)
    per_topic_layer_signed = np.zeros((len(TOPICS), nL))
    topic_tokens = np.zeros(len(TOPICS))
    model.eval()
    ntest = len(C["test"])
    _t = time.time()
    print(f"    [attribution] grad×input over {ntest} test records...", flush=True)
    for _i, r in enumerate(C["test"]):
        if _i and _i % 200 == 0:
            print(
                f"    [attribution] {_i}/{ntest} ({(time.time() - _t) / _i * 1000:.0f} ms/rec)",
                flush=True,
            )
        r = int(r)
        lo, hi = int(C["offsets"][r]), int(C["offsets"][r + 1])
        x = (
            C["features"][lo:hi].float().to(A.DEVICE).clone().requires_grad_(True)
        )  # (T, W) slice->GPU
        # cuDNN RNN backward is training-mode only; disable cuDNN so the GRU can be
        # differentiated in eval mode (no dropout) for clean attribution.
        with torch.backends.cudnn.flags(enabled=False):
            logits = model(x[None])[0]  # (T, Cn)
            lab = C["labels"][lo:hi].clone()
            role = C["roles"][lo:hi]
            keep = (lab != PAD) if A.SUPERVISE_PROMPT else ((role == RESP) & (lab != PAD))
            if keep.sum() == 0:
                continue
            idx = keep.nonzero(as_tuple=True)[0]
            tgt = lab[idx]  # true topic per resp token
            sel = logits[idx, tgt].sum()  # scalar
            model.zero_grad(set_to_none=True)
            sel.backward()
        attr = (x.grad[idx] * x[idx]).detach()  # (n, W) grad x input
        aabs = attr.abs()
        # per column
        per_col_abs += aabs.sum(0).cpu().numpy()
        # per layer, by explicit column range
        al = np.stack([aabs[:, a:b].sum(1).cpu().numpy() for (a, b) in ranges], axis=1)  # (n, nL)
        als = np.stack([attr[:, a:b].sum(1).cpu().numpy() for (a, b) in ranges], axis=1)
        per_layer += al.sum(0)
        per_layer_signed += als.sum(0)
        # per (topic, layer)
        tt = tgt.cpu().numpy()
        for ci in range(len(TOPICS)):
            m = tt == ci
            if m.any():
                per_topic_layer[ci] += al[m].sum(0)
                per_topic_layer_signed[ci] += als[m].sum(0)
                topic_tokens[ci] += int(m.sum())
    # normalize
    layer_frac = per_layer / per_layer.sum()
    ptl_frac = per_topic_layer / per_topic_layer.sum(1, keepdims=True).clip(1e-12)
    abs_signed = np.abs(per_layer_signed)
    layer_frac_signed = abs_signed / max(abs_signed.sum(), 1e-12)
    ptl_signed = np.abs(per_topic_layer_signed)
    ptl_frac_signed = ptl_signed / ptl_signed.sum(1, keepdims=True).clip(1e-12)
    # top features per layer
    sae_index = C["sae_index"]
    top_feats = {}
    for i, L in enumerate(C["layers"]):
        a, b = ranges[i]
        block = per_col_abs[a:b]
        order = np.argsort(-block)[:10]
        top_feats[L] = [
            {
                "col": int(a + o),
                "sae_feature": int(sae_index[a + o]),
                "abs_attr": float(block[o]),
            }
            for o in order
        ]
    return {
        "per_layer_abs": per_layer.tolist(),
        "per_layer_frac": {L: float(layer_frac[i]) for i, L in enumerate(C["layers"])},
        "per_topic_layer_frac": {
            TOPICS[ci]: {L: float(ptl_frac[ci, i]) for i, L in enumerate(C["layers"])}
            for ci in range(len(TOPICS))
        },
        "layer_width": {L: int(b - a) for L, (a, b) in zip(C["layers"], ranges, strict=True)},
        "per_layer_signed": per_layer_signed.tolist(),
        "per_layer_frac_signed": {
            L: float(layer_frac_signed[i]) for i, L in enumerate(C["layers"])
        },
        "per_topic_layer_frac_signed": {
            TOPICS[ci]: {L: float(ptl_frac_signed[ci, i]) for i, L in enumerate(C["layers"])}
            for ci in range(len(TOPICS))
        },
        "per_col_abs": per_col_abs,  # kept for part C (not serialized directly)
        "top_features_per_layer": top_feats,
    }


# ---------- C. attribution-guided knockout in the top layer ----------
def guided_knockout(model, C, baseline, top_layer, per_col_abs) -> dict:
    """Knock out the top attribution-ranked features and measure the resulting score drop."""
    base_lo, base_hi = C["layer_cols"][top_layer]
    width = base_hi - base_lo
    block = per_col_abs[base_lo:base_hi]
    order_top = np.argsort(-block)  # most important first
    order_bot = np.argsort(block)  # least important first
    out = {
        "layer": top_layer,
        "layer_width": int(width),
        "baseline_iou": baseline["macro_iou"],
        "sweep": {},
    }
    # A short layer cannot give up 150 columns; the last rung is its full width, which keeps the
    # sweep from silently reporting the whole-layer knockout twice under two different k.
    for k in [k for k in (10, 30, 75, 150) if k < width] + [min(150, width)]:
        res = {}
        for name, order in [("top", order_top), ("bottom", order_bot)]:
            mask = A.full_mask(C)
            cols = base_lo + order[:k]
            mask[torch.as_tensor(cols, device=A.DEVICE)] = 0.0
            s = A.score(model, C, mask=mask)
            res[name] = {
                "macro_iou": s["macro_iou"],
                "d_iou": s["macro_iou"] - baseline["macro_iou"],
            }
        out["sweep"][k] = res
    return out


def resolve_families(argument: str) -> list[str]:
    """Parse and validate the --families list. The flag is the only way to choose detectors."""
    families = [name.strip() for name in argument.split(",") if name.strip()]
    unknown = [name for name in families if name not in A.FAMILIES]
    if unknown:
        raise SystemExit(f"unknown detector families {unknown}; choose from {list(A.FAMILIES)}")
    return families


def main() -> None:
    """Run the three attribution views (layer knockout, grad x input, guided knockout) per detector."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--families",
        default=",".join(A.FAMILIES),
        help=f"comma-separated detectors to knock out, from {list(A.FAMILIES)} (default: all)",
    )
    args = parser.parse_args()
    fams = resolve_families(args.families)
    outfile = OUT / "exp1_layer_feature_attrib.json"
    all_out = json.load(open(outfile)) if outfile.exists() else {}
    print(f"running knockout for: {fams}  (already in {outfile.name}: {list(all_out)})", flush=True)
    print(A.describe(fams), flush=True)
    missing = [f for f in fams if not A.CKPT[f].exists()]
    if missing:
        raise SystemExit(f"missing checkpoints for {missing}; set KNOCKOUT_CKPT_DIR or KNOCKOUT_CKPT_<FAMILY>")
    C = A.load_cache()
    A.check_supervision(C)
    for fam in fams:
        print(f"\n### {fam} ###", flush=True)
        t0 = time.time()
        model = A.load_model(fam)
        baseline = A.score(model, C)
        ko = layer_knockout(model, C, baseline)
        attr = attribution(model, C)
        per_col_abs = attr.pop("per_col_abs")
        # rank layers by knockout IoU damage (most negative delta = most important)
        dmg = {L: -ko["layers"][L]["d_macro_iou"] for L in C["layers"]}
        top_layer = max(dmg, key=dmg.get)
        guided = guided_knockout(model, C, baseline, top_layer, per_col_abs)
        all_out[fam] = {
            "baseline": baseline,
            "layer_knockout": ko,
            "attribution": attr,
            "top_layer_by_knockout": top_layer,
            "guided_knockout": guided,
        }
        print(
            f"{fam:12s} done in {time.time() - t0:.1f}s | top layer by knockout = {top_layer} "
            f"(ΔIoU {ko['layers'][top_layer]['d_macro_iou']:+.4f}) | "
            f"attr-top layer = {max(attr['per_layer_frac'], key=attr['per_layer_frac'].get)}",
            flush=True,
        )
    json.dump(all_out, open(OUT / "exp1_layer_feature_attrib.json", "w"), indent=2)
    print(f"\nwrote {OUT / 'exp1_layer_feature_attrib.json'}")

    # compact console summary: per-layer knockout ΔIoU
    # merged-from-JSON families carry string layer keys; coerce so the int-indexed summary
    # prints cleanly regardless of which families were freshly computed vs. loaded.
    for _f in all_out:
        _lk = all_out[_f]["layer_knockout"]["layers"]
        all_out[_f]["layer_knockout"]["layers"] = {int(_k): _v for _k, _v in _lk.items()}
        _pl = all_out[_f]["attribution"]["per_layer_frac"]
        all_out[_f]["attribution"]["per_layer_frac"] = {int(_k): _v for _k, _v in _pl.items()}
    print("\n===== LAYER KNOCKOUT: ΔMacro-IoU when each layer's SAE block is zeroed =====")
    print(f"{'layer':>6s}" + "".join(f"{f:>14s}" for f in all_out))
    for L in C["layers"]:
        print(
            f"{L:>6d}"
            + "".join(
                f"{all_out[f]['layer_knockout']['layers'][L]['d_macro_iou']:>+14.4f}"
                for f in all_out
            )
        )
    print("\n===== ATTRIBUTION: fraction of |grad×input| per layer =====")
    print(f"{'layer':>6s}" + "".join(f"{f:>14s}" for f in all_out))
    for L in C["layers"]:
        print(
            f"{L:>6d}"
            + "".join(f"{all_out[f]['attribution']['per_layer_frac'][L]:>14.3f}" for f in all_out)
        )


if __name__ == "__main__":
    main()
