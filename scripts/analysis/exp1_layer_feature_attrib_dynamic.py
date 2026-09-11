#!/usr/bin/env python3
"""Experiment 1, DYNAMIC arm: which layers'/features' SAE signal drives the prediction?

Dynamic counterpart of exp1_layer_feature_attrib.py. Same three views (layer knockout,
grad x input attribution, attribution-guided feature knockout) but on the per-token
top-K dynamic representation, using the dyn150 checkpoints and the sparse GPU cache
(densified per record). Layers have UNEQUAL widths, so per-layer aggregation uses the
explicit layer_cols ranges instead of a fixed reshape.

This is the natural place to test the report's OPEN question: does the classifier
concentrate its reliance on a narrow set of dynamic columns, or genuinely spread across
the ~10x-wider vocabulary? Part C's top-vs-bottom knockout answers exactly that.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

# local imports
from important_scripts.paths import PROJECT_ROOT
from important_scripts.analysis import analysis_common_dynamic as A
from important_scripts.model.models import PAD, TOPICS

ROOT = PROJECT_ROOT

OUT = ROOT / os.environ.get("DYN_ANALYSIS_OUT", "results/analysis_dyn150")
OUT.mkdir(parents=True, exist_ok=True)
RESP = A.RESP


def layer_knockout(model, C, baseline) -> dict:
    """Zero each layer's feature block, re-score, and report the macro-IoU / per-topic drop."""
    res = {"baseline": baseline, "layers": {}}
    for L in C["layers"]:
        t0 = time.time()
        s = A.score(model, C, mask=A.zero_layer_mask(C, L), desc=f"knockout L{L}")
        print(
            f"  layer {L} knockout done in {time.time() - t0:.0f}s "
            f"(ΔIoU {s['macro_iou'] - baseline['macro_iou']:+.4f})",
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


def attribution(model, C) -> dict:
    """Grad x input attribution of each layer/feature to the topic predictions."""
    W = C["width"]
    layers = C["layers"]
    nL = len(layers)
    ranges = [C["layer_cols"][L] for L in layers]
    per_layer = np.zeros(nL)
    per_topic_layer = np.zeros((len(TOPICS), nL))
    per_col_abs = np.zeros(W)
    topic_tokens = np.zeros(len(TOPICS))
    model.eval()
    _n = len(C["test"])
    _t0 = time.time()
    for _i, r in enumerate(C["test"], 1):
        r = int(r)
        xd, lo, hi = A.densify_one(C, r)
        x = xd.clone().requires_grad_(True)  # (T, W)
        with torch.backends.cudnn.flags(enabled=False):
            logits = model(x[None])[0]  # (T, NT)
            lab = C["labels"][lo:hi].clone()
            role = C["roles"][lo:hi]
            keep = (lab != PAD) if A.SUPERVISE_PROMPT else ((role == RESP) & (lab != PAD))
            if keep.sum() == 0:
                continue
            idx = keep.nonzero(as_tuple=True)[0]
            tgt = lab[idx]
            sel = logits[idx, tgt].sum()
            model.zero_grad(set_to_none=True)
            sel.backward()
        attr = (x.grad[idx] * x[idx]).detach().abs()  # (n, W)
        per_col_abs += attr.sum(0).cpu().numpy()
        al = np.stack([attr[:, a:b].sum(1).cpu().numpy() for (a, b) in ranges], axis=1)  # (n, nL)
        per_layer += al.sum(0)
        tt = tgt.cpu().numpy()
        for ci in range(len(TOPICS)):
            m = tt == ci
            if m.any():
                per_topic_layer[ci] += al[m].sum(0)
                topic_tokens[ci] += int(m.sum())
        A._progress("attribution", _i, _n, _t0)
    layer_frac = per_layer / per_layer.sum()
    ptl_frac = per_topic_layer / per_topic_layer.sum(1, keepdims=True).clip(1e-12)
    sae_index = C["sae_index"]
    top_feats = {}
    for i, L in enumerate(layers):
        a, b = ranges[i]
        block = per_col_abs[a:b]
        order = np.argsort(-block)[:10]
        top_feats[L] = [
            {"col": int(a + o), "sae_feature": int(sae_index[a + o]), "abs_attr": float(block[o])}
            for o in order
        ]
    # concentration: how much of total |attr| sits in the top-N columns overall (open question)
    tot = per_col_abs.sum()
    order_all = np.argsort(-per_col_abs)
    cum = np.cumsum(per_col_abs[order_all]) / max(tot, 1e-12)
    conc = {str(n): float(cum[min(n, len(cum)) - 1]) for n in (10, 50, 100, 500, 1000, 5000)}
    nz = int((per_col_abs > 0).sum())
    return {
        "per_layer_abs": per_layer.tolist(),
        "per_layer_frac": {L: float(layer_frac[i]) for i, L in enumerate(layers)},
        "per_topic_layer_frac": {
            TOPICS[ci]: {L: float(ptl_frac[ci, i]) for i, L in enumerate(layers)}
            for ci in range(len(TOPICS))
        },
        "per_col_abs": per_col_abs,
        "top_features_per_layer": top_feats,
        "attr_concentration_topN_frac": conc,
        "nonzero_attr_columns": nz,
        "total_columns": int(W),
    }


def guided_knockout(model, C, baseline, top_layer, per_col_abs) -> dict:
    """Knock out the top attribution-ranked features and measure the resulting score drop."""
    a, b = C["layer_cols"][top_layer]
    block = per_col_abs[a:b]
    order_top = np.argsort(-block)
    order_bot = np.argsort(block)
    out = {
        "layer": top_layer,
        "layer_width": int(b - a),
        "baseline_iou": baseline["macro_iou"],
        "sweep": {},
    }
    for k in [10, 30, 75, 150]:
        res = {}
        for name, order in [("top", order_top), ("bottom", order_bot)]:
            mask = A.full_mask(C)
            cols = a + order[:k]
            mask[torch.as_tensor(cols, device=A.DEVICE)] = 0.0
            s = A.score(model, C, mask=mask, desc=f"guided {name}-{k}")
            res[name] = {
                "macro_iou": s["macro_iou"],
                "d_iou": s["macro_iou"] - baseline["macro_iou"],
            }
        out["sweep"][k] = res
    return out


def main() -> None:
    """Run the three attribution views (layer knockout, grad x input, guided knockout) for every detector."""
    C = A.load_cache()
    A.check_supervision(C)
    all_out = {}
    requested = os.environ.get("DYN_ANALYSIS_FAMILIES", "GRU,Transformer,ConvNeXt")
    families = [family.strip() for family in requested.split(",") if family.strip()]
    unknown = set(families) - {"GRU", "Transformer", "ConvNeXt"}
    if unknown:
        raise ValueError(f"unknown analysis families: {sorted(unknown)}")
    print(f"analysing families: {', '.join(families)}", flush=True)
    for fam in families:
        t0 = time.time()
        model = A.load_model(fam)
        print(f"[{fam}] scoring baseline ...", flush=True)
        baseline = A.score(model, C, desc="baseline")
        print(
            f"[{fam}] baseline macro_iou={baseline['macro_iou']:.4f} "
            f"macro_f1={baseline['macro_f1']:.4f} in {time.time() - t0:.0f}s; layer knockout ...",
            flush=True,
        )
        ko = layer_knockout(model, C, baseline)
        print(
            f"[{fam}] layer knockout done ({time.time() - t0:.0f}s elapsed); attribution ...",
            flush=True,
        )
        attr = attribution(model, C)
        print(
            f"[{fam}] attribution done ({time.time() - t0:.0f}s elapsed); guided knockout ...",
            flush=True,
        )
        per_col_abs = attr.pop("per_col_abs")
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
            f"attr-top layer = {max(attr['per_layer_frac'], key=attr['per_layer_frac'].get)} | "
            f"top-100 cols hold {attr['attr_concentration_topN_frac']['100']:.2%} of |attr|",
            flush=True,
        )
    # Merge into any existing results so re-running a subset of families (e.g. just GRU)
    # does not clobber families computed in a previous run.
    out_path = OUT / "exp1_layer_feature_attrib.json"
    if out_path.exists():
        existing = json.load(open(out_path))
        existing.update(all_out)  # newly computed families overwrite their own stale entries
        all_out = existing
    json.dump(all_out, open(out_path, "w"), indent=2)
    print(f"\nwrote {out_path} (families: {', '.join(all_out)})")

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
    print(
        "\n===== CONCENTRATION: cumulative |attr| fraction in top-N columns (of "
        f"{all_out[next(iter(all_out))]['attribution']['total_columns']}) ====="
    )
    print(f"{'topN':>8s}" + "".join(f"{f:>14s}" for f in all_out))
    for n in ("10", "50", "100", "500", "1000", "5000"):
        print(
            f"{n:>8s}"
            + "".join(
                f"{all_out[f]['attribution']['attr_concentration_topN_frac'][n]:>14.3f}"
                for f in all_out
            )
        )


if __name__ == "__main__":
    main()
