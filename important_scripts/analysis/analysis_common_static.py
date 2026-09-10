"""Static SAE-150 knockout loader: GPU-resident scorer plus maskable feature columns.

Serves both backbones. The layer range and features-per-layer are read from the cache's own
metadata.json rather than hardcoded, so the same module drives the 1B (layers 16-25) and 4B
(layers 24-33) arms, and also the 450-per-layer group-sparse caches. KNOCKOUT_CACHE and
KNOCKOUT_CKPT_DIR are required: there is no single sensible default across backbones, and the
old per-model defaults pointed at response-only caches, which is how a PR run could silently
score the wrong token set. Merged from analysis_common_1b10k.py and analysis_common_4b10k.py;
see MERGE.md for the variables that replaced their constants.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


# local imports
from important_scripts.paths import PROJECT_ROOT
from important_scripts.model.models import PAD, TOPICS, build_model, topic_overlap

ROOT = PROJECT_ROOT

def _required(name: str, examples: str) -> str:
    """Read a required environment variable, naming the known-good values when it is missing."""
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"{name} is required (e.g. {examples})")
    return value


CACHE_DIR = ROOT / _required(
    "KNOCKOUT_CACHE", "cache/sae1500_10k_1b_pr or cache/sae1500_10k_gemma4b_pr"
)
_META = json.loads((CACHE_DIR / "metadata.json").read_text())
DEVICE = "cuda"
# Where the large (~25 GiB) feature matrix lives. Default "cuda" keeps the original
# GPU-resident behavior; set KNOCKOUT_FEAT_DEVICE=cpu to hold it in host RAM and stream
# per-record slices to the GPU, so the run's GPU footprint stays a few GiB and leaves
# the rest of the card free for other jobs.
FEAT_DEVICE = os.environ.get("KNOCKOUT_FEAT_DEVICE", DEVICE)
RESP = 2
# prompt+response scoring drops the response-only mask (scores all labeled topic tokens,
# prompt spans included) to match the PR baselines; default stays response-only.
_SUPERVISE_RAW = os.environ.get("KNOCKOUT_SUPERVISE", "")
SUPERVISE_PROMPT = _SUPERVISE_RAW.lower() in ("prompt_response", "pr", "both")


def check_supervision(C) -> None:
    """Refuse to score a prompt+response cache response-only by accident.

    Scoring masks prompt tokens unless KNOCKOUT_SUPERVISE names a prompt+response mode. A PR
    cache carries labeled prompt spans, so the unset default would silently score a smaller
    token set than the PR baselines did and the numbers would not be comparable. Set
    KNOCKOUT_SUPERVISE=pr to score prompt+response, or =response to accept the mask on purpose.
    """
    if SUPERVISE_PROMPT or _SUPERVISE_RAW:
        return
    prompt_labeled = int(((C["labels"] != PAD) & (C["roles"] != RESP)).sum())
    if prompt_labeled:
        raise RuntimeError(
            f"cache carries {prompt_labeled} labeled prompt tokens but KNOCKOUT_SUPERVISE is "
            "unset, so prompt spans would be masked and the scores would not match the PR "
            "baselines. Set KNOCKOUT_SUPERVISE=pr (prompt+response) or =response (accept mask)."
        )
# Layout comes from the cache itself, so the module never disagrees with the features it reads.
LAYERS = [int(layer) for layer in _META["layers"]]
PER_LAYER = int(_META["selected_features_per_layer"])
BASE_MODEL = _META.get("base_model", "unknown")
# Layer blocks are addressed by an explicit map, not col//per_layer: a group-sparse cache keeps
# only the features that survive its penalty, so layers differ in width and the total falls below
# per-layer x layers. Builds that predate the variable-width metadata are uniform by construction.
if "layer_cols" in _META:
    LAYER_COLS = {int(layer): tuple(bounds) for layer, bounds in _META["layer_cols"].items()}
else:
    LAYER_COLS = {L: (i * PER_LAYER, (i + 1) * PER_LAYER) for i, L in enumerate(LAYERS)}
WIDTH = max(hi for _, hi in LAYER_COLS.values())
UNIFORM_LAYER_WIDTH = bool(_META.get("uniform_layer_width", True))
if sorted(LAYER_COLS) != sorted(LAYERS):
    raise RuntimeError(f"layer_cols {sorted(LAYER_COLS)} does not cover layers {sorted(LAYERS)}")
if "total_features" in _META and int(_META["total_features"]) != WIDTH:
    raise RuntimeError(
        f"metadata total_features {_META['total_features']} != layer_cols width {WIDTH}"
    )

_CKPT_DIR = ROOT / _required(
    "KNOCKOUT_CKPT_DIR",
    "results/gemma1b_baselines_sae150_10k_pr or results/gemma4b_baselines_sae150_pr",
)
# Run-name subpaths, not architecture names: a differently named training run needs a different
# path. Override any single family with KNOCKOUT_CKPT_<FAMILY>, absolute or relative to ROOT.
DEFAULT_CKPT_SUBPATHS = {
    "GRU": "gru_lr3e4/checkpoint_best.pt",
    "Transformer": "transformer_wide/checkpoint_best.pt",
    "ConvNeXt": "convnext_window505/checkpoint_best.pt",
}
FAMILIES = tuple(DEFAULT_CKPT_SUBPATHS)


def _resolve_ckpt(family: str, subpath: str) -> Path:
    """Checkpoint path for one detector family, honouring a per-family env override."""
    override = os.environ.get(f"KNOCKOUT_CKPT_{family.upper()}", "")
    if override:
        candidate = Path(override)
        return candidate if candidate.is_absolute() else ROOT / candidate
    return _CKPT_DIR / subpath


CKPT = {family: _resolve_ckpt(family, sub) for family, sub in DEFAULT_CKPT_SUBPATHS.items()}


def describe(families=None) -> str:
    """One-line-per-setting summary of everything this module resolved, for the run log."""
    lines = [
        f"cache      {CACHE_DIR}",
        f"base model {BASE_MODEL}  layers {LAYERS[0]}-{LAYERS[-1]}  per-layer {PER_LAYER}"
        + (f"  width {WIDTH}" if UNIFORM_LAYER_WIDTH else f"  width {WIDTH} (variable per layer: "
           + ",".join(f"{L}:{hi - lo}" for L, (lo, hi) in sorted(LAYER_COLS.items())) + ")"),
        f"supervise  {'prompt+response' if SUPERVISE_PROMPT else 'response only'}",
    ]
    for family in families or FAMILIES:
        path = CKPT[family]
        lines.append(f"ckpt {family:12s} {path}{'' if path.exists() else '   [MISSING]'}")
    return "\n".join("  " + line for line in lines)


def load_cache(device: str = DEVICE) -> dict:
    """Load the SAE-150 feature cache GPU-resident (fp16 features + labels/roles/split/layout)."""
    # The 1B cache is stored fp16 and the 4B fp32; copy=False makes this a no-op on the former
    # and the essential 49->25 GiB downcast on the latter, so one line covers both backbones.
    feats = np.asarray(np.load(CACHE_DIR / "features.npy")).astype(np.float16, copy=False)
    labels = np.load(CACHE_DIR / "labels.npy").astype(np.int64)
    roles = np.load(CACHE_DIR / "role_ids.npy").astype(np.int8)
    lengths = np.load(CACHE_DIR / "lengths.npy")
    sp = np.load(CACHE_DIR / "split_indices.npz")
    sel = np.load(CACHE_DIR / "selected_features.npz")
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    sae_index = np.concatenate([sel[f"layer_{L}"].astype(np.int64) for L in LAYERS])
    layer_cols = dict(LAYER_COLS)
    if feats.shape[1] != WIDTH:
        raise RuntimeError(f"features.npy is {feats.shape[1]} wide but layout says {WIDTH}")
    if len(sae_index) != WIDTH:
        raise RuntimeError(f"selected_features holds {len(sae_index)} indices but layout says {WIDTH}")
    return {
        "features": torch.from_numpy(feats).to(FEAT_DEVICE),  # fp16 (N,W) on FEAT_DEVICE (~25 GiB)
        "labels": torch.from_numpy(labels).to(device),
        "roles": torch.from_numpy(roles).to(device),
        "offsets": offsets,
        "test": sp["test"],
        "train": sp["train"],
        "validation": sp["validation"],
        "sae_index": sae_index,
        "width": WIDTH,
        "layers": LAYERS,
        # None when layers differ in width, so any col//per_layer arithmetic fails loudly
        # instead of slicing the wrong block.
        "per_layer": PER_LAYER if UNIFORM_LAYER_WIDTH else None,
        "layer_cols": layer_cols,
    }


def load_model(family: str, device: str = DEVICE):
    """Rebuild a detector (GRU/Transformer/ConvNeXt) from its checkpoint onto the device."""
    p = torch.load(CKPT[family], map_location="cpu", weights_only=False)
    m = build_model(p["architecture"], p["model"])
    m.load_state_dict(p["state_dict"])
    return m.to(device).eval()


def _record_x(C, r, device=DEVICE, mask=None):
    """Return one record's dense feature matrix (batched), optionally with a column mask applied."""
    lo, hi = int(C["offsets"][r]), int(C["offsets"][r + 1])
    x = C["features"][lo:hi].float().to(DEVICE)  # slice -> GPU (no-op if already resident)
    if mask is not None:
        x = x * mask
    return x[None], lo, hi


@torch.inference_mode()
def collect_preds(model, C, mask=None, split="test"):
    """Run a detector over a split and return pooled (truth, prediction) over supervised tokens."""
    T_all, P_all = [], []
    for r in C[split]:
        r = int(r)
        x, lo, hi = _record_x(C, r, mask=mask)
        pred = model(x)[0].argmax(-1).cpu().numpy()
        lab = C["labels"][lo:hi].cpu().numpy().copy()
        role = C["roles"][lo:hi].cpu().numpy()
        if not SUPERVISE_PROMPT:
            lab[role != RESP] = PAD
        keep = lab != PAD
        T_all.append(lab[keep])
        P_all.append(pred[keep])
    return np.concatenate(T_all), np.concatenate(P_all)


def score_from_preds(truth, predk) -> dict:
    """Compute accuracy, macro-F1 and per-topic IoU/Dice/recall from pooled truth/preds."""
    ov = topic_overlap(truth, predk)
    per = {}
    for t in TOPICS:
        cid = TOPICS.index(t)
        tm = truth == cid
        per[t] = {
            "iou": ov["per_topic"][t]["iou"],
            "dice": ov["per_topic"][t]["dice"],
            "recall": float((predk[tm] == cid).mean()) if tm.any() else 0.0,
            "support": int(tm.sum()),
        }
    return {
        "acc": float(accuracy_score(truth, predk)),
        "macro_f1": float(f1_score(truth, predk, average="macro", zero_division=0)),
        "macro_iou": ov["macro_iou"],
        "macro_dice": ov["macro_dice"],
        "per_topic": per,
    }


@torch.inference_mode()
def score(model, C, mask=None, split="test") -> dict:
    """Score a detector on a split, optionally zeroing a feature-column mask first."""
    truth, predk = collect_preds(model, C, mask=mask, split=split)
    return score_from_preds(truth, predk)


def full_mask(C, device=DEVICE):
    """All-ones column mask (nothing zeroed)."""
    return torch.ones(C["width"], device=device)


def zero_layer_mask(C, layer: int, device=DEVICE):
    """Column mask that zeros one layer's feature block."""
    m = torch.ones(C["width"], device=device)
    lo, hi = C["layer_cols"][layer]
    m[lo:hi] = 0.0
    return m
