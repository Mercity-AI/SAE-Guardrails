"""Shared GPU-resident loading + scoring for the DYNAMIC (per-token top-K) analysis.

Dynamic counterpart of analysis_common.py. Differences from the fixed arm:
  * Features are stored sparse (CSR-by-token: col/val/tok_ptr) and densified per record
    on the GPU -- there is no giant dense (N, W) resident tensor.
  * Layers have UNEQUAL widths (the union of ever-picked features per layer, capped), so
    layer blocks are addressed by an explicit layer_cols map, not col//per_layer.
  * Every union column is still a named SAE feature: sae_index[col] -> feature id, and
    col -> layer via layer_cols. So layer-knockout, grad x input attribution, causal
    zero-out and prediction-point ablation all carry over unchanged in spirit.

Scoring reproduces score_dynamic.py exactly (response tokens only, role==2, macro-F1 and
topic_overlap IoU), with an optional per-column mask so any feature set can be zeroed.

DYN_CACHE / DYN_OUT env vars select the cache + checkpoints (default: dyn150).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


# local imports
from important_scripts.paths import PROJECT_ROOT
from important_scripts.model.models import PAD, TOPICS, build_model, topic_overlap

ROOT = PROJECT_ROOT

DEVICE = "cuda"
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

# --- opt-in progress logging (does not change results) -------------------------------------
# Enable with DYN_PROGRESS=1; tune cadence with DYN_PROGRESS_EVERY (records between prints).
_PROG = os.environ.get("DYN_PROGRESS", "") not in ("", "0")
_EVERY = int(os.environ.get("DYN_PROGRESS_EVERY", "50"))


def _progress(desc, i, n, t0):
    """Print a single-line progress update every _EVERY records (and on the last)."""
    if not _PROG or desc is None:
        return
    if i % _EVERY and i != n:
        return
    el = time.time() - t0
    rate = i / el if el > 0 else 0.0
    eta = (n - i) / rate if rate > 0 else float("nan")
    end = "\n" if i == n else "\r"
    print(
        f"  [{desc}] {i}/{n} rec  {el:6.0f}s  {rate:5.1f} rec/s  ETA {eta:5.0f}s",
        end=end,
        flush=True,
    )


CACHE = ROOT / os.environ.get("DYN_CACHE", "cache/dyn150_sparse/sparse.pt")
CKPT_ROOT = ROOT / os.environ.get("DYN_OUT", "results/dyn150_2k")
CKPT = {
    "GRU": CKPT_ROOT / "gru/gru_lr3e4/checkpoint_best.pt",
    "Transformer": CKPT_ROOT / "transformer/transformer_wide/checkpoint_best.pt",
    "ConvNeXt": CKPT_ROOT / "tcn/convnext/checkpoint_best.pt",
}


def load_cache(device: str = DEVICE) -> dict:
    """Load either the legacy torch cache or the 10k disk-backed CSR cache.

    The 10k ``col`` and ``val`` arrays are tens of GiB, so the CSR form stays
    memory-mapped on CPU and only a single record's nonzeros move to the GPU.
    """
    if CACHE.is_dir():
        metadata = json.loads((CACHE / "meta.json").read_text())
        lengths = np.load(CACHE / "lengths.npy", mmap_mode="r")
        offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
        split = np.load(CACHE / "split_indices.npz")
        return {
            "disk_backed": True,
            "col": np.load(CACHE / "col.npy", mmap_mode="r"),
            "val": np.load(CACHE / "val.npy", mmap_mode="r"),
            "tok_ptr": np.load(CACHE / "tok_ptr.npy", mmap_mode="r"),
            "labels": torch.from_numpy(np.load(CACHE / "labels.npy")).to(device),
            "roles": torch.from_numpy(np.load(CACHE / "roles.npy")).to(device),
            "offsets": offsets,
            "test": split["test"],
            "train": split["train"],
            "validation": split["validation"],
            "sae_index": np.load(CACHE / "sae_index.npy", mmap_mode="r"),
            "width": int(metadata["width"]),
            "layers": [int(layer) for layer in metadata["layers"]],
            "layer_cols": {
                int(layer): tuple(bounds) for layer, bounds in metadata["layer_cols"].items()
            },
        }
    d = torch.load(CACHE, map_location="cpu", weights_only=False)
    lengths = d["lengths"].numpy()
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    # column index (layer + sae feature id): present inline (dyn50) or in an index.pt sidecar (dyn150)
    if "sae_index" in d and "layer_cols" in d:
        sae_index = d["sae_index"].numpy()
        layer_cols = {int(k): tuple(v) for k, v in d["layer_cols"].items()}
        layers = list(d["layers"])
    else:
        idx = torch.load(CACHE.parent / "index.pt", map_location="cpu", weights_only=False)
        sae_index = idx["sae_index"].numpy()
        layer_cols = {int(k): tuple(v) for k, v in idx["layer_cols"].items()}
        layers = list(idx["layers"])
    C = {
        "disk_backed": False,
        "col": d["col"].to(device),
        "val": d["val"].to(device),
        "tok_ptr": d["tok_ptr"].to(device),
        "labels": d["labels"].to(device),
        "roles": d["roles"].to(device),
        "offsets": offsets,
        "test": d["test"].numpy(),
        "train": d["train"].numpy(),
        "validation": d["validation"].numpy(),
        "sae_index": sae_index,
        "width": int(d["width"]),
        "layers": layers,
        "layer_cols": layer_cols,
    }
    return C


def load_model(family: str, device: str = DEVICE):
    """Rebuild a detector (GRU/Transformer/ConvNeXt) from its checkpoint onto the device."""
    p = torch.load(CKPT[family], map_location="cpu", weights_only=False)
    m = build_model(p["architecture"], p["model"])
    m.load_state_dict(p["state_dict"])
    return m.to(device).eval()


def densify_one(C, r, device=DEVICE) -> torch.Tensor:
    """Densify one record to (T, W) fp32 on the GPU from the CSR-by-token cache."""
    lo, hi = int(C["offsets"][r]), int(C["offsets"][r + 1])
    T = hi - lo
    x = torch.zeros(T, C["width"], device=device)
    ptr = C["tok_ptr"][lo : hi + 1]
    nlo, nhi = int(ptr[0]), int(ptr[-1])
    if nhi > nlo:
        if C["disk_backed"]:
            counts = torch.as_tensor(np.asarray(ptr[1:] - ptr[:-1]), device=device)
            columns = torch.as_tensor(np.asarray(C["col"][nlo:nhi]), device=device)
            values = torch.as_tensor(np.asarray(C["val"][nlo:nhi]), device=device).float()
        else:
            counts, columns, values = (
                ptr[1:] - ptr[:-1],
                C["col"][nlo:nhi],
                C["val"][nlo:nhi].float(),
            )
        rows = torch.repeat_interleave(torch.arange(T, device=device), counts)
        x[rows, columns] = values
    return x, lo, hi


@torch.inference_mode()
def collect_preds(model, C, mask=None, split="test", desc=None):
    """Run a detector over a split and return pooled (truth, prediction) over supervised tokens."""
    T_all, P_all = [], []
    ids = C[split]
    n = len(ids)
    t0 = time.time()
    for i, r in enumerate(ids, 1):
        r = int(r)
        x, lo, hi = densify_one(C, r)
        if mask is not None:
            x = x * mask
        pred = model(x[None])[0].argmax(-1).cpu().numpy()
        lab = C["labels"][lo:hi].cpu().numpy().copy()
        role = C["roles"][lo:hi].cpu().numpy()
        if not SUPERVISE_PROMPT:
            lab[role != RESP] = PAD
        keep = lab != PAD
        T_all.append(lab[keep])
        P_all.append(pred[keep])
        _progress(desc, i, n, t0)
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
def score(model, C, mask=None, split="test", desc=None) -> dict:
    """Score a detector on a split, optionally zeroing a feature-column mask first."""
    truth, predk = collect_preds(model, C, mask=mask, split=split, desc=desc)
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


def col_layer(C, col: int) -> int:
    """Return the layer index that owns a given union column."""
    for L, (lo, hi) in C["layer_cols"].items():
        if lo <= col < hi:
            return int(L)
    return -1


def layer_width(C, layer: int) -> int:
    """Return the number of union columns belonging to a layer."""
    lo, hi = C["layer_cols"][layer]
    return hi - lo
