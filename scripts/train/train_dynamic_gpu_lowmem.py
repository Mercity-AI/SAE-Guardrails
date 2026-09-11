#!/usr/bin/env python3
"""Train ONE of GRU / Transformer / ConvNeXt on the dynamic sparse cache — low GPU footprint.

Same training recipe and outputs as train_dynamic_gpu.py, but built to run THREE copies in
parallel on one card without OOM:

  * the big sparse arrays (col, val) stay in CPU RAM; only the small per-record slices are
    moved to the GPU inside densify(). So each process's *resident* GPU cost is just its model
    plus one dense batch (a few GB) instead of the whole ~tens-of-GB cache.
  * DYN_ARCH selects a single architecture (gru|transformer|convnext) so each process trains
    one model; launch three processes to cover all three in parallel.
  * DYN_BS overrides the batch size (keep it low so three concurrent trainers don't saturate
    the GPU).

Env: DYN_CACHE (sparse.pt), DYN_OUT (output root), DYN_ARCH (which model), DYN_BS (batch size).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn


# local imports
from important_scripts.paths import PROJECT_ROOT
from important_scripts.model.models import PAD, TOPICS, build_model, flatten_labeled, metrics, seed_everything

ROOT = PROJECT_ROOT

# The dense per-token input is 54,100-wide; run the model in bf16 (tensor cores) + TF32 so the
# Linear(W->hidden) first layer isn't stuck on the ~19 TFLOPS pure-fp32 path.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# bf16 autocast for transformer/convnext. GRU stays fp32: cuDNN's RNN has no bf16 kernel, so
# autocast silently drops it to a native per-timestep loop that is ~orders of magnitude slower.
# TF32 still accelerates the wide Linear(W->hidden) input matmul in the fp32 GRU path.
AMP_ENABLED = True
FEAT_DTYPE = torch.bfloat16

CACHE = ROOT / os.environ.get("DYN_CACHE", "cache/dyn150_sparse_10k_1b")  # dir of memmap .npy
OUTROOT = os.environ.get("DYN_OUT", "results/dyn150_2k")
ARCH_SEL = os.environ.get("DYN_ARCH")  # gru|transformer|convnext ; None => all (sequential)
BS_OVERRIDE = os.environ.get("DYN_BS")  # optional int
DEVICE = "cuda"
EPOCHS = 15

ARCHS = {
    "gru": {
        "arch": "gru",
        "out": f"{OUTROOT}/gru/gru_lr3e4",
        "model": {"hidden_width": 256, "depth": 3, "dropout": 0.1},
        "lr": 3e-4,
        "bs": 8,
    },
    "transformer": {
        "arch": "transformer",
        "out": f"{OUTROOT}/transformer/transformer_wide",
        "model": {
            "hidden_width": 256,
            "heads": 8,
            "depth": 2,
            "feedforward_width": 512,
            "dropout": 0.1,
        },
        "lr": 3e-4,
        "bs": 4,
    },
    "convnext": {
        "arch": "convnext",
        "out": f"{OUTROOT}/tcn/convnext",
        "model": {
            "channels": 256,
            "depth": 6,
            "kernel": 7,
            "expansion": 4,
            "dropout": 0.1,
            "layer_scale": 0.01,
        },
        "lr": 1e-3,
        "bs": 8,
    },
}


def load_cache():
    # col/val are memory-mapped read-only .npy — the OS page cache is SHARED across the parallel
    # trainer processes, so N trainers cost ~1 physical copy of the (~55 GiB) cache, not N.
    """Load the sparse dynamic cache (CPU-resident col/val + small GPU arrays)."""
    import json

    meta = json.loads((CACHE / "meta.json").read_text())
    W = int(meta["width"])
    col = np.load(CACHE / "col.npy", mmap_mode="r")  # int32, on disk (shared page cache)
    val = np.load(CACHE / "val.npy", mmap_mode="r")  # fp16
    tok_ptr = torch.from_numpy(np.load(CACHE / "tok_ptr.npy").astype(np.int64)).to(DEVICE)  # small
    labels = torch.from_numpy(np.load(CACHE / "labels.npy").astype(np.int64)).to(DEVICE)  # small
    lengths = np.load(CACHE / "lengths.npy")
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    sp = np.load(CACHE / "split_indices.npz")
    splits = {k: sp[k] for k in ("train", "validation", "test")}
    print(
        f"cache(mmap): nnz={col.shape[0]} width={W}  col/val on disk "
        f"({(col.nbytes + val.nbytes) / 2**30:.1f} GiB, shared page cache)  "
        f"gpu_resident={torch.cuda.memory_allocated() / 2**30:.2f} GiB",
        flush=True,
    )
    return {
        "col": col,
        "val": val,
        "tok_ptr": tok_ptr,
        "labels": labels,
        "offsets": offsets,
        "splits": splits,
        "W": W,
    }


def densify(C, rec_ids):
    """(B, maxT, W) fp32 features + (B, maxT) labels on GPU. col/val slices streamed CPU->GPU."""
    off = C["offsets"]
    tok_ptr = C["tok_ptr"]
    col = C["col"]
    val = C["val"]
    labels = C["labels"]
    Ts = [int(off[r + 1] - off[r]) for r in rec_ids]
    maxT = max(Ts)
    B = len(rec_ids)
    x = torch.zeros(
        B, maxT, C["W"], device=DEVICE, dtype=FEAT_DTYPE
    )  # bf16 (tf/cx) halves dense-input bandwidth; fp32 for gru
    y = torch.full((B, maxT), PAD, dtype=torch.long, device=DEVICE)
    for b, r in enumerate(rec_ids):
        lo, hi = int(off[r]), int(off[r + 1])
        T = hi - lo
        ptr = tok_ptr[lo : hi + 1]
        nlo, nhi = int(ptr[0].item()), int(ptr[-1].item())
        if nhi > nlo:
            counts = ptr[1:] - ptr[:-1]
            rows = torch.repeat_interleave(torch.arange(T, device=DEVICE), counts)
            cslice = torch.from_numpy(np.ascontiguousarray(col[nlo:nhi])).to(DEVICE).long()
            vslice = torch.from_numpy(np.ascontiguousarray(val[nlo:nhi])).to(DEVICE).to(FEAT_DTYPE)
            x[b, rows, cslice] = vslice
        y[b, :T] = labels[lo:hi]
    return x, y


@torch.inference_mode()
def predict_rows(model, C, rec_ids, bs):
    """Densify and run a detector over given records, returning per-token predictions."""
    model.eval()
    rows = []
    for s in range(0, len(rec_ids), bs):
        batch = rec_ids[s : s + bs]
        x, y = densify(C, batch)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP_ENABLED):
            pred = model(x).argmax(-1)
        for b, r in enumerate(batch):
            T = int(C["offsets"][r + 1] - C["offsets"][r])
            rows.append(
                {"labels": y[b, :T].cpu().numpy(), "predictions": pred[b, :T].cpu().numpy()}
            )
    return rows


def train_arch(name, cfg, C):
    """Train one architecture on the dynamic cache with best-validation-macro-F1 checkpointing."""
    seed_everything(42, deterministic=True)
    global AMP_ENABLED, FEAT_DTYPE
    AMP_ENABLED = cfg["arch"] != "gru"  # gru -> fp32 (cuDNN RNN, no bf16 kernel)
    FEAT_DTYPE = torch.bfloat16 if AMP_ENABLED else torch.float32
    mcfg = {"input_features": C["W"], "classes": len(TOPICS), **cfg["model"]}
    model = build_model(cfg["arch"], mcfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    train_ids = list(C["splits"]["train"])
    val_ids = list(C["splits"]["validation"])
    test_ids = list(C["splits"]["test"])
    bs = int(BS_OVERRIDE) if BS_OVERRIDE else cfg["bs"]
    best_f1, best_epoch, best_state = -1.0, 0, None
    params = sum(p.numel() for p in model.parameters())
    print(
        f"\n=== {name} ({cfg['arch']}) params={params / 1e6:.2f}M width={C['W']} bs={bs} ===",
        flush=True,
    )
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        order = np.random.permutation(len(train_ids))
        for st in range(0, len(order), bs):
            ids = [train_ids[i] for i in order[st : st + bs]]
            x, y = densify(C, ids)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP_ENABLED):
                logits = model(x)
                loss = loss_fn(logits.reshape(-1, len(TOPICS)), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        rows = predict_rows(model, C, val_ids, max(bs, 8))
        truth, pred = flatten_labeled(rows)
        f1 = f1_score(truth, pred, average="macro", zero_division=0)
        print(
            f"{name} epoch {epoch:2d}/{EPOCHS}: val macro-F1={f1:.4f}  ({time.time() - t0:.1f}s)",
            flush=True,
        )
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    out = ROOT / cfg["out"]
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "architecture": cfg["arch"],
            "model": mcfg,
            "classes": TOPICS,
            "epoch": best_epoch,
            "selection": "best_validation_macro_f1",
        },
        out / "checkpoint_best.pt",
    )
    test_rows = predict_rows(model, C, test_ids, max(bs, 8))
    m = metrics(test_rows, [5, 10])
    (out / "results.json").write_text(
        json.dumps({"parameters": params, "best_epoch": best_epoch, "test": m}, indent=2) + "\n"
    )
    print(
        f"{name} DONE best_epoch={best_epoch} test macroF1={m['macro_f1']:.4f} "
        f"IoU={m['topic_overlap']['macro_iou']:.4f}",
        flush=True,
    )


def main():
    """Train the selected dynamic detector(s) and write checkpoints + results.json."""
    C = load_cache()
    todo = {ARCH_SEL: ARCHS[ARCH_SEL]} if ARCH_SEL else ARCHS
    for name, cfg in todo.items():
        train_arch(name, cfg, C)
        torch.cuda.empty_cache()
    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
