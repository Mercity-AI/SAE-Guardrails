"""
MLP architecture ablation over pre-computed SAE feature vectors.

Sweeps depth (number of hidden layers) and width across all capture-mode
datasets. No GPU generation needed — reads the _train_features.parquet /
_test_features.parquet files produced by training/main.py directly.

Architecture grid includes linear probes, 1-layer, 2-layer, and 3-layer MLPs
from 16 to 512 units wide. Each configuration is evaluated over N_SEEDS random
seeds and the mean/std accuracy reported.

Usage:
    python training/mlp_ablations.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Datasets: (label, train_features_path, test_features_path)
# ---------------------------------------------------------------------------

CHECKPOINTS = [
    ("decode (run1 94.1%)",
     "checkpoints_shuffled_512/gemma-3-1b-it_20260618_112951_train_features.parquet",
     "checkpoints_shuffled_512/gemma-3-1b-it_20260618_112951_test_features.parquet"),
    ("decode (run2 95.6%)",
     "checkpoints_shuffled_512/gemma-3-1b-it_20260618_115855_train_features.parquet",
     "checkpoints_shuffled_512/gemma-3-1b-it_20260618_115855_test_features.parquet"),
    ("prefill+decode (94.6%)",
     "checkpoints_shuffled_512_prefill_decode/gemma-3-1b-it_20260618_122805_train_features.parquet",
     "checkpoints_shuffled_512_prefill_decode/gemma-3-1b-it_20260618_122805_test_features.parquet"),
    ("prefill (75.4%)",
     "checkpoints_shuffled_512_prefill/gemma-3-1b-it_20260618_123516_train_features.parquet",
     "checkpoints_shuffled_512_prefill/gemma-3-1b-it_20260618_123516_test_features.parquet"),
]

# ---------------------------------------------------------------------------
# Architecture grid
# ---------------------------------------------------------------------------
# Each entry is a list of hidden layer widths; [] = linear probe.
HIDDEN_CONFIGS = [
    [],               # linear probe
    [16],
    [32],
    [64],             # baseline from report
    [128],
    [256],
    [512],
    [64, 32],
    [128, 64],
    [256, 128],
    [512, 256],
    [64, 64, 32],
    [128, 64, 32],
    [256, 128, 64],
]

N_SEEDS = 5
EPOCHS = 50          # slightly more than the report's 30 to give deeper nets a chance
LR = 1e-3
BATCH_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FlexMLP(nn.Module):
    def __init__(self, n_in, n_classes, hidden_sizes):
        super().__init__()
        layers = []
        prev = n_in
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_eval(X_tr, y_tr, X_te, y_te, hidden_sizes, n_classes, seed):
    torch.manual_seed(seed)
    n_in = X_tr.shape[1]
    model = FlexMLP(n_in, n_classes, hidden_sizes).to(DEVICE)
    use_fused = DEVICE == "cuda"
    opt = torch.optim.AdamW(model.parameters(), lr=LR, fused=use_fused)
    loss_fn = nn.CrossEntropyLoss()

    X_tr = X_tr.to(DEVICE)
    y_tr = y_tr.to(DEVICE)
    X_te = X_te.to(DEVICE)
    y_te = y_te.to(DEVICE)
    n = X_tr.shape[0]

    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, BATCH_SIZE):
            idx = perm[s: s + BATCH_SIZE]
            loss = loss_fn(model(X_tr[idx]), y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_te).argmax(dim=-1)
    acc = (preds == y_te).float().mean().item()
    return acc


def load_dataset(train_path, test_path):
    tr = pd.read_parquet(train_path)
    te = pd.read_parquet(test_path)
    feat_cols = [c for c in tr.columns if c != "label"]
    X_tr = torch.tensor(tr[feat_cols].values, dtype=torch.float32)
    y_tr = torch.tensor(tr["label"].values, dtype=torch.long)
    X_te = torch.tensor(te[feat_cols].values, dtype=torch.float32)
    y_te = torch.tensor(te["label"].values, dtype=torch.long)
    n_classes = int(y_tr.max().item()) + 1
    return X_tr, y_tr, X_te, y_te, n_classes


def arch_label(hidden_sizes):
    if not hidden_sizes:
        return "linear"
    return "→".join(str(h) for h in hidden_sizes)


def n_params(n_in, n_classes, hidden_sizes):
    sizes = [n_in] + list(hidden_sizes) + [n_classes]
    return sum(sizes[i] * sizes[i+1] + sizes[i+1] for i in range(len(sizes) - 1))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

all_results = []

for ds_label, train_path, test_path in CHECKPOINTS:
    print(f"\n{'='*72}")
    print(f"Dataset: {ds_label}")
    print(f"{'='*72}")
    print(f"{'Architecture':<22} {'Params':>7}  {'Mean acc':>9}  {'Std':>7}  Seeds")

    X_tr, y_tr, X_te, y_te, n_classes = load_dataset(train_path, test_path)
    n_in = X_tr.shape[1]

    for hidden in HIDDEN_CONFIGS:
        accs = [
            train_eval(X_tr, y_tr, X_te, y_te, hidden, n_classes, seed=s)
            for s in range(N_SEEDS)
        ]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        params = n_params(n_in, n_classes, hidden)
        label = arch_label(hidden)
        print(f"  {label:<20} {params:>7}  {mean_acc:>8.1%}  ±{std_acc:.3f}  {[f'{a:.3f}' for a in accs]}")
        all_results.append({
            "dataset": ds_label,
            "architecture": label,
            "hidden_sizes": hidden,
            "n_params": params,
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "seed_accs": accs,
        })

# ---------------------------------------------------------------------------
# Summary: best architecture per dataset
# ---------------------------------------------------------------------------
print(f"\n{'='*72}")
print("SUMMARY — best architecture per dataset")
print(f"{'='*72}")
print(f"{'Dataset':<30} {'Architecture':<22} {'Mean acc':>9}")
for ds_label, _, _ in CHECKPOINTS:
    ds_rows = [r for r in all_results if r["dataset"] == ds_label]
    best = max(ds_rows, key=lambda r: r["mean_acc"])
    print(f"  {ds_label:<28} {best['architecture']:<22} {best['mean_acc']:.1%}")

# Save results JSON
out = Path("ablation_results.json")
with open(out, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nFull results saved to {out}")
