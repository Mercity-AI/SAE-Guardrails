"""
Extended classifier and MLP ablations over pre-computed SAE feature vectors.

Runs a comprehensive hyperparameter sweep over the feature parquets produced by
training/main.py. No GPU generation is needed — this script only reads the
pre-computed _train_features.parquet / _test_features.parquet files.

Sections:
  1. Classical sklearn classifiers (LogReg, SVM, kNN, RF, GBM, NB)
  2. Feature count sweep (top-N from the 100, ordered by variance desc)
  3. MLP activation functions (ReLU, GELU, SiLU, ELU, Tanh)
  4. MLP dropout sweep
  5. MLP weight decay sweep
  6. MLP with batch normalisation

Usage:
    python training/ablations.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

CHECKPOINTS = [
    ("decode",
     "checkpoints_shuffled_512/gemma-3-1b-it_20260618_112951_train_features.parquet",
     "checkpoints_shuffled_512/gemma-3-1b-it_20260618_112951_test_features.parquet"),
    ("prefill+decode",
     "checkpoints_shuffled_512_prefill_decode/gemma-3-1b-it_20260618_122805_train_features.parquet",
     "checkpoints_shuffled_512_prefill_decode/gemma-3-1b-it_20260618_122805_test_features.parquet"),
    ("prefill",
     "checkpoints_shuffled_512_prefill/gemma-3-1b-it_20260618_123516_train_features.parquet",
     "checkpoints_shuffled_512_prefill/gemma-3-1b-it_20260618_123516_test_features.parquet"),
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SEEDS = 5
BASE_EPOCHS = 50
LR = 1e-3
BATCH = 512

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(train_path, test_path, n_features=100):
    tr = pd.read_parquet(train_path)
    te = pd.read_parquet(test_path)
    feat_cols = [c for c in tr.columns if c != "label"][:n_features]
    X_tr = tr[feat_cols].values.astype(np.float32)
    y_tr = tr["label"].values.astype(np.int64)
    X_te = te[feat_cols].values.astype(np.float32)
    y_te = te["label"].values.astype(np.int64)
    n_classes = int(y_tr.max()) + 1
    return X_tr, y_tr, X_te, y_te, n_classes


# ---------------------------------------------------------------------------
# Flexible MLP
# ---------------------------------------------------------------------------

class FlexMLP(nn.Module):
    def __init__(self, n_in, n_classes, hidden_sizes,
                 activation="relu", dropout=0.0, batch_norm=False):
        super().__init__()
        act_map = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "elu": nn.ELU,
            "tanh": nn.Tanh,
        }
        Act = act_map[activation]
        layers = []
        prev = n_in
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(Act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_eval_mlp(X_tr, y_tr, X_te, y_te, n_classes, hidden_sizes,
                   activation="relu", dropout=0.0, batch_norm=False,
                   weight_decay=0.0, epochs=BASE_EPOCHS, seed=0):
    torch.manual_seed(seed)
    n_in = X_tr.shape[1]
    model = FlexMLP(n_in, n_classes, hidden_sizes,
                    activation=activation, dropout=dropout,
                    batch_norm=batch_norm).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR,
                             weight_decay=weight_decay,
                             fused=(DEVICE == "cuda"))
    loss_fn = nn.CrossEntropyLoss()

    Xtr = torch.tensor(X_tr).to(DEVICE)
    ytr = torch.tensor(y_tr).to(DEVICE)
    Xte = torch.tensor(X_te).to(DEVICE)
    yte = torch.tensor(y_te).to(DEVICE)
    n = Xtr.shape[0]

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, BATCH):
            idx = perm[s: s + BATCH]
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        preds = model(Xte).argmax(-1)
    return (preds == yte).float().mean().item()


def run_mlp(X_tr, y_tr, X_te, y_te, n_classes, hidden_sizes, **kwargs):
    accs = [train_eval_mlp(X_tr, y_tr, X_te, y_te, n_classes,
                            hidden_sizes, seed=s, **kwargs)
            for s in range(N_SEEDS)]
    return np.mean(accs), np.std(accs)


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def header(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def row(label, mean, std, extra=""):
    print(f"  {label:<35} {mean:>7.1%}  ±{std:.3f}  {extra}")


# ---------------------------------------------------------------------------
# Section 1: Classical sklearn classifiers
# ---------------------------------------------------------------------------

SKLEARN_CLASSIFIERS = [
    ("LogReg  C=0.01",    LogisticRegression(C=0.01,  max_iter=1000, random_state=42)),
    ("LogReg  C=0.1",     LogisticRegression(C=0.1,   max_iter=1000, random_state=42)),
    ("LogReg  C=1",       LogisticRegression(C=1.0,   max_iter=1000, random_state=42)),
    ("LogReg  C=10",      LogisticRegression(C=10.0,  max_iter=1000, random_state=42)),
    ("SVM linear C=0.1",  SVC(kernel="linear", C=0.1,  random_state=42)),
    ("SVM linear C=1",    SVC(kernel="linear", C=1.0,  random_state=42)),
    ("SVM linear C=10",   SVC(kernel="linear", C=10.0, random_state=42)),
    ("SVM RBF   C=1",     SVC(kernel="rbf",    C=1.0,  random_state=42)),
    ("SVM RBF   C=10",    SVC(kernel="rbf",    C=10.0, random_state=42)),
    ("SVM RBF   C=100",   SVC(kernel="rbf",    C=100., random_state=42)),
    ("kNN  k=1",          KNeighborsClassifier(n_neighbors=1)),
    ("kNN  k=3",          KNeighborsClassifier(n_neighbors=3)),
    ("kNN  k=5",          KNeighborsClassifier(n_neighbors=5)),
    ("kNN  k=11",         KNeighborsClassifier(n_neighbors=11)),
    ("RandomForest 100",  RandomForestClassifier(n_estimators=100,  random_state=42)),
    ("RandomForest 500",  RandomForestClassifier(n_estimators=500,  random_state=42)),
    ("GradientBoosting",  GradientBoostingClassifier(n_estimators=200, random_state=42)),
    ("BernoulliNB",       BernoulliNB()),
]

all_results = []

for ds_label, train_path, test_path in CHECKPOINTS:
    X_tr, y_tr, X_te, y_te, n_classes = load_dataset(train_path, test_path)

    header(f"[1] Classical classifiers — {ds_label}")
    print(f"  {'Classifier':<35} {'Acc':>7}")
    for name, clf in SKLEARN_CLASSIFIERS:
        clf.fit(X_tr, y_tr)
        acc = (clf.predict(X_te) == y_te).mean()
        print(f"  {name:<35} {acc:>7.1%}")
        all_results.append({"section": "sklearn", "dataset": ds_label,
                             "label": name, "mean_acc": acc, "std_acc": 0})


# ---------------------------------------------------------------------------
# Section 2: Feature count sweep  (top-N of the 100, ordered by variance)
# ---------------------------------------------------------------------------

FEATURE_COUNTS = [5, 10, 20, 30, 50, 75, 100]
# Use best MLP from previous ablations (512→256) and LogReg C=1 for comparison
BASE_ARCH = [512, 256]

for ds_label, train_path, test_path in CHECKPOINTS:
    header(f"[2] Feature count sweep — {ds_label}")
    print(f"  {'N features':<20} {'MLP 512→256':>12}  {'LogReg C=1':>12}")
    for nf in FEATURE_COUNTS:
        X_tr, y_tr, X_te, y_te, n_classes = load_dataset(train_path, test_path, n_features=nf)
        mlp_mean, mlp_std = run_mlp(X_tr, y_tr, X_te, y_te, n_classes, BASE_ARCH)
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        lr.fit(X_tr, y_tr)
        lr_acc = (lr.predict(X_te) == y_te).mean()
        print(f"  top-{nf:<15} {mlp_mean:>11.1%}  {lr_acc:>11.1%}")
        all_results.append({"section": "feature_count", "dataset": ds_label,
                             "label": f"top-{nf}", "mlp": mlp_mean, "lr": lr_acc})


# ---------------------------------------------------------------------------
# Section 3: MLP activation functions  (arch: 256→128, decode only)
# ---------------------------------------------------------------------------

ACTIVATIONS = ["relu", "gelu", "silu", "elu", "tanh"]
ABLATION_ARCH = [256, 128]

for ds_label, train_path, test_path in CHECKPOINTS:
    X_tr, y_tr, X_te, y_te, n_classes = load_dataset(train_path, test_path)
    header(f"[3] Activation functions — {ds_label}  (arch: 256→128)")
    for act in ACTIVATIONS:
        mean, std = run_mlp(X_tr, y_tr, X_te, y_te, n_classes, ABLATION_ARCH, activation=act)
        row(act, mean, std)
        all_results.append({"section": "activation", "dataset": ds_label,
                             "label": act, "mean_acc": mean, "std_acc": std})


# ---------------------------------------------------------------------------
# Section 4: Dropout  (arch: 512→256)
# ---------------------------------------------------------------------------

DROPOUTS = [0.0, 0.1, 0.2, 0.3, 0.5]

for ds_label, train_path, test_path in CHECKPOINTS:
    X_tr, y_tr, X_te, y_e, n_classes = load_dataset(train_path, test_path)
    header(f"[4] Dropout — {ds_label}  (arch: 512→256)")
    for dr in DROPOUTS:
        mean, std = run_mlp(X_tr, y_tr, X_te, y_e, n_classes, BASE_ARCH, dropout=dr)
        row(f"dropout={dr}", mean, std)
        all_results.append({"section": "dropout", "dataset": ds_label,
                             "label": f"dropout={dr}", "mean_acc": mean, "std_acc": std})


# ---------------------------------------------------------------------------
# Section 5: Weight decay  (arch: 512→256)
# ---------------------------------------------------------------------------

WEIGHT_DECAYS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 0.5]

for ds_label, train_path, test_path in CHECKPOINTS:
    X_tr, y_tr, X_te, y_te, n_classes = load_dataset(train_path, test_path)
    header(f"[5] Weight decay — {ds_label}  (arch: 512→256)")
    for wd in WEIGHT_DECAYS:
        mean, std = run_mlp(X_tr, y_tr, X_te, y_te, n_classes, BASE_ARCH, weight_decay=wd)
        row(f"wd={wd}", mean, std)
        all_results.append({"section": "weight_decay", "dataset": ds_label,
                             "label": f"wd={wd}", "mean_acc": mean, "std_acc": std})


# ---------------------------------------------------------------------------
# Section 6: Batch normalisation  (arch: 256→128, with/without, ±dropout)
# ---------------------------------------------------------------------------

BN_CONFIGS = [
    ("no BN, no drop",       dict(batch_norm=False, dropout=0.0)),
    ("BN, no drop",          dict(batch_norm=True,  dropout=0.0)),
    ("no BN, drop=0.2",      dict(batch_norm=False, dropout=0.2)),
    ("BN + drop=0.2",        dict(batch_norm=True,  dropout=0.2)),
]

for ds_label, train_path, test_path in CHECKPOINTS:
    X_tr, y_tr, X_te, y_te, n_classes = load_dataset(train_path, test_path)
    header(f"[6] Batch norm — {ds_label}  (arch: 256→128)")
    for name, kwargs in BN_CONFIGS:
        mean, std = run_mlp(X_tr, y_tr, X_te, y_te, n_classes, ABLATION_ARCH, **kwargs)
        row(name, mean, std)
        all_results.append({"section": "batch_norm", "dataset": ds_label,
                             "label": name, "mean_acc": mean, "std_acc": std})


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

out = Path("ablations2_results.json")
with open(out, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\n\nAll results saved to {out}")
