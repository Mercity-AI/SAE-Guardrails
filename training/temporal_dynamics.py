"""
Temporal dynamics analysis with properly calibrated per-step classifiers.

For each snapshot step t (every SNAPSHOT_INTERVAL steps), trains a separate
LogReg on cumulative SAE features 0..t from the training set, then evaluates
on the test set. No distribution shift — the classifier at step t has only
ever seen inputs of the same density as what it's tested on.

Also computes cross-topic variance across all 16,384 SAE features at each
snapshot to show which steps and which features carry the most information.

Requires a checkpoint from training/main.py and the same prompts.jsonl used
for training so it can reproduce the train/test split.

Usage:
    python training/temporal_dynamics.py
    python training/temporal_dynamics.py --checkpoint-dir my_checkpoints --max-steps 500
"""

import argparse
import glob
import json
import sys
import types
import importlib.machinery
from pathlib import Path

if "torchaudio" not in sys.modules:
    _ta_stub = types.ModuleType("torchaudio")
    _ta_stub.__spec__ = importlib.machinery.ModuleSpec("torchaudio", None)
    sys.modules["torchaudio"] = _ta_stub

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as safetensors_load
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer

from models import JumpReLUSAE
from utils import load_topic_data, split_data, build_labeled_set

# ---------------------------------------------------------------------------
# Config — defaults; overridden by argparse below
# ---------------------------------------------------------------------------

CHECKPOINT_DIR    = "checkpoints_1k_1b"
SNAPSHOT_INTERVAL = 10
MAX_STEPS         = 300
BATCH_SIZE        = 32
MAX_INPUT_TOKENS  = 2048
SEED              = 42
TRAIN_RATIO       = 0.8
CONF_TARGETS      = [0.90, 0.95, 0.99]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if DEVICE == "cuda" else torch.float32

parser = argparse.ArgumentParser(
    description="Temporal dynamics analysis with per-step calibrated classifiers.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR, help="Directory containing checkpoint files.")
parser.add_argument("--snapshot-interval", type=int, default=SNAPSHOT_INTERVAL, help="Analyse every N generation steps.")
parser.add_argument("--max-steps", type=int, default=MAX_STEPS, help="Max generation steps to analyse.")
parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Generation batch size.")
parser.add_argument("--data-path", type=str, default="prompts.jsonl", help="Path to prompts.jsonl.")
_args = parser.parse_args()
CHECKPOINT_DIR = _args.checkpoint_dir
SNAPSHOT_INTERVAL = _args.snapshot_interval
MAX_STEPS = _args.max_steps
BATCH_SIZE = _args.batch_size
_DATA_PATH = _args.data_path

# ---------------------------------------------------------------------------
# Locate checkpoint
# ---------------------------------------------------------------------------

runs = sorted(glob.glob(f"{CHECKPOINT_DIR}/*_meta.json"))
if not runs:
    raise FileNotFoundError(f"No checkpoint found in {CHECKPOINT_DIR}/")
meta_path = runs[-1]   # most recent
run_prefix = meta_path.replace("_meta.json", "")
meta       = json.load(open(meta_path))

feature_ids = np.array(meta["feature_ids"])
topics      = meta["topics"]
best_layer  = meta["best_layer"]
n_classes   = len(topics)

print(f"Checkpoint: {run_prefix}")
print(f"Best layer: L{best_layer} | Features: {len(feature_ids)} | Topics: {n_classes}\n")

# ---------------------------------------------------------------------------
# Load SAE (best layer only)
# ---------------------------------------------------------------------------

print(f"Loading SAE L{best_layer} ...")
filename = f"resid_post_all/layer_{best_layer}_width_{SAE_WIDTH}_l0_{SAE_L0}/params.safetensors"
path     = hf_hub_download(repo_id=SAE_REPO, filename=filename)
raw      = safetensors_load(path, device="cpu")
sae      = JumpReLUSAE(raw["w_enc"].shape[0], raw["w_enc"].shape[1])
with torch.no_grad():
    sae.W_enc.copy_(raw["w_enc"].float())
    sae.b_enc.copy_(raw["b_enc"].float())
    sae.b_dec.copy_(raw["b_dec"].float())
    sae.threshold.copy_(raw["threshold"].float())
sae = sae.to(DEVICE).eval()
d_sae = sae.d_sae
print(f"  SAE: d_model={sae.d_model}, d_sae={d_sae}\n")

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

print("Loading model ...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=DTYPE, trust_remote_code=True,
    attn_implementation="flash_attention_2"
).to(DEVICE).eval()
model.config.use_cache = True
layers_module = model.model.layers
print("  model loaded\n")

# ---------------------------------------------------------------------------
# Load data (same split as training run)
# ---------------------------------------------------------------------------

data = load_topic_data({}, 5000, SEED, jsonl_path=Path(_DATA_PATH))
train_data, test_data = split_data(data, TRAIN_RATIO, seed=SEED)
train_texts, train_labels = build_labeled_set(train_data, topics)
test_texts,  test_labels  = build_labeled_set(test_data,  topics)
train_labels = np.array(train_labels)
test_labels  = np.array(test_labels)

print(f"Data: {len(train_texts)} train / {len(test_texts)} test\n")

# ---------------------------------------------------------------------------
# Generation with per-step capture at best_layer
# ---------------------------------------------------------------------------

snapshot_steps = list(range(SNAPSHOT_INTERVAL, MAX_STEPS + 1, SNAPSHOT_INTERVAL))
n_snapshots    = len(snapshot_steps)
n_train        = len(train_texts)
n_test         = len(test_texts)

# Accumulators for cross-topic variance
# topic_sum_fired[s, topic, f] = sum of cumulative fired indicator over training samples
topic_sum_fired = np.zeros((n_snapshots, n_classes, d_sae), dtype=np.float32)
topic_counts    = np.bincount(train_labels, minlength=n_classes).astype(np.float32)

# Per-sample cumulative feature vectors at each snapshot (top-100 only)
train_cum = np.zeros((n_train, n_snapshots, len(feature_ids)), dtype=np.bool_)
test_cum  = np.zeros((n_test,  n_snapshots, len(feature_ids)), dtype=np.bool_)

captured = {}

def make_hook():
    def _hook(_module, _input, output):
        hs = output[0] if isinstance(output, tuple) else output
        captured["h"] = hs.float()
    return _hook

hook = layers_module[best_layer].register_forward_hook(make_hook())


@torch.inference_mode()
def run_capture(texts, labels_arr, cum_out, topic_sum=None):
    """Generate and capture per-step SAE firings. Updates cum_out in-place."""
    n          = len(texts)
    n_batches  = (n + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        end   = min(start + BATCH_SIZE, n)
        bs    = end - start
        print(f"    batch {batch_idx+1}/{n_batches}", end="\r")

        formatted = [
            tok.apply_chat_template(
                [{"role": "user", "content": t}],
                tokenize=False, add_generation_prompt=True,
            )
            for t in texts[start:end]
        ]
        enc = tok(
            formatted, return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_INPUT_TOKENS,
            add_special_tokens=False,
        ).to(DEVICE)

        # running cumulative fired per sample in this batch: (bs, d_sae)
        batch_cum = np.zeros((bs, d_sae), dtype=np.bool_)
        active    = torch.ones(bs, dtype=torch.bool, device=DEVICE)

        # prefill
        captured.clear()
        out        = model(**enc, use_cache=True)
        past_kv    = out.past_key_values
        next_tok   = out.logits[:, -1, :].argmax(-1, keepdim=True)

        for step in range(1, MAX_STEPS + 1):
            active = active & (next_tok.squeeze(-1) != tok.eos_token_id)
            if not active.any():
                break

            captured.clear()
            out     = model(next_tok, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)

            h = captured["h"]
            if h.dim() == 3:
                h = h[:, -1, :]

            fired = sae.fired_mask(h).cpu().numpy()  # (bs, d_sae)

            for i in range(bs):
                if active[i].item():
                    batch_cum[i] |= fired[i]

            # snapshot?
            if step % SNAPSHOT_INTERVAL == 0:
                si = step // SNAPSHOT_INTERVAL - 1
                for i in range(bs):
                    gi = start + i   # global sample index
                    cum_out[gi, si] = batch_cum[i][feature_ids]
                    if topic_sum is not None:
                        # Include all samples (inactive ones have frozen cumulative features)
                        topic_sum[si, int(labels_arr[gi])] += batch_cum[i]

        del past_kv, out, enc
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print()


try:
    print("="*72)
    print("Generating from train set ...")
    print("="*72)
    run_capture(train_texts, train_labels, train_cum, topic_sum=topic_sum_fired)

    print("="*72)
    print("Generating from test set ...")
    print("="*72)
    run_capture(test_texts, test_labels, test_cum, topic_sum=None)
finally:
    hook.remove()

# ---------------------------------------------------------------------------
# Cross-topic variance at each snapshot (all d_sae features)
# ---------------------------------------------------------------------------

# topic_mean_fired[s, t, f] = mean firing rate for snapshot s, topic t, feature f
topic_mean_fired = topic_sum_fired / topic_counts[None, :, None]   # broadcast
# variance across 7 topics
variance_per_snapshot = topic_mean_fired.var(axis=1)               # (n_snapshots, d_sae)

# ---------------------------------------------------------------------------
# Train per-step LogReg and evaluate
# ---------------------------------------------------------------------------

print("Training per-snapshot LogReg classifiers ...")

accuracy_per_snapshot   = np.zeros(n_snapshots)
mean_conf_per_snapshot  = np.zeros(n_snapshots)
# per-topic: (n_classes, n_snapshots)
topic_accuracy = np.zeros((n_classes, n_snapshots))
topic_conf     = np.zeros((n_classes, n_snapshots))

for si, t in enumerate(snapshot_steps):
    X_tr = train_cum[:, si, :].astype(np.float32)
    X_te = test_cum[:,  si, :].astype(np.float32)

    clf = LogisticRegression(C=10.0, max_iter=1000, random_state=SEED)
    clf.fit(X_tr, train_labels)

    proba = clf.predict_proba(X_te)          # (n_test, n_classes)
    preds = proba.argmax(axis=1)
    confs = proba.max(axis=1)

    accuracy_per_snapshot[si]  = (preds == test_labels).mean()
    mean_conf_per_snapshot[si] = confs.mean()

    for t_idx in range(n_classes):
        mask = test_labels == t_idx
        topic_accuracy[t_idx, si] = (preds[mask] == test_labels[mask]).mean()
        topic_conf[t_idx, si]     = proba[mask].max(axis=1).mean()

    print(f"  step {t:>4}: acc={accuracy_per_snapshot[si]:.1%}  conf={mean_conf_per_snapshot[si]:.3f}", end="\r")

print("\nDone.\n")

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------

print("="*72)
print("SECTION 1 — Overall accuracy and confidence per snapshot step")
print("="*72)
print(f"  {'Step':>5}  {'Accuracy':>9}  {'Mean conf':>10}  {'Top-3 variance features'}")
for si, t in enumerate(snapshot_steps):
    top3_idx = np.argsort(variance_per_snapshot[si])[::-1][:3]
    top3_str = "  ".join(f"feat_{i}({variance_per_snapshot[si,i]:.4f})" for i in top3_idx)
    print(f"  {t:>5}  {accuracy_per_snapshot[si]:>9.1%}  {mean_conf_per_snapshot[si]:>10.3f}  {top3_str}")

print()
print("="*72)
print("SECTION 2 — Per-topic accuracy across steps")
print("="*72)
header = f"  {'Step':>5}"
for topic in topics:
    header += f"  {topic[:8]:>8}"
print(header)
for si, t in enumerate(snapshot_steps):
    row = f"  {t:>5}"
    for t_idx in range(n_classes):
        row += f"  {topic_accuracy[t_idx, si]:>8.1%}"
    print(row)

print()
print("="*72)
print("SECTION 3 — Per-topic mean confidence across steps")
print("="*72)
print(header)
for si, t in enumerate(snapshot_steps):
    row = f"  {t:>5}"
    for t_idx in range(n_classes):
        row += f"  {topic_conf[t_idx, si]:>8.3f}"
    print(row)

print()
print("="*72)
print("SECTION 4 — Per-topic: step at which confidence first crosses threshold")
print("="*72)
hdr = f"  {'Topic':<30}"
for ct in CONF_TARGETS:
    hdr += f"  {int(ct*100)}%"
print(hdr)

for t_idx, topic in enumerate(topics):
    row = f"  {topic:<30}"
    for ct in CONF_TARGETS:
        cross = np.where(topic_conf[t_idx] >= ct)[0]
        step  = snapshot_steps[cross[0]] if len(cross) > 0 else MAX_STEPS
        row  += f"  {step:>4}"
    print(row)

print()
print("="*72)
print("SECTION 5 — Variance distribution at each snapshot (all 16384 features)")
print("="*72)
print(f"  {'Step':>5}  {'Mean var':>10}  {'p50':>8}  {'p90':>8}  {'p99':>8}  {'Max':>8}  {'Top feat var':>12}")
for si, t in enumerate(snapshot_steps):
    v = variance_per_snapshot[si]
    print(f"  {t:>5}  {v.mean():>10.5f}  {np.percentile(v,50):>8.5f}  "
          f"{np.percentile(v,90):>8.5f}  {np.percentile(v,99):>8.5f}  "
          f"{v.max():>8.5f}  {v[np.argsort(v)[::-1][0]]:>12.5f}")

print()
print("="*72)
print("SECTION 6 — Top-10 most variant features at each snapshot")
print("="*72)
for si, t in enumerate(snapshot_steps):
    v    = variance_per_snapshot[si]
    top  = np.argsort(v)[::-1][:10]
    vals = "  ".join(f"{i}({v[i]:.4f})" for i in top)
    print(f"  step {t:>4}: {vals}")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

results = {
    "snapshot_steps": snapshot_steps,
    "accuracy_per_snapshot": accuracy_per_snapshot.tolist(),
    "mean_conf_per_snapshot": mean_conf_per_snapshot.tolist(),
    "topic_accuracy": topic_accuracy.tolist(),
    "topic_conf": topic_conf.tolist(),
    "topics": topics,
    "variance_per_snapshot_stats": [
        {
            "step": snapshot_steps[si],
            "mean": float(variance_per_snapshot[si].mean()),
            "p90": float(np.percentile(variance_per_snapshot[si], 90)),
            "p99": float(np.percentile(variance_per_snapshot[si], 99)),
            "max": float(variance_per_snapshot[si].max()),
            "top10_feat_ids": np.argsort(variance_per_snapshot[si])[::-1][:10].tolist(),
            "top10_variances": sorted(variance_per_snapshot[si].tolist(), reverse=True)[:10],
        }
        for si in range(n_snapshots)
    ],
}

out = f"{run_prefix}_temporal_dynamics.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
np.save(f"{run_prefix}_variance_per_snapshot.npy", variance_per_snapshot)
print(f"\nResults saved to {out}")
print(f"Variance array saved to {run_prefix}_variance_per_snapshot.npy")
