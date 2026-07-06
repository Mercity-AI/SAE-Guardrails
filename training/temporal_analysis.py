"""
Temporal analysis of SAE feature accumulation during generation.

For each test sample, runs generation token-by-token on the best layer only,
building a cumulative any() feature vector after each step and querying the
classifier at every step. Produces:
  - Confidence curve: accuracy and mean confidence vs generation step
  - Per-topic stopping points: median step to cross 90 / 95 / 99% confidence
  - Early stopping simulation: accuracy and mean tokens used at each threshold

Requires a checkpoint from training/main.py. Reads the checkpoint's metadata
to determine the model, SAE, best layer, and feature IDs automatically.

Usage:
    python training/temporal_analysis.py --checkpoint checkpoints/gemma-3-4b-it_20260619_122931
"""

import argparse
import json
import sys
import types
import importlib.machinery

# torchaudio ABI stub (same workaround as main.py)
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

# ---------------------------------------------------------------------------
# Config — defaults; overridden by argparse below
# ---------------------------------------------------------------------------

CHECKPOINT   = "checkpoints_gemma4b_decode/gemma-3-4b-it_20260619_122931"
MODEL_NAME   = "google/gemma-3-4b-it"
SAE_REPO     = "google/gemma-scope-2-4b-it"
SAE_WIDTH    = "16k"
SAE_L0       = "small"
BEST_LAYER   = 24
BATCH_SIZE   = 32
MAX_STEPS    = 512
MIN_STEPS    = 10
MAX_INPUT_TOKENS = 2048

THRESHOLDS   = [0.70, 0.80, 0.90, 0.95, 0.99]
CONF_TARGETS = [0.90, 0.95, 0.99]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if DEVICE == "cuda" else torch.float32

parser = argparse.ArgumentParser(
    description="Temporal analysis of SAE feature accumulation during generation.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--checkpoint", default=CHECKPOINT, help="Run prefix for the checkpoint (without _meta.json suffix).")
parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Generation batch size.")
parser.add_argument("--max-steps", type=int, default=MAX_STEPS, help="Max generation steps to analyse.")
parser.add_argument("--min-steps", type=int, default=MIN_STEPS, help="Min steps before early stopping is allowed.")
_args = parser.parse_args()
CHECKPOINT = _args.checkpoint
BATCH_SIZE = _args.batch_size
MAX_STEPS = _args.max_steps
MIN_STEPS = _args.min_steps

# ---------------------------------------------------------------------------
# Load metadata and retrain LogReg
# ---------------------------------------------------------------------------

meta        = json.load(open(f"{CHECKPOINT}_meta.json"))
feature_ids = np.array(meta["feature_ids"])   # (100,) — indices into 16384-dim SAE
topics      = meta["topics"]
n_classes   = len(topics)

print("Retraining LogReg C=10 on 4B train features ...")
tr = pd.read_parquet(f"{CHECKPOINT}_train_features.parquet")
te_feat = pd.read_parquet(f"{CHECKPOINT}_test_features.parquet")
feat_cols = [c for c in tr.columns if c != "label"]
X_tr = tr[feat_cols].values.astype(np.float32)
y_tr = tr["label"].values
X_te_check = te_feat[feat_cols].values.astype(np.float32)
y_te_check  = te_feat["label"].values

clf = LogisticRegression(C=10.0, max_iter=1000, random_state=42)
clf.fit(X_tr, y_tr)
print(f"  sanity check accuracy: {(clf.predict(X_te_check) == y_te_check).mean():.1%}\n")

# ---------------------------------------------------------------------------
# Load SAE (best layer only)
# ---------------------------------------------------------------------------

print(f"Loading SAE layer {BEST_LAYER} ...")
filename = f"resid_post_all/layer_{BEST_LAYER}_width_{SAE_WIDTH}_l0_{SAE_L0}/params.safetensors"
path = hf_hub_download(repo_id=SAE_REPO, filename=filename)
raw  = safetensors_load(path, device="cpu")
sae  = JumpReLUSAE(raw["w_enc"].shape[0], raw["w_enc"].shape[1])
with torch.no_grad():
    sae.W_enc.copy_(raw["w_enc"].float())
    sae.b_enc.copy_(raw["b_enc"].float())
    sae.b_dec.copy_(raw["b_dec"].float())
    sae.threshold.copy_(raw["threshold"].float())
sae = sae.to(DEVICE).eval()
print(f"  SAE loaded (d_model={sae.d_model}, d_sae={sae.d_sae})\n")

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
print("  model loaded\n")

# Layer reference
layers_module = model.model.language_model.layers

# ---------------------------------------------------------------------------
# Load test prompts
# ---------------------------------------------------------------------------

te_df   = pd.read_parquet(f"{CHECKPOINT}_test.parquet")
prompts = te_df["prompt"].tolist()
labels  = te_df["label"].values   # (203,)
n_test  = len(prompts)
print(f"Test samples: {n_test}\n")

# ---------------------------------------------------------------------------
# Per-step generation and capture
# ---------------------------------------------------------------------------
# step_fired[i, t] = 100-dim binary vector of which selected features fired at step t
# We build this sample-by-sample to keep memory clean, then stack.

print("="*72)
print("Running per-step generation (decode only, L24) ...")
print("="*72)

# Will hold per-step binary vectors: shape (n_test, MAX_STEPS, 100)
all_step_fired = np.zeros((n_test, MAX_STEPS, len(feature_ids)), dtype=np.bool_)
# How many steps each sample actually ran
n_steps_per_sample = np.zeros(n_test, dtype=np.int32)

n_batches = (n_test + BATCH_SIZE - 1) // BATCH_SIZE

captured = {}

def make_hook():
    def _hook(_module, _input, output):
        hs = output[0] if isinstance(output, tuple) else output
        captured["L24"] = hs.float()
    return _hook

hook = layers_module[BEST_LAYER].register_forward_hook(make_hook())

try:
    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        end   = min(start + BATCH_SIZE, n_test)
        bs    = end - start
        batch_prompts = prompts[start:end]
        print(f"  batch {batch_idx+1}/{n_batches} (samples {start}–{end-1})", end="\r")

        # Apply chat template
        formatted = [
            tok.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in batch_prompts
        ]

        enc = tok(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
            add_special_tokens=False,
        ).to(DEVICE)

        # Per-sample cumulative fired: (bs, 100) binary
        cumulative = np.zeros((bs, len(feature_ids)), dtype=np.bool_)
        # Track which samples are still active
        active = torch.ones(bs, dtype=torch.bool, device=DEVICE)

        # Prefill
        captured.clear()
        with torch.inference_mode():
            out = model(**enc, use_cache=True)
        past_kv    = out.past_key_values
        next_tokens = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        for step in range(MAX_STEPS):
            active = active & (next_tokens.squeeze(-1) != tok.eos_token_id)
            if not active.any():
                break

            captured.clear()
            with torch.inference_mode():
                out = model(next_tokens, past_key_values=past_kv, use_cache=True)
            past_kv     = out.past_key_values
            next_tokens = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            # Hidden state: (bs, 1, d_model) → (bs, d_model)
            h = captured["L24"]
            if h.dim() == 3:
                h = h[:, -1, :]

            # SAE fired mask: (bs, d_sae)
            with torch.inference_mode():
                fired = sae.fired_mask(h).cpu().numpy()  # (bs, 16384)

            # Extract top-100 selected features
            step_vec = fired[:, feature_ids]  # (bs, 100)

            for i in range(bs):
                si = start + i
                if active[i].item():
                    all_step_fired[si, step] = step_vec[i]
                    n_steps_per_sample[si]   = step + 1

        del past_kv, out, enc
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

finally:
    hook.remove()

print(f"\n  done. mean steps per sample: {n_steps_per_sample.mean():.1f}\n")

# ---------------------------------------------------------------------------
# Build per-step cumulative confidence
# ---------------------------------------------------------------------------
# confidence[i, t] = max softmax probability after seeing steps 0..t
# predicted[i, t]  = argmax class after seeing steps 0..t

print("Computing cumulative confidence at each step ...")

# We only evaluate at a subset of steps for speed (every 5 steps up to 200, then every 20)
eval_steps = list(range(1, 51)) + list(range(55, 201, 5)) + list(range(220, MAX_STEPS+1, 20))
eval_steps = sorted(set(s for s in eval_steps if s <= MAX_STEPS))

confidence = np.zeros((n_test, len(eval_steps)), dtype=np.float32)
predicted  = np.zeros((n_test, len(eval_steps)), dtype=np.int32)

for ei, t in enumerate(eval_steps):
    # Cumulative any() over steps 0..t-1
    cum = all_step_fired[:, :t, :].any(axis=1).astype(np.float32)  # (n_test, 100)
    proba = clf.predict_proba(cum)   # (n_test, n_classes)
    confidence[:, ei] = proba.max(axis=1)
    predicted[:, ei]  = proba.argmax(axis=1)

print("  done.\n")

# ---------------------------------------------------------------------------
# Section 1: Confidence and accuracy curve
# ---------------------------------------------------------------------------

print("="*72)
print("SECTION 1 — Accuracy and mean confidence vs step")
print("="*72)
print(f"  {'Step':>6}  {'Accuracy':>9}  {'Mean conf':>10}  {'Med conf':>9}")
for ei, t in enumerate(eval_steps):
    if t > 200 and t % 40 != 0:
        continue
    acc  = (predicted[:, ei] == labels).mean()
    mc   = confidence[:, ei].mean()
    medc = np.median(confidence[:, ei])
    print(f"  {t:>6}  {acc:>9.1%}  {mc:>10.3f}  {medc:>9.3f}")

# ---------------------------------------------------------------------------
# Section 2: Per-topic stopping points
# ---------------------------------------------------------------------------

print("\n" + "="*72)
print("SECTION 2 — Per-topic: median step to first cross confidence threshold")
print("="*72)

# For each sample, find the first eval_step where confidence >= threshold
# (subject to step >= MIN_STEPS)
eval_steps_arr = np.array(eval_steps)

header = f"  {'Topic':<30}"
for ct in CONF_TARGETS:
    header += f"  {int(ct*100)}%-conf"
print(header)

per_topic_stop = {}
for label_idx, topic in enumerate(topics):
    mask = labels == label_idx
    conf_topic = confidence[mask]          # (n_topic, n_eval_steps)
    row = f"  {topic:<30}"
    stops = {}
    for ct in CONF_TARGETS:
        # For each sample, first eval step >= MIN_STEPS where conf >= ct
        valid = eval_steps_arr >= MIN_STEPS
        above = (conf_topic[:, valid] >= ct)           # (n_topic, n_valid)
        valid_steps = eval_steps_arr[valid]
        first_cross = []
        for s_above in above:
            idx = np.argmax(s_above)  # first True; argmax returns 0 if all False
            if s_above[idx]:
                first_cross.append(valid_steps[idx])
            else:
                first_cross.append(MAX_STEPS)  # never crossed
        median_stop = int(np.median(first_cross))
        stops[ct] = median_stop
        row += f"  {median_stop:>8}"
    per_topic_stop[topic] = stops
    print(row)

# Overall row
overall_row = f"  {'OVERALL':<30}"
for ct in CONF_TARGETS:
    all_stops = []
    for label_idx in range(n_classes):
        mask = labels == label_idx
        conf_topic = confidence[mask]
        valid = eval_steps_arr >= MIN_STEPS
        above = conf_topic[:, valid] >= ct
        valid_steps = eval_steps_arr[valid]
        for s_above in above:
            idx = np.argmax(s_above)
            all_stops.append(valid_steps[idx] if s_above[idx] else MAX_STEPS)
    overall_row += f"  {int(np.median(all_stops)):>8}"
print(overall_row)

# ---------------------------------------------------------------------------
# Section 3: Early stopping simulation
# ---------------------------------------------------------------------------

print("\n" + "="*72)
print("SECTION 3 — Early stopping simulation")
print(f"  (min {MIN_STEPS} steps before stopping allowed)")
print("="*72)
print(f"  {'Threshold':>10}  {'Accuracy':>9}  {'Mean tokens':>12}  {'Median tokens':>14}  {'% stopped early':>16}")

valid_mask = eval_steps_arr >= MIN_STEPS
valid_steps = eval_steps_arr[valid_mask]
valid_conf  = confidence[:, valid_mask]
valid_pred  = predicted[:, valid_mask]

early_stop_results = {}
for thresh in THRESHOLDS:
    tokens_used = np.full(n_test, MAX_STEPS, dtype=np.int32)
    preds_used  = predicted[:, -1].copy()   # default: full-generation prediction

    for i in range(n_test):
        above = np.where(valid_conf[i] >= thresh)[0]
        if len(above) > 0:
            ei = above[0]
            tokens_used[i] = valid_steps[ei]
            preds_used[i]  = valid_pred[i, ei]

    acc = (preds_used == labels).mean()
    pct_early = (tokens_used < MAX_STEPS).mean()
    print(f"  {thresh:>10.0%}  {acc:>9.1%}  {tokens_used.mean():>12.1f}  {np.median(tokens_used):>14.0f}  {pct_early:>15.1%}")
    early_stop_results[thresh] = {
        "accuracy": acc,
        "mean_tokens": float(tokens_used.mean()),
        "median_tokens": float(np.median(tokens_used)),
        "pct_early": float(pct_early),
        "tokens_used": tokens_used.tolist(),
        "preds": preds_used.tolist(),
    }

# Per-topic breakdown at 95% threshold
DEMO_THRESH = 0.95
print(f"\n  Per-topic breakdown at threshold={DEMO_THRESH:.0%}:")
print(f"  {'Topic':<30}  {'Accuracy':>9}  {'Mean tokens':>12}  {'Median tokens':>14}")

valid_mask2 = eval_steps_arr >= MIN_STEPS
valid_steps2 = eval_steps_arr[valid_mask2]
valid_conf2  = confidence[:, valid_mask2]
valid_pred2  = predicted[:, valid_mask2]

for label_idx, topic in enumerate(topics):
    mask = labels == label_idx
    tc   = valid_conf2[mask]
    tp   = valid_pred2[mask]
    tl   = labels[mask]
    n    = mask.sum()
    tok_used = np.full(n, MAX_STEPS, dtype=np.int32)
    pred_used = predicted[mask, -1].copy()
    for i in range(n):
        above = np.where(tc[i] >= DEMO_THRESH)[0]
        if len(above) > 0:
            ei = above[0]
            tok_used[i]  = valid_steps2[ei]
            pred_used[i] = tp[i, ei]
    acc = (pred_used == tl).mean()
    print(f"  {topic:<30}  {acc:>9.1%}  {tok_used.mean():>12.1f}  {np.median(tok_used):>14.0f}")

# ---------------------------------------------------------------------------
# Section 4: Example traces
# ---------------------------------------------------------------------------

print("\n" + "="*72)
print("SECTION 4 — Example confidence traces (one per topic)")
print("="*72)
show_steps = [1, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 512]
show_ei    = [eval_steps.index(s) for s in show_steps if s in eval_steps]
show_s     = [eval_steps[ei] for ei in show_ei]

for label_idx, topic in enumerate(topics):
    idxs = np.where(labels == label_idx)[0]
    i    = idxs[0]
    true_label = labels[i]
    print(f"\n  {topic} (sample {i})")
    print(f"  {'Step':>6}  {'Conf':>7}  {'Pred':>25}  {'Correct':>8}")
    for ei, s in zip(show_ei, show_s):
        conf = confidence[i, ei]
        pred = topics[predicted[i, ei]]
        correct = "✓" if predicted[i, ei] == true_label else "✗"
        print(f"  {s:>6}  {conf:>7.3f}  {pred:>25}  {correct:>8}")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

results = {
    "eval_steps": eval_steps,
    "accuracy_curve": (predicted == labels[:, None]).mean(axis=0).tolist(),
    "mean_confidence_curve": confidence.mean(axis=0).tolist(),
    "per_topic_stop": {t: {str(k): v for k, v in s.items()} for t, s in per_topic_stop.items()},
    "early_stop": {str(k): v for k, v in early_stop_results.items()},
}
import json as _json
with open("temporal_results.json", "w") as f:
    _json.dump(results, f, indent=2)
print("\n\nResults saved to temporal_results.json")
