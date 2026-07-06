"""
SAE-based topic classification — training pipeline.

Trains a lightweight MLP to classify which topic an LLM is responding about,
using sparse autoencoder (SAE) feature activations as input rather than raw
text. Supports both Qwen TopK SAEs and Gemma Scope JumpReLU SAEs.

Pipeline:
  1. Load model + per-layer SAEs (Qwen TopK or Gemma Scope JumpReLU)
  2. Load topic data from a JSONL file or HuggingFace datasets, split train/test
  3. Generate from all prompts, capturing hidden states at every layer & step
  4. Run SAEs on stored hidden states → per-sample binary firing vectors per layer
  5. Discover the best layer + top-K features via cross-topic variance
  6. Build binary feature vectors, train a two-layer MLP classifier, evaluate

Usage:
    python training/main.py
    python training/main.py --model-sizes gemma-3-1b-it --capture-mode decode
    python training/main.py --n-pos 1000 --checkpoint-dir my_checkpoints
"""

import argparse
import json
import os
import sys
import types
from datetime import datetime
from pathlib import Path

# torchaudio has a CUDA ABI mismatch with the current PyTorch build; stub it out
# before transformers imports it (via loss_rnnt) so the fallback attention path works.
if "torchaudio" not in sys.modules:
    import importlib.machinery
    _ta_stub = types.ModuleType("torchaudio")
    _ta_stub.__spec__ = importlib.machinery.ModuleSpec("torchaudio", None)
    sys.modules["torchaudio"] = _ta_stub

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as safetensors_load
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoModelForCausalLM, AutoTokenizer

# Local imports here
from models import JumpReLUSAE, TopKSAE, TopicClassifier
from utils import (
    build_dataset_df,
    build_labeled_set,
    filter_dead_samples,
    load_topic_data,
    print_results,
    split_data,
)


# ==========================================================================
# CONFIG
# ==========================================================================

DEFAULT_MODEL_SIZES = ["gemma-3-1b-it"]

CONFIGS = {
    "0.6B": dict(
        model_name="Qwen/Qwen3-0.6B-Base",
        sae_repo="Qwen/SAE-Res-Qwen3-0.6B-Base-W32K-L0_100",
        sae_format="qwen",
        top_k_sae=100,
        layer_stride=1,
    ),
    "1.7B": dict(
        model_name="Qwen/Qwen3-1.7B-Base",
        sae_repo="Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_100",
        sae_format="qwen",
        top_k_sae=100,
        layer_stride=1,
    ),
    "1.7B-Instruct": dict(
        model_name="Qwen/Qwen3-1.7B",
        sae_repo="Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_100",
        sae_format="qwen",
        top_k_sae=100,
        layer_stride=1,
    ),
    "8B": dict(
        model_name="Qwen/Qwen3-8B-Base",
        sae_repo="Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100",
        sae_format="qwen",
        top_k_sae=100,
        layer_stride=2,  # skip every other layer to save memory
    ),
    # Gemma Scope v1 (Gemma 2) — residual-stream JumpReLU SAEs, only 3 IT layers available
    "gemma-2-9b-it": dict(
        model_name="google/gemma-2-9b-it",
        sae_repo="google/gemma-scope-9b-it-res",
        sae_format="gemma_scope_1",
        sae_width="16k",
        sae_l0="47",  # available l0s for width_16k: 14, 26, 47, 88, 186
        sae_layers=[9, 20, 31],  # only these 3 layers have IT SAEs in this repo
        layer_stride=1,
    ),
    # Gemma Scope v2 (Gemma 3) — residual-stream JumpReLU SAEs, all layers covered
    "gemma-3-1b-it": dict(
        model_name="google/gemma-3-1b-it",
        sae_repo="google/gemma-scope-2-1b-it",
        sae_format="gemma_scope_2",
        sae_width="16k",
        sae_l0="small",  # l0≈10 active features per token; "big" gives l0≈40
        layer_stride=1,
    ),
    "gemma-3-4b-it": dict(
        model_name="google/gemma-3-4b-it",
        sae_repo="google/gemma-scope-2-4b-it",
        sae_format="gemma_scope_2",
        sae_width="16k",
        sae_l0="small",
        layer_stride=1,
    ),
}

N_POS = 5000  # max samples per topic
SEED = 42  # global RNG seed
BATCH_SIZE = 64  # generation batch size
CAPTURE_MODE = "decode"  # "decode" | "prefill" | "prefill+decode"
N_TOP_FEATURES = 100  # SAE features selected per layer
TRAIN_RATIO = 0.8  # fraction of data used for training
MLP_HIDDEN = 64  # hidden units in the classifier
MLP_EPOCHS = 30  # training epochs
MLP_LR = 1e-3  # AdamW learning rate
MLP_BATCH_SIZE = 512  # mini-batch size during MLP training
MAX_GEN_TOKENS = 512  # max new tokens per generation
MAX_INPUT_TOKENS = 2048  # prompt truncation length
CHECKPOINT_DIR = "checkpoints_1k_1b"

# Path to prompts.jsonl from topic-data-gen; falls back to HF datasets if missing.
JSONL_DATA_PATH = Path("prompts.jsonl")

# HuggingFace dataset specs used when JSONL_DATA_PATH is absent.
DATASETS = {
    "medical": {
        "hf_id": "starmpcc/Asclepius-Synthetic-Clinical-Notes",
        "config": None,
        "split": "train",
        "columns": ["question"],
        "label": "Asclepius clinical questions",
    },
    "insurance": {
        "hf_id": "bitext/Bitext-insurance-llm-chatbot-training-dataset",
        "config": None,
        "split": "train",
        "columns": ["instruction"],
        "label": "Bitext Insurance questions",
    },
}


# ==========================================================================
# MODEL LOADING
# ==========================================================================


def configure_runtime_speed():
    """Enable TF32 and high-precision matmul for faster CUDA computation."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def text_config(model):
    """Return the sub-config that holds num_hidden_layers / hidden_size.

    Multimodal models (Gemma 3 4B+) nest these under model.config.text_config;
    text-only models expose them directly on model.config.
    """
    cfg = model.config
    return getattr(cfg, "text_config", cfg)


def transformer_layers(model):
    """Return the nn.ModuleList of transformer layers for hook registration.

    Text-only models (Gemma 3 1B, Qwen): model.model.layers
    Multimodal models (Gemma 3 4B+):     model.model.language_model.layers
    """
    inner = model.model
    if hasattr(inner, "language_model"):
        return inner.language_model.layers
    return inner.layers


def load_model(name, device, dtype):
    """Load a causal LM with the best available attention backend, return (model, tokenizer)."""
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # left-padding keeps the real tokens right-aligned so generation starts at the correct position
    tok.padding_side = "left"

    model = None
    last_error = None
    for attn_impl in ("flash_attention_2", "sdpa", None):
        try:
            kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
            if attn_impl is not None:
                kwargs["attn_implementation"] = attn_impl
            model = (
                AutoModelForCausalLM.from_pretrained(name, **kwargs).to(device).eval()
            )
            print(f"  attention: {attn_impl or 'default'}")
            break
        except Exception as exc:
            last_error = exc
            if attn_impl is not None:
                print(f"  attention {attn_impl} unavailable; falling back")

    if model is None:
        raise RuntimeError(f"failed to load model {name}") from last_error

    model.config.use_cache = True
    tcfg = text_config(model)
    print(
        f"  model: {name} | {tcfg.num_hidden_layers} layers, "
        f"d={tcfg.hidden_size}"
    )
    return model, tok


def load_all_saes(repo, layers, expected_d_model, k, device):
    """Download and instantiate one TopKSAE per layer from a Qwen-Scope HuggingFace repo."""
    saes = {}
    d_sae = 0  # set inside loop; defined here so it's always bound after
    for layer in layers:
        print(f"  loading SAE layer {layer}/{max(layers)} ...", end="\r")
        path = hf_hub_download(repo_id=repo, filename=f"layer{layer}.sae.pt")
        # weights_only=True prevents arbitrary code execution from pickle payloads
        params = torch.load(path, map_location="cpu", weights_only=True)
        d_sae, d_model = params["W_enc"].shape
        if d_model != expected_d_model:
            raise ValueError(
                f"SAE layer {layer} in {repo} has d_model={d_model}, "
                f"but model hidden_size={expected_d_model}"
            )

        sae = TopKSAE(d_model, d_sae, k)
        with torch.no_grad():
            # cast to float32 — SAE checkpoints are often bfloat16/float16
            sae.W_enc.copy_(params["W_enc"].float())
            sae.b_enc.copy_(params["b_enc"].float())
        saes[layer] = sae.to(device).eval()

    print(f"  loaded {len(saes)} SAEs (d_sae={d_sae}, top_k={k})" + " " * 30)
    return saes


def load_gemma_scope_saes(cfg, layers, expected_d_model, device):
    """Download and instantiate JumpReLUSAEs from a Gemma Scope repo.

    Handles both v1 (.npz, Gemma 2) and v2 (.safetensors, Gemma 3) formats.
    Both formats store W_enc as (d_model, d_sae) — opposite of the Qwen convention.
    """
    saes = {}
    repo = cfg["sae_repo"]
    sae_format = cfg["sae_format"]
    width = cfg.get("sae_width", "16k")
    l0 = cfg.get("sae_l0", "47")
    d_sae = 0  # set inside loop; defined here so it's always bound after

    for layer in layers:
        print(f"  loading SAE layer {layer}/{max(layers)} ...", end="\r")

        if sae_format == "gemma_scope_1":
            # Gemma Scope v1 path: layer_{N}/width_{W}/average_l0_{L}/params.npz
            filename = f"layer_{layer}/width_{width}/average_l0_{l0}/params.npz"
            path = hf_hub_download(repo_id=repo, filename=filename)
            raw = np.load(path)
            # W_enc shape is (d_model, d_sae) in Gemma Scope — note the transpose vs Qwen
            W_enc = torch.from_numpy(np.array(raw["W_enc"])).float()
            b_enc = torch.from_numpy(np.array(raw["b_enc"])).float()
            b_dec = torch.from_numpy(np.array(raw["b_dec"])).float()
            threshold = torch.from_numpy(np.array(raw["threshold"])).float()

        else:
            # Gemma Scope v2 path: resid_post_all/layer_{N}_width_{W}_l0_{L}/params.safetensors
            filename = (
                f"resid_post_all/layer_{layer}_width_{width}_l0_{l0}/params.safetensors"
            )
            path = hf_hub_download(repo_id=repo, filename=filename)
            raw = safetensors_load(path, device="cpu")
            W_enc = raw["w_enc"].float()
            b_enc = raw["b_enc"].float()
            b_dec = raw["b_dec"].float()
            threshold = raw["threshold"].float()

        d_model, d_sae = W_enc.shape
        if d_model != expected_d_model:
            raise ValueError(
                f"SAE layer {layer} in {repo} has d_model={d_model}, "
                f"but model hidden_size={expected_d_model}"
            )

        sae = JumpReLUSAE(d_model, d_sae)
        with torch.no_grad():
            sae.W_enc.copy_(W_enc)
            sae.b_enc.copy_(b_enc)
            sae.b_dec.copy_(b_dec)
            sae.threshold.copy_(threshold)
        saes[layer] = sae.to(device).eval()

    print(f"  loaded {len(saes)} JumpReLU SAEs (d_sae={d_sae})" + " " * 30)
    return saes


# ==========================================================================
# GENERATION + HIDDEN STATE CAPTURE
# ==========================================================================


@torch.inference_mode()
def generate_and_capture(texts, model, tok, layers, device, batch_size=BATCH_SIZE, capture_mode="decode"):
    """Generate from each text, capturing hidden states at all layers.

    capture_mode controls which hidden states are stored per sample:
      "decode"         — one vector per generated token (existing behaviour)
      "prefill"        — last-position hidden state of the prompt only; no generation
      "prefill+decode" — prefill last position prepended to all decode-step vectors

    Returns:
        hidden_states: dict[layer] -> list of (n_steps, d_model) float16 numpy arrays
        generations:   list of decoded generation strings (empty strings for prefill-only)
    """
    n = len(texts)
    hidden_states = {layer: [] for layer in layers}
    generations = []
    n_batches = (n + batch_size - 1) // batch_size

    hooks = []
    captured = {}

    def make_hook(layer_idx):
        # layer_idx is a function argument, so it's captured by value correctly;
        # using it directly from the loop variable would cause all hooks to share the last value
        def _hook(_module, _input, output):
            # output may be a plain tensor (Gemma3) or a tuple; output[0] on a plain
            # tensor picks only the first sample, not the hidden states, so unwrap explicitly
            hs = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hs.float()

        return _hook

    for layer in layers:
        h = transformer_layers(model)[layer].register_forward_hook(make_hook(layer))
        hooks.append(h)

    try:
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            batch_texts = texts[start : start + batch_size]
            bs = len(batch_texts)
            print(f"    batch {batch_idx + 1}/{n_batches} ({bs} samples)", end="\r")

            chat_template_applied = tok.chat_template is not None
            if chat_template_applied:
                batch_texts = [
                    tok.apply_chat_template(
                        [{"role": "user", "content": t}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for t in batch_texts
                ]

            enc = tok(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_INPUT_TOKENS,
                # chat template already inserts BOS; avoid duplicating it
                add_special_tokens=not chat_template_applied,
            ).to(device)

            sample_hidden = {layer: [[] for _ in range(bs)] for layer in layers}
            gen_tokens = [[] for _ in range(bs)]
            active = torch.ones(bs, dtype=torch.bool, device=device)

            # prefill: run the full prompt through the model once
            captured.clear()
            out = model(**enc, use_cache=True)
            past_kv = out.past_key_values
            next_tokens = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            # store the last-position prefill hidden state when requested
            if capture_mode in ("prefill", "prefill+decode"):
                for layer in layers:
                    h = captured[layer]
                    if h.dim() == 3:
                        h = h[:, -1, :]  # (bs, d_model) — last prompt position
                    if h.shape[0] == 1 and bs > 1:
                        h = h.expand(bs, -1)
                    for i in range(bs):
                        sample_hidden[layer][i].append(h[i].cpu().half())

            for _step in range(MAX_GEN_TOKENS if capture_mode != "prefill" else 0):
                active = active & (next_tokens.squeeze(-1) != tok.eos_token_id)
                if not active.any():
                    break

                for i in range(bs):
                    if active[i]:
                        gen_tokens[i].append(next_tokens[i, 0].item())

                # decode step: single new token per active sample
                captured.clear()
                out = model(next_tokens, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                next_tokens = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

                for layer in layers:
                    h = captured[layer]
                    # squeeze sequence dim if model returned (batch, 1, d_model)
                    if h.dim() == 3:
                        h = h[:, -1, :]
                    # some model variants broadcast a single hidden state across the batch
                    if h.shape[0] == 1 and bs > 1:
                        h = h.expand(bs, -1)
                    for i in range(bs):
                        if active[i]:
                            sample_hidden[layer][i].append(h[i].cpu().half())

            for layer in layers:
                for i in range(bs):
                    steps = sample_hidden[layer][i]
                    if steps:
                        hidden_states[layer].append(torch.stack(steps).numpy())
                    else:
                        # dead sample: EOS was the very first token
                        d_model = text_config(model).hidden_size
                        hidden_states[layer].append(
                            np.zeros((0, d_model), dtype=np.float16)
                        )

            for i in range(bs):
                generations.append(tok.decode(gen_tokens[i], skip_special_tokens=True))

            del past_kv, out, enc, sample_hidden, gen_tokens
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    finally:
        for h in hooks:
            h.remove()

    print(f"    done ({n} samples, {len(layers)} layers)" + " " * 20)
    return hidden_states, generations


# ==========================================================================
# SAE FIRING RATES
# ==========================================================================


def compute_firing_rates(hidden_states_for_layer, sae, device):
    """Compute per-feature binary firing matrix from stored hidden states.

    Returns (n_samples, d_sae) fired_matrix and (d_sae,) mean firing rate vector.
    A feature is considered fired for a sample if it was active at any generation step.
    Works with both TopKSAE and JumpReLUSAE via the shared fired_mask() interface.
    """
    n = len(hidden_states_for_layer)
    d_sae = sae.d_sae
    fired_matrix = np.zeros((n, d_sae), dtype=np.float32)

    for i in range(n):
        hs = hidden_states_for_layer[i]
        if hs.shape[0] == 0:
            continue
        h_tensor = torch.from_numpy(hs).float().to(device)
        # (n_steps, d_sae) bool → collapse across steps: fired if active at any step
        step_mask = sae.fired_mask(h_tensor)
        fired_matrix[i] = step_mask.any(dim=0).cpu().numpy().astype(np.float32)

    firing_rate = fired_matrix.mean(axis=0)
    return fired_matrix, firing_rate


# ==========================================================================
# FEATURE DISCOVERY
# ==========================================================================


def discover_features(firing_rates_by_topic, k=N_TOP_FEATURES, layer_override=None):
    """Find the layer + k features with highest cross-topic variance in firing rates.

    Returns best_layer, best_feats, layer_scores, best_layer_variances where
    best_layer_variances is the full (d_sae,) variance vector for the best layer.
    """
    topics = list(firing_rates_by_topic.keys())
    layers = list(firing_rates_by_topic[topics[0]].keys())
    layer_scores = {}

    if layer_override is not None:
        stacked = np.stack([firing_rates_by_topic[t][layer_override] for t in topics])
        var = stacked.var(axis=0)
        return layer_override, np.argsort(var)[::-1][:k], {}, var

    best_layer, best_score, best_feats, best_var = None, -1, None, None
    for layer in layers:
        stacked = np.stack([firing_rates_by_topic[t][layer] for t in topics])
        var = stacked.var(axis=0)
        top_k_var = np.sort(var)[::-1][:k]
        score = float(top_k_var.mean())
        layer_scores[layer] = score
        if score > best_score:
            best_score = score
            best_layer = layer
            best_feats = np.argsort(var)[::-1][:k]
            best_var = var

    return best_layer, best_feats, layer_scores, best_var


# ==========================================================================
# MLP TRAINING & EVALUATION
# ==========================================================================


def train_mlp(features, labels, n_classes, device):
    """Train TopicClassifier with AdamW and cross-entropy, printing loss every epoch."""
    n_in = features.shape[1]
    mlp = TopicClassifier(n_in, n_classes, MLP_HIDDEN).to(device)
    # fused=True uses the CUDA fused kernel; requires CUDA and float params
    use_fused = device.startswith("cuda")
    opt = torch.optim.AdamW(mlp.parameters(), lr=MLP_LR, fused=use_fused)
    loss_fn = nn.CrossEntropyLoss()

    X = torch.tensor(features, dtype=torch.float32).to(device)
    y = torch.tensor(labels, dtype=torch.long).to(device)
    n = X.shape[0]

    loss_history = []
    mlp.train()
    for epoch in range(MLP_EPOCHS):
        perm = torch.randperm(n, device=device)
        total_loss, n_batches = 0.0, 0
        for s in range(0, n, MLP_BATCH_SIZE):
            idx = perm[s : s + MLP_BATCH_SIZE]
            loss = loss_fn(mlp(X[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        epoch_loss = total_loss / n_batches
        loss_history.append(epoch_loss)
        print(f"    epoch {epoch + 1}/{MLP_EPOCHS}  loss={epoch_loss:.4f}")

    mlp.eval()
    return mlp, loss_history


def compute_per_topic_metrics(preds, labels, topics):
    """Compute precision, recall, F1 per topic using sklearn, return as dict."""
    raw = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(topics))), zero_division="warn"
    )
    # unpack as arrays so Pyright knows they're indexable
    precision, recall, f1, support = (np.asarray(x) for x in raw)
    metrics = {}
    for i, topic in enumerate(topics):
        metrics[topic] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "n_test": int(support[i]),
        }
    return metrics


# ==========================================================================
# MAIN PIPELINE
# ==========================================================================


def run_for_model_size(model_size):
    """Run the full SAE topic-classification pipeline for one model size."""
    cfg = CONFIGS[model_size]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    configure_runtime_speed()

    print("=" * 72)
    print(f"LOAD  {cfg['model_name']} + SAEs ({cfg['sae_format']}) on {device}")
    print("=" * 72)
    model, tok = load_model(cfg["model_name"], device, dtype)

    # sae_layers overrides layer_stride when only select layers have SAEs (e.g. gemma-scope-9b-it-res)
    if "sae_layers" in cfg:
        layers = list(cfg["sae_layers"])
    else:
        layers = list(
            range(0, text_config(model).num_hidden_layers, int(cfg.get("layer_stride", 1)))
        )

    if cfg["sae_format"] == "qwen":
        saes = load_all_saes(
            cfg["sae_repo"], layers, text_config(model).hidden_size, cfg["top_k_sae"], device
        )
    else:
        saes = load_gemma_scope_saes(cfg, layers, text_config(model).hidden_size, device)

    # -- Stage 1: load & split data ------------------------------------------
    print("\n" + "=" * 72)
    print("STAGE 1  load topic data")
    print("=" * 72)
    data = load_topic_data(DATASETS, N_POS, SEED, jsonl_path=JSONL_DATA_PATH)
    train_data, test_data = split_data(data, TRAIN_RATIO, seed=SEED)
    topics = list(data.keys())
    for topic in topics:
        print(
            f"  {topic}: {len(train_data[topic])} train / {len(test_data[topic])} test"
        )

    # -- Stage 2: generate & capture hidden states ----------------------------
    print("\n" + "=" * 72)
    print("STAGE 2  generate & capture hidden states (all layers)")
    print("=" * 72)

    train_texts, train_labels = build_labeled_set(train_data, topics)
    test_texts, test_labels = build_labeled_set(test_data, topics)

    print(f"  generating from {len(train_texts)} train prompts (capture_mode={CAPTURE_MODE}) ...")
    train_hidden, train_gens = generate_and_capture(
        train_texts, model, tok, layers, device, capture_mode=CAPTURE_MODE
    )

    print(f"  generating from {len(test_texts)} test prompts (capture_mode={CAPTURE_MODE}) ...")
    test_hidden, test_gens = generate_and_capture(
        test_texts, model, tok, layers, device, capture_mode=CAPTURE_MODE
    )

    train_hidden, train_labels, train_texts, train_gens, n_dead_train = (
        filter_dead_samples(
            train_hidden, train_labels, train_texts, train_gens, layers, "train"
        )
    )
    test_hidden, test_labels, test_texts, test_gens, n_dead_test = filter_dead_samples(
        test_hidden, test_labels, test_texts, test_gens, layers, "test"
    )

    total_bytes = sum(
        hs.nbytes
        for layer in layers
        for split in (train_hidden, test_hidden)
        for hs in split[layer]
    )
    print(f"  stored hidden states: {total_bytes / 1e9:.2f} GB")

    # -- Stage 3: run SAEs, compute firing rates, discover features -----------
    print("\n" + "=" * 72)
    print("STAGE 3  SAE firing rates & feature discovery")
    print("=" * 72)

    # keep fired matrices for all layers until discover_features picks the best one
    train_fired = {}  # layer -> (n_train, d_sae) binary matrix
    firing_rates_by_topic = {t: {} for t in topics}

    for layer in layers:
        print(f"  processing L{layer} ...", end="\r")
        fired_matrix, _ = compute_firing_rates(train_hidden[layer], saes[layer], device)
        train_fired[layer] = fired_matrix

        for i, topic in enumerate(topics):
            mask = train_labels == i
            firing_rates_by_topic[topic][layer] = fired_matrix[mask].mean(axis=0)

    print(f"  done ({len(layers)} layers)" + " " * 30)

    best_layer, best_feats, layer_scores, best_layer_variances = discover_features(firing_rates_by_topic)
    assert best_feats is not None, (
        "discover_features returned no features (empty layers?)"
    )
    print(f"  best layer: L{best_layer}")
    print(f"  selected {len(best_feats)} features: {best_feats.tolist()}")

    # Variance stats for the best layer across all d_sae features
    var = best_layer_variances
    var_stats = {
        "mean": float(var.mean()), "std": float(var.std()),
        "p50": float(np.percentile(var, 50)), "p90": float(np.percentile(var, 90)),
        "p95": float(np.percentile(var, 95)), "p99": float(np.percentile(var, 99)),
        "max": float(var.max()), "n_nonzero": int((var > 0).sum()),
    }
    # Variance score for each of the top-100 selected features (in rank order)
    top_feature_variances = var[best_feats].tolist()
    print(f"  variance stats: mean={var_stats['mean']:.5f} p95={var_stats['p95']:.5f} max={var_stats['max']:.5f}")

    # -- Stage 4: build feature vectors & train MLP ---------------------------
    print("\n" + "=" * 72)
    print("STAGE 4  train & evaluate MLP")
    print("=" * 72)

    # slice only the discovered feature columns from the binary fired matrix
    train_feat_vectors = train_fired[best_layer][:, best_feats]

    # only run the SAE on the test set for the best layer — no need to process all layers
    test_fired_matrix, _ = compute_firing_rates(
        test_hidden[best_layer], saes[best_layer], device
    )
    test_feat_vectors = test_fired_matrix[:, best_feats]

    print(
        f"  feature matrix: train {train_feat_vectors.shape}, test {test_feat_vectors.shape}"
    )

    print("  training MLP ...")
    torch.manual_seed(SEED)
    mlp, loss_history = train_mlp(train_feat_vectors, train_labels, len(topics), device)

    with torch.no_grad():
        X_test = torch.tensor(test_feat_vectors, dtype=torch.float32).to(device)
        preds = mlp(X_test).argmax(dim=-1).cpu().numpy()

    per_topic_metrics = compute_per_topic_metrics(preds, test_labels, topics)
    print_results(preds, test_labels, per_topic_metrics)

    # -- Save checkpoint -------------------------------------------------------
    acc = float((preds == test_labels).mean())
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = f"{CHECKPOINT_DIR}/{model_size}_{ts}"

    ref_layer = layers[0]
    train_df = build_dataset_df(
        train_texts, train_gens, train_labels, topics, train_hidden[ref_layer]
    )
    test_df = build_dataset_df(
        test_texts, test_gens, test_labels, topics, test_hidden[ref_layer]
    )
    train_df.to_parquet(f"{run_prefix}_train.parquet", index=False)
    test_df.to_parquet(f"{run_prefix}_test.parquet", index=False)

    # feature vectors saved as parquet for easy inspection
    feat_col_names = pd.Index([f"feat_{int(fid)}" for fid in best_feats])
    feat_df = pd.DataFrame(train_feat_vectors).set_axis(feat_col_names, axis=1)
    feat_df["label"] = train_labels
    feat_df.to_parquet(f"{run_prefix}_train_features.parquet", index=False)

    feat_test_df = pd.DataFrame(test_feat_vectors).set_axis(feat_col_names, axis=1)
    feat_test_df["label"] = test_labels
    feat_test_df.to_parquet(f"{run_prefix}_test_features.parquet", index=False)

    # move to CPU before saving so the checkpoint loads without CUDA
    torch.save(mlp.cpu().state_dict(), f"{run_prefix}_mlp.pt")
    np.save(f"{run_prefix}_best_layer_variances.npy", best_layer_variances)

    # all config + scores + metadata in a single JSON
    meta = {
        "model_size": model_size,
        "model_name": cfg["model_name"],
        "sae_repo": cfg["sae_repo"],
        "sae_format": cfg["sae_format"],
        "top_k_sae": cfg.get("top_k_sae"),
        "layer_stride": cfg.get("layer_stride", 1),
        "topics": topics,
        "n_pos": N_POS,
        "seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "batch_size": BATCH_SIZE,
        "capture_mode": CAPTURE_MODE,
        "max_gen_tokens": MAX_GEN_TOKENS,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "mlp_hidden": MLP_HIDDEN,
        "mlp_epochs": MLP_EPOCHS,
        "mlp_lr": MLP_LR,
        "best_layer": best_layer,
        "feature_ids": best_feats.tolist(),
        "feature_variances": top_feature_variances,
        "best_layer_variance_stats": var_stats,
        "n_features": len(best_feats),
        "layer_scores": layer_scores,
        "accuracy": acc,
        "per_topic_metrics": per_topic_metrics,
        "loss_history": loss_history,
        "dead_samples_train": n_dead_train,
        "dead_samples_test": n_dead_test,
        "train_samples": len(train_labels),
        "test_samples": len(test_labels),
        "timestamp": ts,
    }
    with open(f"{run_prefix}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  saved: {run_prefix}_mlp.pt")
    print(f"         {run_prefix}_meta.json")
    print(f"         {run_prefix}_train.parquet / _test.parquet")
    print(f"         {run_prefix}_train_features.parquet / _test_features.parquet")

    print("\n" + "=" * 72)
    print(f"Done: {model_size}.")
    print("=" * 72)


def parse_model_sizes(raw):
    """Parse and validate a comma-separated list of model size strings."""
    model_sizes = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in model_sizes if item not in CONFIGS]
    if unknown:
        raise ValueError(f"unknown model size(s): {unknown}; valid: {sorted(CONFIGS)}")
    return model_sizes


def main():
    """Entry point: parse args and run pipeline for each requested model size."""
    parser = argparse.ArgumentParser(
        description="SAE-based topic classification training pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-sizes",
        default=",".join(DEFAULT_MODEL_SIZES),
        help=f"Comma-separated model sizes. Valid: {', '.join(CONFIGS)}",
    )
    parser.add_argument("--n-pos", type=int, default=N_POS, help="Max samples per topic.")
    parser.add_argument("--seed", type=int, default=SEED, help="Global RNG seed.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Generation batch size.")
    parser.add_argument(
        "--capture-mode",
        choices=["decode", "prefill", "prefill+decode"],
        default=CAPTURE_MODE,
        help="Which hidden states to capture.",
    )
    parser.add_argument("--n-top-features", type=int, default=N_TOP_FEATURES, help="SAE features selected per layer.")
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO, help="Fraction of data for training.")
    parser.add_argument("--mlp-hidden", type=int, default=MLP_HIDDEN, help="Hidden units in the classifier MLP.")
    parser.add_argument("--mlp-epochs", type=int, default=MLP_EPOCHS, help="MLP training epochs.")
    parser.add_argument("--mlp-lr", type=float, default=MLP_LR, help="AdamW learning rate.")
    parser.add_argument("--mlp-batch-size", type=int, default=MLP_BATCH_SIZE, help="MLP mini-batch size.")
    parser.add_argument("--max-gen-tokens", type=int, default=MAX_GEN_TOKENS, help="Max new tokens per generation.")
    parser.add_argument("--max-input-tokens", type=int, default=MAX_INPUT_TOKENS, help="Prompt truncation length.")
    parser.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR, help="Directory for checkpoint outputs.")
    parser.add_argument("--data-path", type=Path, default=JSONL_DATA_PATH, help="Path to prompts.jsonl from data generation.")
    args = parser.parse_args()

    # Override module-level config with CLI values so all functions see them
    global N_POS, SEED, BATCH_SIZE, CAPTURE_MODE, N_TOP_FEATURES, TRAIN_RATIO
    global MLP_HIDDEN, MLP_EPOCHS, MLP_LR, MLP_BATCH_SIZE, MAX_GEN_TOKENS
    global MAX_INPUT_TOKENS, CHECKPOINT_DIR, JSONL_DATA_PATH
    N_POS = args.n_pos
    SEED = args.seed
    BATCH_SIZE = args.batch_size
    CAPTURE_MODE = args.capture_mode
    N_TOP_FEATURES = args.n_top_features
    TRAIN_RATIO = args.train_ratio
    MLP_HIDDEN = args.mlp_hidden
    MLP_EPOCHS = args.mlp_epochs
    MLP_LR = args.mlp_lr
    MLP_BATCH_SIZE = args.mlp_batch_size
    MAX_GEN_TOKENS = args.max_gen_tokens
    MAX_INPUT_TOKENS = args.max_input_tokens
    CHECKPOINT_DIR = args.checkpoint_dir
    JSONL_DATA_PATH = args.data_path

    for model_size in parse_model_sizes(args.model_sizes):
        run_for_model_size(model_size)


if __name__ == "__main__":
    main()
