"""Shared Gemma-1B + Gemma-Scope loading and feature extraction for the decode ablation.

Replicates the exact conventions of the original feature-cache builder (see
scrap/scripts/simulate_live_prefill.py) so features are byte-compatible with the
cache the GRU was trained on:
  * SAE release gemma-scope-2-1b-it-res-all, layer_{N}_width_16k_l0_small, layers 16-25
  * legacy encode: relu((act - b_dec) @ W_enc[:, idx] + b_enc[idx])   (NOT sae.encode)
  * hidden_states[layer + 1] (index 0 is embeddings), features stored float16
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer

# local imports
from important_scripts.paths import PROJECT_ROOT
from important_scripts.model.models import TOPICS

ROOT = PROJECT_ROOT

CACHE = ROOT / "cache/sae500_2k_clean_prompt_response"
GEMMA_ID = "google/gemma-3-1b-it"
SAE_RELEASE = "gemma-scope-2-1b-it-res-all"
LAYERS = list(range(16, 26))
TAG_RE = re.compile(r"</?(?:" + "|".join(re.escape(t) for t in TOPICS) + r")>")


def strip_tags(text: str) -> str:
    return TAG_RE.sub("", text)


def load_gemma(device: str, attn: str = "sdpa"):
    tokenizer = AutoTokenizer.from_pretrained(GEMMA_ID)
    model = AutoModelForCausalLM.from_pretrained(
        GEMMA_ID, torch_dtype=torch.bfloat16, attn_implementation=attn
    ).to(device).eval()
    return tokenizer, model


def load_saes(device: str):
    selected = np.load(CACHE / "selected_features.npz")
    saes = []
    for layer in LAYERS:
        sae = SAE.from_pretrained(
            release=SAE_RELEASE,
            sae_id=f"layer_{layer}_width_16k_l0_small",
            device=device,
        )
        sae.eval()
        idx = torch.as_tensor(selected[f"layer_{layer}"], device=device)
        saes.append((layer, sae, idx))
        print(f"loaded SAE layer {layer}", flush=True)
    return saes


def legacy_sae_encode_selected(sae, activation: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Cache-builder convention (reverse-engineered from the 2k cache; corr>0.999).

    JumpReLU with manual b_dec centering: center on the decoder bias, project,
    then gate by the learned per-feature threshold. NOTE: sae.encode() does NOT
    reproduce this because this SAE has apply_b_dec_to_input=False, yet the cache
    builder subtracted b_dec anyway. Plain relu (no threshold) drifts at depth.
    """
    pre = (activation - sae.b_dec) @ sae.W_enc[:, indices] + sae.b_enc[indices]
    return pre * (pre > sae.threshold[indices])


@torch.inference_mode()
def extract_features(input_ids, gemma, saes, device: str) -> np.ndarray:
    """One prefill pass over input_ids -> (T, 500) float16 SAE feature rows."""
    tokens = torch.tensor([list(input_ids)], device=device)
    output = gemma(input_ids=tokens, use_cache=False, output_hidden_states=True, return_dict=True)
    columns = []
    for layer, sae, indices in saes:
        activation = output.hidden_states[layer + 1][0].float()
        columns.append(legacy_sae_encode_selected(sae, activation, indices))
    return torch.cat(columns, dim=-1).cpu().numpy().astype(np.float16)


def encode_teacher_forced(record: dict, tokenizer):
    """Rebuild a GPT record's exact input_ids (tags stripped), as the cache builder did."""
    prompt = strip_tags(record["prompt"])
    response = strip_tags(record["response"])
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
        tokenize=True, add_generation_prompt=False,
    )
    if ids and not isinstance(ids[0], int):
        ids = ids[0].ids
    return list(ids)


def prompt_input_ids(prompt_text: str, tokenizer):
    """Templated prompt with the assistant generation prompt appended, ready for generate()."""
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": strip_tags(prompt_text)}],
        tokenize=True, add_generation_prompt=True,
    )
    if ids and not isinstance(ids[0], int):
        ids = ids[0].ids
    return list(ids)
