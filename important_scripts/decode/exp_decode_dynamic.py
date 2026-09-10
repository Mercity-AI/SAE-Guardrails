#!/usr/bin/env python3
"""Decode ablation with DYNAMIC per-token feature selection.

One-for-one replica of ablation/gemma_arm/run_gemma_arm.py (same prompts, greedy config,
same per-sequence extraction and response-token scoring vs the assumed/prompt label), but
the feature stage is dynamic: per token, per layer, take that token's own top-K JumpReLU
features, map them through the TRAINING vocabulary (dyn150_sparse/selected_features.npz) to
their permanent union columns, and score with the dynamic-trained detectors (results/dyn150_2k).

All feature/prediction tensors are torch (no numpy storage); the per-record predictions are
persisted as a single torch .pt file. Greedy decoding + argmax scoring are deterministic;
seed 42 is set throughout (deterministic=False: cuDNN determinism is not required here).

Serves both backbones. ``--model-size {1b,4b}`` picks the profile; every cache, checkpoint
and output path is derived from it. Merged from exp_decode_dynamic_1b10k.py and
exp_decode_dynamic_4b10k.py, which differed only in those constants. See MERGE.md.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np  # only for reading the vocab .npz + reference counts
import torch
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer


# local imports
from important_scripts.paths import PROJECT_ROOT
from important_scripts.model.models import TOPICS, build_model, seed_everything

ROOT = PROJECT_ROOT

# PROMPT+RESPONSE arm: test-only manifest (no train/val leak) + PR-trained dynamic detectors.
MANIFEST = ROOT / "ablation/gemma_arm/manifest_4b_10k_test.jsonl"
DATASET = (
    ROOT / "runs/GPT_5.6_10k_2108/dataset.final.jsonl"
)  # GPT reference responses (join by index)
PROFILES = {
    "1b": {
        "gemma_id": "google/gemma-3-1b-it",
        "sae_release": "gemma-scope-2-1b-it-res-all",
        "layers": list(range(16, 26)),
        "cache": "cache/dyn150_sparse_10k_1b",
        "results": "results/dyn150_10k_1b_pr",
    },
    "4b": {
        "gemma_id": "google/gemma-3-4b-it",
        "sae_release": "gemma-scope-2-4b-it-res-all",
        "layers": list(range(24, 34)),
        "cache": "cache/dyn150_sparse_10k_4b",
        "results": "results/dyn150_10k_4b_pr",
    },
}
DETECTOR_SUBPATHS = {
    "GRU": "gru/gru_lr3e4/checkpoint_best.pt",
    "Transformer": "transformer/transformer_wide/checkpoint_best.pt",
    "ConvNeXt": "tcn/convnext/checkpoint_best.pt",
}
# Bound by apply_profile() before anything reads them; --model-size selects the profile.
VOCAB = META = OUT_JSON = OUT_PT = OUT_JSONL = None
CKPT: dict[str, Path] = {}
GEMMA_ID = REL = ""
LAYERS: list[int] = []
DEVICE = "cuda"


def apply_profile(model_size: str) -> dict:
    """Bind every backbone-dependent global from one profile and return it.

    The sparse width, per-token K and per-layer column offsets are NOT set here: they are read
    from the profile's own meta.json and selected_features.npz at run time, and cross-checked
    against each detector checkpoint, so a mismatched profile fails an assertion rather than
    scoring quietly against the wrong vocabulary.
    """
    global VOCAB, META, CKPT, OUT_JSON, OUT_PT, OUT_JSONL, GEMMA_ID, REL, LAYERS
    profile = PROFILES[model_size]
    cache, results = ROOT / profile["cache"], ROOT / profile["results"]
    VOCAB = cache / "selected_features.npz"
    META = cache / "meta.json"
    CKPT = {family: results / sub for family, sub in DETECTOR_SUBPATHS.items()}
    OUT_JSON = results / "decode_dynamic.json"
    OUT_PT = results / "decode_dynamic_preds.pt"
    OUT_JSONL = results / "decode_dynamic_per_record.jsonl"
    GEMMA_ID = profile["gemma_id"]
    REL = profile["sae_release"]
    LAYERS = list(profile["layers"])
    return profile
SEED = 42
TAG_RE = re.compile(r"</?(?:" + "|".join(re.escape(t) for t in TOPICS) + r")>")


def strip_tags(t):
    """Remove every <Topic>/</Topic> marker from the given string."""
    return TAG_RE.sub("", t)


def prompt_input_ids(prompt_text, tokenizer):
    """Tokenize a (tag-stripped) prompt with the generation prompt appended, ready for greedy decode."""
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": strip_tags(prompt_text)}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if ids and not isinstance(ids[0], int):
        ids = ids[0].ids
    return list(ids)


def load_detector(ckpt):
    """Rebuild one detector from its checkpoint (architecture + model cfg + weights) onto the device."""
    p = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = build_model(p["architecture"], p["model"])
    m.load_state_dict(p["state_dict"])
    return m.to(DEVICE).eval(), int(p["model"]["input_features"])


@torch.inference_mode()
def dynamic_features(hidden_states, saes, luts, layer_off, K, W):
    """Build a dense (T, W) torch feature tensor for one sequence, dynamic top-K per token."""
    T = hidden_states[LAYERS[0] + 1].shape[1]
    x = torch.zeros(T, W, device=DEVICE)
    rows_full = torch.arange(T, device=DEVICE).unsqueeze(1).expand(-1, K)  # (T,K)
    for li, (L, Wenc, b_enc, b_dec, thr) in enumerate(saes):
        act = hidden_states[L + 1][0].float()  # (T, 1152)
        pre = (act - b_dec) @ Wenc + b_enc
        feat = pre * (pre > thr)
        vals, idx = torch.topk(feat, K, dim=-1)  # (T, K)
        mapped = luts[li][idx]  # (T, K), -1 if not in training vocab
        valid = (vals > 0) & (mapped >= 0)
        gcol = layer_off[li] + mapped  # (T, K)
        x[rows_full[valid], gcol[valid]] = vals[valid]
    return x


def main():
    """Generate responses, extract features once, and score every detector on the held-out decode set."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument(
        "--model-size",
        choices=sorted(PROFILES),
        required=True,
        help="backbone profile: selects model id, SAE release, layers, cache and results paths",
    )
    args = ap.parse_args()
    apply_profile(args.model_size)
    seed_everything(SEED, deterministic=False)

    tok = AutoTokenizer.from_pretrained(GEMMA_ID)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id
    gemma = (
        AutoModelForCausalLM.from_pretrained(
            GEMMA_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .to(DEVICE)
        .eval()
    )

    # SAEs (full dictionary) for per-token top-K
    saes = []
    for L in LAYERS:
        s = SAE.from_pretrained(release=REL, sae_id=f"layer_{L}_width_16k_l0_small", device=DEVICE)
        s.eval()
        saes.append((L, s.W_enc.float(), s.b_enc.float(), s.b_dec.float(), s.threshold.float()))
    d_sae = saes[0][1].shape[1]

    # training vocabulary -> per-layer lookup tables (raw SAE idx -> union column offset)
    meta = json.loads(META.read_text())
    K, W = int(meta["K"]), int(meta["width"])
    vocab = np.load(VOCAB)
    widths, luts = [], []
    for L in LAYERS:
        keep = torch.as_tensor(vocab[f"layer_{L}"], device=DEVICE, dtype=torch.long)
        lut = torch.full((d_sae,), -1, dtype=torch.long, device=DEVICE)
        lut[keep] = torch.arange(len(keep), device=DEVICE)
        luts.append(lut)
        widths.append(len(keep))
    layer_off = torch.tensor(
        np.concatenate(([0], np.cumsum(widths))), device=DEVICE, dtype=torch.long
    )
    assert int(layer_off[-1]) == W, f"vocab width {int(layer_off[-1])} != meta {W}"

    detectors = {}
    for fam, ck in CKPT.items():
        m, dim = load_detector(ck)
        assert dim == W, f"{fam} expects {dim} != vocab width {W}"
        detectors[fam] = m

    end_of_turn = tok.convert_tokens_to_ids("<end_of_turn>")
    eos_ids = {tok.eos_token_id, end_of_turn}
    special = set(tok.all_special_ids) | {end_of_turn}

    manifest = [json.loads(l) for l in MANIFEST.open()]
    if args.limit:
        manifest = manifest[: args.limit]
    prompts = [prompt_input_ids(r["prompt"], tok) for r in manifest]
    gen_config = {"max_new_tokens": args.max_new_tokens, "do_sample": False, "pad_token_id": pad_id}

    # GPT reference responses: manifest "index" == line number in dataset.final.jsonl
    ds = [json.loads(l) for l in DATASET.open()]
    gpt_by_index = {
        i: strip_tags(ds[i]["record"]["response"]) for i in {r["index"] for r in manifest}
    }

    # per-arm accumulators (torch/python only)
    single = {f: [] for f in CKPT}  # assumed-label acc per single-topic record
    two = {f: [] for f in CKPT}  # (set_recall, set_prec) per two-topic record
    saved = []  # per-record predictions (torch tensors)
    bs = args.batch_size

    for start in range(0, len(manifest), bs):
        chunk = list(range(start, min(start + bs, len(manifest))))
        batch_ids = [prompts[i] for i in chunk]
        width = max(len(x) for x in batch_ids)
        input_ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        attn = torch.zeros((len(chunk), width), dtype=torch.long)
        for r, ids in enumerate(batch_ids):
            input_ids[r, width - len(ids) :] = torch.tensor(ids)
            attn[r, width - len(ids) :] = 1
        with torch.inference_mode():
            out = gemma.generate(input_ids.to(DEVICE), attention_mask=attn.to(DEVICE), **gen_config)
        out = out.cpu().tolist()

        for r, gi in enumerate(chunk):
            row = manifest[gi]
            prompt_ids = batch_ids[r]
            gen_full = out[r][width:]
            cut = len(gen_full)
            for j, t in enumerate(gen_full):
                if t in eos_ids:
                    cut = j
                    break
            gen_content = gen_full[:cut]
            full_ids = prompt_ids + gen_content
            plen = len(prompt_ids)
            resp_positions = [plen + k for k, t in enumerate(gen_content) if t not in special]
            rec = {
                "index": row["index"],
                "single_topic": row["single_topic"],
                "assumed_label_id": row["assumed_label_id"],
                "topics": row["topics"],
                "generated_tokens": len(gen_content),
                "response_tokens": len(resp_positions),
                "prompt": strip_tags(row["prompt"]),
                "gpt_response": gpt_by_index.get(row["index"], ""),
                "gemma_response": tok.decode(gen_content, skip_special_tokens=True),
            }
            if resp_positions:
                with torch.inference_mode():
                    hs = gemma(
                        input_ids=torch.tensor([full_ids], device=DEVICE),
                        use_cache=False,
                        output_hidden_states=True,
                        return_dict=True,
                    ).hidden_states
                    x = dynamic_features(hs, saes, luts, layer_off, K, W)  # (T, W) torch
                    rp = torch.tensor(resp_positions, device=DEVICE)
                    preds = {}
                    for fam, m in detectors.items():
                        pred_full = m(x[None])[0].argmax(-1)  # (T,)
                        preds[fam] = pred_full[rp].to("cpu")  # response preds (torch)
                rec["pred"] = preds
                for fam in CKPT:
                    pr = preds[fam]
                    if row["single_topic"]:
                        single[fam].append(float((pr == row["assumed_label_id"]).float().mean()))
                    else:
                        truth = {TOPICS.index(t) for t in row["topics"]}
                        ps = {int(v) for v in pr.tolist()}
                        two[fam].append(
                            (len(ps & truth) / len(truth), len(ps & truth) / max(1, len(ps)))
                        )
            saved.append(rec)
        print(f"  {min(start + bs, len(manifest))}/{len(manifest)} generated+scored", flush=True)

    def mean(v):
        return float(np.mean(v)) if v else None

    def median(v):
        return float(np.median(v)) if v else None

    result = {
        "arm": "gemma-decode-dynamic",
        "model": GEMMA_ID,
        "model_size": args.model_size,
        "decoding": "greedy",
        "seed": SEED,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": bs,
        "width": W,
        "K": K,
        "test_records": len(manifest),
        "n_single": sum(r["single_topic"] for r in saved),
        "n_two": sum(not r["single_topic"] for r in saved),
        "generated_tokens_mean": float(np.mean([r["generated_tokens"] for r in saved])),
        "per_arm": {},
    }
    for fam in CKPT:
        t = two[fam]
        result["per_arm"][fam] = {
            "single_assumed_acc_mean": mean(single[fam]),
            "single_assumed_acc_median": median(single[fam]),
            "two_set_recall_mean": mean([a for a, _ in t]),
            "two_set_prec_mean": mean([b for _, b in t]),
        }
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    torch.save({"records": saved, "topics": TOPICS, "seed": SEED, "width": W, "K": K}, OUT_PT)
    # Human-readable per-record log: texts + response predictions as plain lists (mirrors static decode).
    with OUT_JSONL.open("w") as f:
        for rec in saved:
            j = {k: v for k, v in rec.items() if k != "pred"}
            if "pred" in rec:
                j["pred"] = {fam: t.tolist() for fam, t in rec["pred"].items()}
            f.write(json.dumps(j) + "\n")
    print(json.dumps(result, indent=2))
    print(f"\nwrote {OUT_JSON}, {OUT_PT}, {OUT_JSONL}")


if __name__ == "__main__":
    main()
