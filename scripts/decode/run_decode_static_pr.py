#!/usr/bin/env python3
"""Static SAE-150 decode ablation for GRU / Transformer / ConvNeXt, both backbones.

``--model-size {1b,4b}`` picks the profile. The backbone id, SAE release and layer range are
read from the chosen cache's own metadata.json, so the script cannot disagree with the features
it encodes. Merged from run_decode_1b_static_pr.py and run_decode_4b_static_pr.py; see MERGE.md.

The backbone generates its own response per test prompt (greedy, batched, flash-attn), SAE-1500
features are extracted for [prompt+generated] with the same manual-JumpReLU convention and the
same top-150/layer indices used to build the cache, and every detector scores the generated
RESPONSE tokens. Generation is detector-independent: generate+extract once, run all detectors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer


# local imports
from important_scripts.paths import PROJECT_ROOT
from important_scripts.model.models import TOPICS, build_model, seed_everything

ROOT = PROJECT_ROOT

MANIFEST = ROOT / "ablation/gemma_arm/manifest_4b_10k_test.jsonl"
# Only the three paths differ per backbone. MODEL_ID / SAE_RELEASE / LAYERS come from the
# cache metadata, so pointing at a cache is enough to fix the whole encode convention.
PROFILES = {
    "1b": {
        "cache": "cache/sae1500_10k_1b_pr",
        "baselines": "results/gemma1b_baselines_sae150_10k_pr",
        "out": "ablation/gemma_arm/decode_1b_static_pr",
    },
    "4b": {
        "cache": "cache/sae1500_10k_gemma4b_pr",
        "baselines": "results/gemma4b_baselines_sae150_pr",
        "out": "ablation/gemma_arm/decode_4b_static_pr",
    },
}
DETECTOR_SUBPATHS = {
    "GRU": "gru_lr3e4/checkpoint_best.pt",
    "Transformer": "transformer_wide/checkpoint_best.pt",
    "ConvNeXt": "convnext_window505/checkpoint_best.pt",
}
# Bound by apply_profile() before anything reads them.
MODEL_ID = SAE_RELEASE = ""
LAYERS: list[int] = []
CACHE = OUT = BASE = None
DETECTORS: dict[str, Path] = {}
TAG_RE = None


def apply_profile(model_size: str, cache: str | None = None, baselines: str | None = None,
                  out: str | None = None) -> dict:
    """Bind every backbone-dependent global from one profile and the cache's own metadata.

    ``cache``/``baselines``/``out`` override the profile's three paths, which is how a selection
    variant (F-statistic, group-sparse) is decoded: the backbone, SAE release, layer range and the
    per-layer feature indices all still come from the cache that is named, so the encode convention
    cannot drift from the one the detectors were trained on.
    """
    global MODEL_ID, SAE_RELEASE, LAYERS, CACHE, OUT, BASE, DETECTORS
    profile = dict(PROFILES[model_size])
    if cache:
        profile["cache"] = cache
    if baselines:
        profile["baselines"] = baselines
    if out:
        profile["out"] = out
    CACHE = ROOT / profile["cache"]
    meta = json.loads((CACHE / "metadata.json").read_text())
    MODEL_ID = meta["base_model"]
    SAE_RELEASE = meta["sae_release"]
    LAYERS = [int(layer) for layer in meta["layers"]]
    BASE = ROOT / profile["baselines"]
    OUT = ROOT / profile["out"]
    DETECTORS = {family: BASE / sub for family, sub in DETECTOR_SUBPATHS.items()}
    return profile


def strip_tags(text):
    """Remove every <Topic>/</Topic> marker from a prompt/response string (regex cached globally)."""
    import re

    global TAG_RE
    if TAG_RE is None:
        TAG_RE = re.compile(r"</?(?:" + "|".join(re.escape(t) for t in TOPICS) + r")>")
    return TAG_RE.sub("", text)


def find_backbone(model):
    """Locate the Gemma text-transformer submodule (the one owning ``.layers``) across HF variants."""
    for cand in (
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(model, "language_model", None),
        getattr(model, "model", None),
    ):
        if cand is not None and hasattr(cand, "layers"):
            return cand
    raise TypeError("no backbone")


def load_det(path, device):
    """Rebuild one detector from its checkpoint (architecture + model cfg + weights) onto ``device``."""
    p = torch.load(path, map_location="cpu", weights_only=False)
    m = build_model(p["architecture"], p["model"])
    m.load_state_dict(p["state_dict"])
    return m.to(device).eval()


def prompt_ids(prompt_text, tok):
    """Tokenize a (tag-stripped) prompt with the generation prompt appended, ready for greedy decode."""
    ids = tok.apply_chat_template(
        [{"role": "user", "content": strip_tags(prompt_text)}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if ids and not isinstance(ids[0], int):
        ids = ids[0].ids
    return list(ids)


def main() -> None:
    """Generate responses, extract SAE-1500 features once, and score every detector on the decode set."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--extract-batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--manifest", default=None, help="override MANIFEST (e.g. full-1500 test manifest)"
    )
    ap.add_argument("--outdir", default=None, help="override OUT dir")
    ap.add_argument("--cache", default=None, help="override the profile's feature cache")
    ap.add_argument(
        "--baselines", default=None, help="override the profile's detector checkpoint dir"
    )
    ap.add_argument(
        "--model-size",
        choices=sorted(PROFILES),
        required=True,
        help="backbone profile: selects the cache, baselines and output dir; the model id, "
        "SAE release and layer range come from that cache's metadata.json",
    )
    args = ap.parse_args()
    apply_profile(args.model_size, cache=args.cache, baselines=args.baselines, out=args.outdir)
    manifest_path = Path(args.manifest) if args.manifest else MANIFEST
    out_dir = OUT
    print(f"cache={CACHE}\nbaselines={BASE}", flush=True)
    device = "cuda"
    seed_everything(args.seed, deterministic=True)
    torch.backends.cuda.matmul.allow_tf32 = True

    attn = "flash_attention_2"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, attn_implementation=attn
        )
        .to(device)
        .eval()
    )
    backbone = find_backbone(model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id
    print(f"attn={attn} loaded {MODEL_ID} layers={LAYERS[0]}-{LAYERS[-1]}", flush=True)

    # SAE params with the cache's selected 150 indices per layer (manual JumpReLU)
    sel = np.load(CACHE / "selected_features.npz")
    saesel = {}
    for L in LAYERS:
        s = SAE.from_pretrained(
            release=SAE_RELEASE, sae_id=f"layer_{L}_width_16k_l0_small", device=device
        )
        s = s[0] if isinstance(s, tuple) else s
        idx = torch.as_tensor(sel[f"layer_{L}"], device=device)
        saesel[L] = (
            s.W_enc.float()[:, idx],
            s.b_enc.float()[idx],
            s.b_dec.float(),
            s.threshold.float()[idx],
        )
    dets = {name: load_det(p, device) for name, p in DETECTORS.items()}
    print(f"loaded 10 SAEs + {len(dets)} detectors", flush=True)

    end_of_turn = tok.convert_tokens_to_ids("<end_of_turn>")
    eos_ids = {tok.eos_token_id, end_of_turn}
    special = set(tok.all_special_ids) | {end_of_turn}

    manifest = [json.loads(l) for l in manifest_path.open()]
    if args.limit:
        manifest = manifest[: args.limit]
    prompts = [prompt_ids(r["prompt"], tok) for r in manifest]
    gen_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "pad_token_id": pad_id,
        "use_cache": True,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"manifest={manifest_path} ({len(manifest)} records) -> out={out_dir}", flush=True)

    # GPT reference responses: manifest "index" == line number in dataset.final.jsonl
    DATASET = ROOT / "runs/GPT_5.6_10k_2108/dataset.final.jsonl"
    ds = [json.loads(l) for l in DATASET.open()]
    gpt_by_index = {
        i: strip_tags(ds[i]["record"]["response"]) for i in {r["index"] for r in manifest}
    }

    @torch.inference_mode()
    def extract(full_ids_list):
        """Batched manual-JumpReLU SAE-1500 features; returns list of (T,1500) fp32 numpy."""
        width = max(len(x) for x in full_ids_list)
        ii = torch.full((len(full_ids_list), width), pad_id, dtype=torch.long)
        am = torch.zeros((len(full_ids_list), width), dtype=torch.long)
        for r, s in enumerate(full_ids_list):
            ii[r, : len(s)] = torch.tensor(s)
            am[r, : len(s)] = 1
        out = backbone(
            input_ids=ii.to(device),
            attention_mask=am.to(device),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        res = []
        for r, s in enumerate(full_ids_list):
            cols = []
            for L in LAYERS:
                h = out.hidden_states[L + 1][r, : len(s)].float()
                W, be, bd, thr = saesel[L]
                pre = (h - bd) @ W + be
                cols.append(pre * (pre > thr))
            res.append(torch.cat(cols, -1).float().cpu().numpy())
        return res

    per_record = []
    bs = args.batch_size
    t0 = __import__("time").time()
    for start in range(0, len(manifest), bs):
        chunk = list(range(start, min(start + bs, len(manifest))))
        batch_ids = [prompts[i] for i in chunk]
        width = max(len(x) for x in batch_ids)
        input_ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        attn_mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for r, ids in enumerate(batch_ids):
            input_ids[r, width - len(ids) :] = torch.tensor(ids)
            attn_mask[r, width - len(ids) :] = 1
        with torch.inference_mode():
            out = model.generate(
                input_ids.to(device), attention_mask=attn_mask.to(device), **gen_config
            )
        out = out.cpu().tolist()

        # assemble full [prompt+gen_content] per record, then extract features in sub-batches
        full_list, metas = [], []
        for r, gi in enumerate(chunk):
            row = manifest[gi]
            p_ids = batch_ids[r]
            gen_full = out[r][width:]
            cut = len(gen_full)
            for j, t in enumerate(gen_full):
                if t in eos_ids:
                    cut = j
                    break
            gen_content = gen_full[:cut]
            plen = len(p_ids)
            resp_positions = [plen + k for k, t in enumerate(gen_content) if t not in special]
            gemma_text = tok.decode(gen_content, skip_special_tokens=True)
            metas.append((row, plen, len(gen_content), resp_positions, gemma_text))
            full_list.append(p_ids + gen_content)

        preds_all = {name: [None] * len(chunk) for name in dets}
        for es in range(0, len(full_list), args.extract_batch):
            sub = list(range(es, min(es + args.extract_batch, len(full_list))))
            feats = extract([full_list[k] for k in sub])
            for k, fidx in zip(sub, range(len(sub)), strict=False):
                row, plen, ngen, resp_positions, _gtext = metas[k]
                if not resp_positions:
                    continue
                x = torch.from_numpy(feats[fidx])[None].to(device)
                with torch.inference_mode():
                    for name, m in dets.items():
                        pf = m(x)[0].argmax(-1).cpu().numpy()
                        preds_all[name][k] = pf[resp_positions].astype(int).tolist()

        for k, gi in enumerate(chunk):
            row, plen, ngen, resp_positions, gemma_text = metas[k]
            per_record.append(
                {
                    "index": row["index"],
                    "topics": row["topics"],
                    "single_topic": row["single_topic"],
                    "assumed_label_id": row["assumed_label_id"],
                    "generated_tokens": ngen,
                    "response_content_tokens": len(resp_positions),
                    "prompt": strip_tags(row["prompt"]),
                    "gpt_response": gpt_by_index.get(row["index"], ""),
                    "gemma_response": gemma_text,
                    "pred": {
                        name: preds_all[name][k] for name in dets if preds_all[name][k] is not None
                    },
                }
            )
        el = __import__("time").time() - t0
        done = min(start + bs, len(manifest))
        print(
            f"{done}/{len(manifest)} generated+scored  ({el:.0f}s, {el / done:.1f}s/rec)",
            flush=True,
        )

    with (out_dir / "per_record.jsonl").open("w") as f:
        for r in per_record:
            f.write(json.dumps(r) + "\n")
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "attn": attn,
                "model_size": args.model_size,
                "model": MODEL_ID,
                "cache": str(CACHE),
                "baselines": str(BASE),
                "layers": LAYERS,
                "batch_size": bs,
                "max_new_tokens": args.max_new_tokens,
                "seed": args.seed,
                "manifest": str(manifest_path),
                "detectors": {k: str(v) for k, v in DETECTORS.items()},
                "records": len(per_record),
            },
            indent=2,
        )
    )
    print(f"wrote {out_dir}/per_record.jsonl ({len(per_record)} records)", flush=True)


if __name__ == "__main__":
    main()
