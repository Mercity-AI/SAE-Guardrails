#!/usr/bin/env python3
"""Raw-hidden (no-SAE) decode ablation, PROMPT+RESPONSE footing -- 4B.

4B analogue of decode_raw_1b_pr.py, built on run_decode_static_pr.py's batched generation +
sub-batched extraction, but the feature stage is RAW hidden -- the concat of the 10 mid-stack
layers' residual states (10 x 2560 = 25600 dims) -- instead of SAE-1500, and the detectors are
the PR raw-hidden 4B checkpoints from ablation/nosae_stream_4b_10k_pr/raw.

Gemma-3-4B generates its own response per test prompt (greedy, batched, flash-attn); raw features
for [prompt + generated] are extracted (decode hidden states == prefill hidden states for the same
tokens) and every detector scores the generated RESPONSE tokens. Generation is detector-independent:
generate+extract once, run all detectors. Greedy + argmax; seed 42. Test-only manifest, 8000-cap.

  python decode_raw_4b_pr.py --batch-size 64 --max-new-tokens 8000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# local imports
from important_scripts.paths import PROJECT_ROOT
from important_scripts.model.models import TOPICS, build_model, seed_everything

ROOT = PROJECT_ROOT

MODEL_ID = "google/gemma-3-4b-it"
LAYERS = list(range(24, 34))
MANIFEST = ROOT / "ablation/gemma_arm/manifest_4b_10k_test.jsonl"  # test-only (1000)
DATASET = ROOT / "runs/GPT_5.6_10k_2108/dataset.final.jsonl"
CKPT_DIR = ROOT / "ablation/nosae_stream_4b_10k_pr/raw"  # PR raw-hidden 4B detectors
DETECTORS = {
    "GRU": CKPT_DIR / "gru" / "checkpoint_best.pt",
    "Transformer": CKPT_DIR / "transformer" / "checkpoint_best.pt",
    "ConvNeXt": CKPT_DIR / "convnext" / "checkpoint_best.pt",
}
TAG_RE = None


def strip_tags(text):
    """Remove every <Topic>/</Topic> marker from the given string."""
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
    """Rebuild one detector from its checkpoint (architecture + model cfg + weights) onto the device."""
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


def main():
    """Generate responses, extract features once, and score every detector on the held-out decode set."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=8000)
    ap.add_argument("--extract-batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="ablation/nosae_stream_4b_10k_pr")
    args = ap.parse_args()
    device = "cuda"
    out_dir = ROOT / args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)
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
    print(f"attn={attn} loaded gemma-3-4b-it", flush=True)

    dets = {name: load_det(p, device) for name, p in DETECTORS.items()}
    print(f"loaded {len(dets)} raw detectors (raw dim = {len(LAYERS)}x hidden)", flush=True)

    end_of_turn = tok.convert_tokens_to_ids("<end_of_turn>")
    eos_ids = {tok.eos_token_id, end_of_turn}
    special = set(tok.all_special_ids) | {end_of_turn}

    manifest = [json.loads(l) for l in MANIFEST.open()]
    if args.limit:
        manifest = manifest[: args.limit]
    prompts = [prompt_ids(r["prompt"], tok) for r in manifest]
    gen_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "pad_token_id": pad_id,
        "use_cache": True,
    }
    print(f"manifest={MANIFEST.name} ({len(manifest)} records) -> {out_dir}", flush=True)

    ds = [json.loads(l) for l in DATASET.open()]
    gpt_by_index = {
        i: strip_tags(ds[i]["record"]["response"]) for i in {r["index"] for r in manifest}
    }

    @torch.inference_mode()
    def extract(full_ids_list):
        """Batched RAW hidden features; returns list of (T, hidden*len(LAYERS)) fp32 numpy."""
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
            cols = [out.hidden_states[L + 1][r, : len(s)].float() for L in LAYERS]
            res.append(torch.cat(cols, -1).float().cpu().numpy())
        return res

    per_record = []
    bs = args.batch_size
    t0 = time.time()
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
        el = time.time() - t0
        done = min(start + bs, len(manifest))
        print(
            f"{done}/{len(manifest)} generated+scored  ({el:.0f}s, {el / done:.1f}s/rec)",
            flush=True,
        )

    # ---- scoring: single-topic assumed-label acc; two-topic set recall/precision ----
    metrics = {
        "arm": "raw-hidden-decode-pr",
        "model": MODEL_ID,
        "decoding": "greedy",
        "seed": args.seed,
        "attn": attn,
        "max_new_tokens": args.max_new_tokens,
        "manifest": str(MANIFEST),
        "ckpt_dir": str(CKPT_DIR),
        "test_records": len(per_record),
        "n_single": sum(r["single_topic"] for r in per_record),
        "n_two": sum(not r["single_topic"] for r in per_record),
        "generated_tokens_mean": float(np.mean([r["generated_tokens"] for r in per_record])),
        "per_arm": {},
    }
    for name in DETECTORS:
        single_acc, two_rec, two_prec = [], [], []
        for r in per_record:
            if not r["pred"].get(name):
                continue
            pred = np.asarray(r["pred"][name])
            if r["single_topic"]:
                single_acc.append(float((pred == r["assumed_label_id"]).mean()))
            else:
                truth = {TOPICS.index(t) for t in r["topics"]}
                pset = {int(v) for v in pred}
                two_rec.append(len(pset & truth) / len(truth))
                two_prec.append(len(pset & truth) / max(1, len(pset)))
        metrics["per_arm"][name] = {
            "single_assumed_acc_mean": float(np.mean(single_acc)) if single_acc else None,
            "single_assumed_acc_median": float(np.median(single_acc)) if single_acc else None,
            "two_set_recall_mean": float(np.mean(two_rec)) if two_rec else None,
            "two_set_prec_mean": float(np.mean(two_prec)) if two_prec else None,
        }

    (out_dir / "decode_raw.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with (out_dir / "decode_raw_per_record.jsonl").open("w") as f:
        for r in per_record:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(metrics["per_arm"], indent=2))
    print(f"\nwrote {out_dir}/decode_raw.json\nRAW DECODE 4B PR DONE", flush=True)


if __name__ == "__main__":
    main()
