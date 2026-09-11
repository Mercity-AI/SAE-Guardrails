#!/usr/bin/env python3
"""Raw-hidden (no-SAE) decode ablation, PROMPT+RESPONSE footing -- 1B.

Non-destructive PR variant of decode_raw_1b.py. Differences (all to match the other PR decodes,
run_decode_static_pr.py / exp_decode_dynamic.py):
  * MANIFEST = ablation/gemma_arm/manifest_4b_10k_test.jsonl (test-only; the old manifest_4b_10k
    leaked 725 train / 145 val into the 1000-prompt decode set).
  * detectors = the PR raw-hidden checkpoints ablation/nosae_stream_1b_10k_pr/raw/{gru,transformer,convnext}.
  * text logging: each record carries the (tag-stripped) prompt, the GPT reference response, and the
    Gemma-generated response, exactly like the static/dynamic PR decode per_record.jsonl.
  * generation cap defaults to 8000 tokens, EOS-natural (the 1536 cap truncated ~8-14% of responses).

Gemma-3-1B generates its own response per test prompt (greedy, batched); raw features for
[prompt + generated] are extracted (decode hidden states == prefill hidden states for the same
tokens) and every raw detector scores the generated RESPONSE tokens. Generation is
detector-independent, so we generate+extract once and run all three. Greedy + argmax; seed 42.

  python decode_raw_1b_pr.py --batch-size 32 --max-new-tokens 8000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


# local imports
from important_scripts.paths import PROJECT_ROOT
from important_scripts.decode.gemma_common import load_gemma, prompt_input_ids, strip_tags
from important_scripts.model.models import TOPICS, build_model, seed_everything

ROOT = PROJECT_ROOT

MANIFEST = ROOT / "ablation/gemma_arm/manifest_4b_10k_test.jsonl"  # test-only decode prompts
DATASET = ROOT / "runs/GPT_5.6_10k_2108/dataset.final.jsonl"  # GPT reference responses
LAYERS = list(range(16, 26))
CKPT_DIR = ROOT / "ablation/nosae_stream_1b_10k_pr/raw"  # PR raw-hidden detectors
DETECTORS = {
    "GRU": CKPT_DIR / "gru" / "checkpoint_best.pt",
    "Transformer": CKPT_DIR / "transformer" / "checkpoint_best.pt",
    "ConvNeXt": CKPT_DIR / "convnext" / "checkpoint_best.pt",
}
DEVICE = "cuda"


def load_det(path):
    """Rebuild one detector from its checkpoint (architecture + model cfg + weights) onto the device."""
    p = torch.load(path, map_location="cpu", weights_only=False)
    m = build_model(p["architecture"], p["model"])
    m.load_state_dict(p["state_dict"])
    return m.to(DEVICE).eval()


@torch.inference_mode()
def extract_raw(full_ids, gemma):
    """One prefill over [prompt+generated] -> (T, HIDDEN*len(LAYERS)) float32 raw rows.

    hidden_states[L+1] is the output of layer L (index 0 is embeddings) -- exactly the tensor the
    training hook on backbone.layers[L] captured, so decode features match training features."""
    tokens = torch.tensor([list(full_ids)], device=DEVICE)
    out = gemma(input_ids=tokens, use_cache=False, output_hidden_states=True, return_dict=True)
    cols = [out.hidden_states[L + 1][0].float() for L in LAYERS]
    return torch.cat(cols, dim=-1)  # (T, HIDDEN*10) on GPU


def main():
    """Generate responses, extract features once, and score every detector on the held-out decode set."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="ablation/nosae_stream_1b_10k_pr")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    outdir = ROOT / a.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    seed_everything(a.seed, deterministic=True)

    attn = "flash_attention_2"
    try:
        tokenizer, gemma = load_gemma(DEVICE, attn=attn)
    except Exception as e:
        print(f"flash_attention_2 unavailable ({str(e)[:60]}); using sdpa", flush=True)
        attn = "sdpa"
        tokenizer, gemma = load_gemma(DEVICE, attn=attn)
    print(f"attn={attn} loaded gemma-3-1b-it", flush=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id
    dets = {name: load_det(p) for name, p in DETECTORS.items()}
    print(f"loaded {len(dets)} raw detectors", flush=True)

    end_of_turn = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    eos_ids = {tokenizer.eos_token_id, end_of_turn}
    special = set(tokenizer.all_special_ids) | {end_of_turn}

    manifest = [json.loads(l) for l in MANIFEST.open()]
    if a.limit:
        manifest = manifest[: a.limit]
    prompts = [prompt_input_ids(r["prompt"], tokenizer) for r in manifest]
    gen_config = {"max_new_tokens": a.max_new_tokens, "do_sample": False, "pad_token_id": pad_id}

    # GPT reference responses: manifest "index" == line number in dataset.final.jsonl
    ds = [json.loads(l) for l in DATASET.open()]
    gpt_by_index = {
        i: strip_tags(ds[i]["record"]["response"]) for i in {r["index"] for r in manifest}
    }

    per_record, bs = [], a.batch_size
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
            out = gemma.generate(
                input_ids.to(DEVICE), attention_mask=attn_mask.to(DEVICE), **gen_config
            )
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
            gemma_text = tokenizer.decode(gen_content, skip_special_tokens=True)
            rec = {
                "index": row["index"],
                "topics": row["topics"],
                "single_topic": row["single_topic"],
                "assumed_label_id": row["assumed_label_id"],
                "generated_tokens": len(gen_content),
                "response_content_tokens": len(resp_positions),
                "prompt": strip_tags(row["prompt"]),
                "gpt_response": gpt_by_index.get(row["index"], ""),
                "gemma_response": gemma_text,
                "pred": {},
            }
            if resp_positions:
                x = extract_raw(full_ids, gemma)[None]  # (1, T, 11520) on GPU
                with torch.inference_mode():
                    for name, m in dets.items():
                        pred_full = m(x)[0].argmax(-1).cpu().numpy()
                        rec["pred"][name] = pred_full[resp_positions].astype(int).tolist()
            per_record.append(rec)
        print(
            f"{min(start + bs, len(manifest))}/{len(manifest)} gen+scored "
            f"[{(time.time() - t0) / 60:.1f}m, {(time.time() - t0) / len(per_record):.1f}s/rec]",
            flush=True,
        )

    # ---- scoring: single-topic assumed-label acc; two-topic set recall/precision ----
    metrics = {
        "arm": "raw-hidden-decode-pr",
        "model": "google/gemma-3-1b-it",
        "decoding": "greedy",
        "seed": a.seed,
        "attn": attn,
        "max_new_tokens": a.max_new_tokens,
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

    (outdir / "decode_raw.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with (outdir / "decode_raw_per_record.jsonl").open("w") as f:
        for r in per_record:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(metrics["per_arm"], indent=2))
    print(f"\nwrote {outdir / 'decode_raw.json'}\nRAW DECODE PR DONE", flush=True)


if __name__ == "__main__":
    main()
