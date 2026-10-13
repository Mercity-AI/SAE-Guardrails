#!/usr/bin/env python3
"""Decode transfer: Gemma writes its own response and the prefill-trained detectors label it.

For every manifest prompt, Gemma 3 generates a greedy response. Prompt plus response then go back
through Gemma once, the ten mid-stack layers are turned into the detector input (``--features``:
static SAE columns, each token's dynamic top-K SAE features, or raw hidden states), and the GRU,
Transformer and ConvNeXt label the response tokens. Generated text has no per-token labels, so a
single-topic record is scored against the prompt's topic and a multi-topic record by which topics
appear at all; ``score_decode.py`` computes the full set of proxies from the per-record file.

``--teacher`` skips generation and teacher-forces GPT's reference response instead, which carries
real per-token labels. That gives the same detectors' score on in-distribution text, the ceiling
the decode numbers are compared against.

    python evals/decode/decode.py --model-size 1b --features static
    python evals/decode/decode.py --model-size 4b --features raw --teacher

Writes ``per_record.jsonl`` (texts and response-token predictions) and ``summary.json``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

# local imports
from common.data import seed_everything
from common.labels import label_record_pr
from common.paths import PROJECT_ROOT
from models import PAD, TOPICS, build_model
from preprocessing.build_static_cache import find_backbone, jumprelu

DEVICE = "cuda"
MANIFEST = "ablation/gemma_arm/manifest_4b_10k_test.jsonl"  # 1,000 test prompts, build_test_manifest.py
DATASET = "runs/GPT_5.6_10k_2108/dataset.final.jsonl"  # GPT reference responses, by manifest index
BACKBONES = {  # model, SAE release, the ten layers the detectors read
    "1b": ("google/gemma-3-1b-it", "gemma-scope-2-1b-it-res-all", list(range(16, 26))),
    "4b": ("google/gemma-3-4b-it", "gemma-scope-2-4b-it-res-all", list(range(24, 34))),
}
# Where each input's cache and detectors live, and the settings of the recorded runs
# (all of them generated up to 8,000 new tokens; batch sizes are the ones recorded).
FEATURES = {
    "static": {
        "cache": {"1b": "cache/sae1500_10k_1b_pr", "4b": "cache/sae1500_10k_gemma4b_pr"},
        "ckpt_dir": {
            "1b": "results/gemma1b_baselines_sae150_10k_pr",
            "4b": "results/gemma4b_baselines_sae150_pr",
        },
        "detectors": {
            "GRU": "gru_lr3e4",
            "Transformer": "transformer_wide",
            "ConvNeXt": "convnext_window505",
        },
        "attention": "flash_attention_2",
        "batch_size": {"1b": 48, "4b": 64},
        "extract_batch": {"1b": 16, "4b": 16},
    },
    "dynamic": {
        "cache": {"1b": "cache/dyn150_sparse_10k_1b", "4b": "cache/dyn150_sparse_10k_4b"},
        "ckpt_dir": {"1b": "results/dyn150_10k_1b_pr", "4b": "results/dyn150_10k_4b_pr"},
        "detectors": {
            "GRU": "gru/gru_lr3e4",
            "Transformer": "transformer/transformer_wide",
            "ConvNeXt": "tcn/convnext",
        },
        "attention": "sdpa",
        "batch_size": {"1b": 16, "4b": 48},
        "extract_batch": {"1b": 1, "4b": 1},
    },
    "raw": {
        "cache": {"1b": None, "4b": None},
        "ckpt_dir": {
            "1b": "ablation/nosae_stream_1b_10k_pr/raw",
            "4b": "ablation/nosae_stream_4b_10k_pr/raw",
        },
        "detectors": {"GRU": "gru", "Transformer": "transformer", "ConvNeXt": "convnext"},
        "attention": "flash_attention_2",
        "batch_size": {"1b": 32, "4b": 64},
        "extract_batch": {"1b": 1, "4b": 8},
    },
}
TAG_RE = re.compile(r"</?(?:" + "|".join(re.escape(t) for t in TOPICS) + r")>")
RESPONSE_ROLE = 2


def resolve(path: str) -> Path:
    """Resolve a path against the data root unless it is already absolute."""
    return Path(path) if Path(path).is_absolute() else PROJECT_ROOT / path


def strip_tags(text: str) -> str:
    """Remove every <Topic>/</Topic> marker."""
    return TAG_RE.sub("", text)


def chat_ids(tokenizer, prompt: str, response: str | None = None) -> list[int]:
    """Templated ids for a tag-stripped prompt, either ready to generate or with a response appended."""
    messages = [{"role": "user", "content": strip_tags(prompt)}]
    if response is not None:
        messages.append({"role": "assistant", "content": strip_tags(response)})
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=response is None)
    return list(ids[0].ids if ids and not isinstance(ids[0], int) else ids)


def load_detector(path: Path, device: str = DEVICE) -> torch.nn.Module:
    """Rebuild one detector from its checkpoint onto the device."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["architecture"], checkpoint["model"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def load_saes(release: str, layers: list[int]) -> dict[int, tuple]:
    """Full Gemma Scope SAEs per layer, as the (W_enc, b_enc, b_dec, threshold) that jumprelu takes."""
    saes = {}
    for L in layers:
        sae = SAE.from_pretrained(release=release, sae_id=f"layer_{L}_width_16k_l0_small", device=DEVICE)
        sae = sae[0] if isinstance(sae, tuple) else sae
        saes[L] = tuple(p.float() for p in (sae.W_enc, sae.b_enc, sae.b_dec, sae.threshold))
    return saes


class Featurizer:
    """Turns one sequence's per-layer hidden states into the input the chosen detectors were trained on."""

    def __init__(
        self, features: str, release: str, layers: list[int], cache: Path | None, saes: dict | None = None
    ):
        """Load the SAEs (unless given) and the cache's feature index; raw hidden states need neither."""
        self.features, self.layers = features, layers
        if features == "raw":
            return
        saes = saes or load_saes(release, layers)
        selected = np.load(cache / "selected_features.npz")
        columns = {
            L: torch.as_tensor(selected[f"layer_{L}"], device=DEVICE, dtype=torch.long) for L in layers
        }
        if features == "static":
            # keep only the selected columns of each SAE, as the cache builder did
            self.saes = {
                L: (W[:, columns[L]], b_enc[columns[L]], b_dec, thr[columns[L]])
                for L, (W, b_enc, b_dec, thr) in saes.items()
            }
            return
        # dynamic: per layer, a lookup from SAE feature id to its fixed training-vocabulary column
        self.saes, self.k = saes, int(json.loads((cache / "meta.json").read_text())["K"])
        self.lookups, self.offsets, offset = {}, {}, 0
        for L in layers:
            lookup = torch.full((saes[L][0].shape[1],), -1, dtype=torch.long, device=DEVICE)
            lookup[columns[L]] = torch.arange(len(columns[L]), device=DEVICE)
            self.lookups[L], self.offsets[L] = lookup, offset
            offset += len(columns[L])
        self.width = offset

    def __call__(self, hidden: dict[int, torch.Tensor]) -> torch.Tensor:
        """Hidden states (layer -> (T, d) fp32) to the (T, width) detector input."""
        if self.features == "raw":
            return torch.cat([hidden[L] for L in self.layers], dim=-1)
        if self.features == "static":
            return torch.cat([jumprelu(hidden[L], *self.saes[L]) for L in self.layers], dim=-1)
        tokens = hidden[self.layers[0]].shape[0]
        x = torch.zeros(tokens, self.width, device=DEVICE)
        rows = torch.arange(tokens, device=DEVICE)[:, None].expand(-1, self.k)
        for L in self.layers:
            values, ids = torch.topk(jumprelu(hidden[L], *self.saes[L]), self.k, dim=-1)
            column = self.lookups[L][ids]
            keep = (values > 0) & (column >= 0)
            x[rows[keep], self.offsets[L] + column[keep]] = values[keep]
        return x


@torch.inference_mode()
def layer_outputs(
    backbone, sequences: list[list[int]], pad_id: int, layers: list[int], hooked: dict
) -> list[dict]:
    """One right-padded forward pass; per sequence, each layer's output trimmed to its length.

    Generated responses are read from ``hidden_states[L + 1]``, as the recorded decode runs did.
    The teacher-forced path reads the forward hooks instead, as the recorded GPT reference run and
    the cache builders did; ``hooked`` is the hook buffer, or None for the generated path.
    """
    width = max(len(s) for s in sequences)
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), width), dtype=torch.long)
    for r, s in enumerate(sequences):
        ids[r, : len(s)], mask[r, : len(s)] = torch.tensor(s), 1
    if hooked is not None:
        hooked.clear()
    out = backbone(
        input_ids=ids.to(DEVICE),
        attention_mask=mask.to(DEVICE),
        use_cache=False,
        output_hidden_states=hooked is None,
        return_dict=True,
    )
    source = (
        {L: hooked[L] for L in layers}
        if hooked is not None
        else {L: out.hidden_states[L + 1] for L in layers}
    )
    return [{L: source[L][r, : len(s)].float() for L in layers} for r, s in enumerate(sequences)]


def generate(model, tokenizer, prompts: list[list[int]], max_new_tokens: int) -> list[list[int]]:
    """Greedy, left-padded batch generation; returns each response's ids up to its first end token."""
    pad_id = tokenizer.pad_token_id
    width = max(len(p) for p in prompts)
    ids = torch.full((len(prompts), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(prompts), width), dtype=torch.long)
    for r, p in enumerate(prompts):
        ids[r, width - len(p) :], mask[r, width - len(p) :] = torch.tensor(p), 1
    with torch.inference_mode():
        out = model.generate(
            ids.to(DEVICE),
            attention_mask=mask.to(DEVICE),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
            use_cache=True,
        )
    stop = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<end_of_turn>")}
    responses = []
    for row in out[:, width:].tolist():
        cut = next((j for j, t in enumerate(row) if t in stop), len(row))
        responses.append(row[:cut])
    return responses


def summarize(records: list[dict], families: list[str], teacher: bool) -> dict:
    """Headline decode scores per detector.

    Single-topic accuracy against the prompt's topic, multi-topic set recall and precision, and
    (teacher text only) token accuracy and macro-IoU against the true labels.
    """
    summary = {}
    for name in families:
        single, recall, precision, truth_all, pred_all = [], [], [], [], []
        for r in records:
            pred = np.asarray(r["pred"].get(name, []))
            if teacher and pred.size:  # teacher text: score only the tokens that carry a topic label
                true = np.asarray(r["true_labels"])
                pred, true = pred[true != PAD], true[true != PAD]
                truth_all.append(true)
                pred_all.append(pred)
            if pred.size == 0:
                continue
            if r["single_topic"]:
                single.append(float((pred == r["assumed_label_id"]).mean()))
            else:
                truth, seen = {TOPICS.index(t) for t in r["topics"]}, {int(v) for v in pred}
                recall.append(len(seen & truth) / len(truth))
                precision.append(len(seen & truth) / max(1, len(seen)))
        row = {
            "single_assumed_acc_mean": float(np.mean(single)) if single else None,
            "single_assumed_acc_median": float(np.median(single)) if single else None,
            "two_set_recall_mean": float(np.mean(recall)) if recall else None,
            "two_set_prec_mean": float(np.mean(precision)) if precision else None,
        }
        if teacher and truth_all:
            T, P = np.concatenate(truth_all), np.concatenate(pred_all)
            ious = [((T == c) & (P == c)).sum() / ((T == c) | (P == c)).sum() for c in np.unique(T)]
            row |= {
                "true_label_token_acc": float((T == P).mean()),
                "true_label_macro_iou": float(np.mean(ious)),
            }
        summary[name] = row
    return summary


def main() -> None:
    """Generate (or teacher-force) every manifest response, label it with each detector, and save."""
    parser = argparse.ArgumentParser(description="Decode transfer of the prefill-trained detectors.")
    parser.add_argument("--model-size", choices=sorted(BACKBONES), required=True)
    parser.add_argument("--features", choices=sorted(FEATURES), required=True)
    parser.add_argument(
        "--teacher", action="store_true", help="teacher-force GPT's response instead of generating"
    )
    parser.add_argument("--manifest", default=MANIFEST, help="prompts to decode (build_test_manifest.py)")
    parser.add_argument("--dataset", default=DATASET, help="dataset.final.jsonl, for the GPT responses")
    parser.add_argument("--cache", default=None, help="feature cache holding the SAE feature index")
    parser.add_argument("--ckpt-dir", default=None, help="detector checkpoints (e.g. a group-lasso run)")
    parser.add_argument("--out", default=None, help="output directory")
    parser.add_argument("--max-new-tokens", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=None, help="generation batch size")
    parser.add_argument("--extract-batch", type=int, default=None, help="sequences per feature pass")
    parser.add_argument("--attention", default=None, help="Gemma attention implementation")
    parser.add_argument("--limit", type=int, default=0, help="only the first N prompts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    spec, size = FEATURES[args.features], args.model_size
    model_id, release, layers = BACKBONES[size]
    cache = resolve(args.cache or spec["cache"][size]) if spec["cache"][size] or args.cache else None
    ckpt_dir = resolve(args.ckpt_dir or spec["ckpt_dir"][size])
    mode = "teacher" if args.teacher else "decode"
    out_dir = resolve(args.out or f"ablation/decode/{size}_{args.features}_{mode}")
    attention = args.attention or spec["attention"]
    batch_size = args.batch_size or spec["batch_size"][size]
    extract_batch = args.extract_batch or spec["extract_batch"][size]
    if cache is not None and (cache / "metadata.json").exists():
        cached_layers = [int(L) for L in json.loads((cache / "metadata.json").read_text())["layers"]]
        assert cached_layers == layers, f"{cache} holds layers {cached_layers}, expected {layers}"

    seed_everything(args.seed, deterministic=args.features != "dynamic")
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation=attention
    )
    model = model.to(DEVICE).eval()
    backbone = find_backbone(model)
    hooked = None
    if args.teacher:
        hooked = {}
        for L in layers:
            backbone.layers[L].register_forward_hook(
                lambda _m, _i, out, L=L: hooked.__setitem__(L, out[0] if isinstance(out, tuple) else out)
            )
    featurize = Featurizer(args.features, release, layers, cache)
    detectors = {
        name: load_detector(ckpt_dir / sub / "checkpoint_best.pt") for name, sub in spec["detectors"].items()
    }
    special = set(tokenizer.all_special_ids) | {tokenizer.convert_tokens_to_ids("<end_of_turn>")}

    manifest = [json.loads(line) for line in resolve(args.manifest).open()][: args.limit or None]
    dataset = [json.loads(line)["record"] for line in resolve(args.dataset).open()]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{model_id} | {args.features} | {mode} | {len(manifest)} prompts -> {out_dir}", flush=True)

    records, started = [], time.time()
    for start in range(0, len(manifest), batch_size):
        rows = manifest[start : start + batch_size]
        items = []  # (full ids, positions the detectors are scored on, record)
        if args.teacher:
            for row in rows:
                gpt = dataset[row["index"]]
                ids, roles, labels = label_record_pr(gpt, tokenizer)
                positions = [i for i, role in enumerate(roles) if role == RESPONSE_ROLE]
                record = {"true_labels": [int(labels[i]) for i in positions]}
                items.append((list(ids), positions, record))
        else:
            prompts = [chat_ids(tokenizer, row["prompt"]) for row in rows]
            for prompt, response in zip(prompts, generate(model, tokenizer, prompts, args.max_new_tokens)):
                positions = [len(prompt) + k for k, t in enumerate(response) if t not in special]
                record = {
                    "generated_tokens": len(response),
                    "gemma_response": tokenizer.decode(response, skip_special_tokens=True),
                }
                items.append((prompt + response, positions, record))

        for row, (_, positions, record) in zip(rows, items):
            record |= {
                "index": row["index"],
                "topics": row["topics"],
                "single_topic": row["single_topic"],
                "assumed_label_id": row["assumed_label_id"],
                "response_content_tokens": len(positions),
                "prompt": strip_tags(row["prompt"]),
                "gpt_response": strip_tags(dataset[row["index"]]["response"]),
                "pred": {},
            }
        for s in range(0, len(items), extract_batch):
            group = [item for item in items[s : s + extract_batch] if item[1]]
            if not group:
                continue
            hidden = layer_outputs(
                backbone, [ids for ids, _, _ in group], tokenizer.pad_token_id, layers, hooked
            )
            with torch.inference_mode():
                for (_, positions, record), h in zip(group, hidden):
                    x = featurize(h)[None]
                    for name, detector in detectors.items():
                        record["pred"][name] = detector(x)[0].argmax(-1)[positions].cpu().tolist()
        records += [record for _, _, record in items]
        print(f"  {len(records)}/{len(manifest)} ({time.time() - started:.0f}s)", flush=True)

    summary = {
        "model": model_id,
        "features": args.features,
        "mode": mode,
        "attention": attention,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": batch_size,
        "seed": args.seed,
        "manifest": str(resolve(args.manifest)),
        "cache": str(cache) if cache else None,
        "ckpt_dir": str(ckpt_dir),
        "records": len(records),
        "per_detector": summarize(records, list(detectors), args.teacher),
    }
    with (out_dir / "per_record.jsonl").open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["per_detector"], indent=2))
    print(f"wrote {out_dir / 'per_record.jsonl'} and summary.json")


if __name__ == "__main__":
    main()
