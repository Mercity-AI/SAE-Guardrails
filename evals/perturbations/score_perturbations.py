#!/usr/bin/env python3
"""Score the detectors on a perturbed test set, before and after the perturbation.

Every record is encoded twice through Gemma, as written (before) and perturbed (after), labelled
with the canonical prompt+response labeller, and scored by all three detectors of each feature
input (``--arm``: summation, F-statistic and group-lasso static SAE features, or dynamic SAE
features). Each kind reports per-topic F1 and IoU before and after, plus its own extra:

  insertion     off-topic asides are inserted (insert_offtopic.py). Their tokens are left unlabelled,
                so the scores measure what the asides do to the real topics. Adds confidence and
                true-class probability per topic, and where each topic's tokens go afterwards.
  confidence    the same inserted asides, scored for abstention: max-softmax confidence on topic
                tokens against aside tokens, the AUROC between the two, and a threshold sweep.
                The detectors have no "none" class, so the question is how confidently wrong they are.
  removal       one requested topic is deleted from the response. Adds the "phantom" rate: how
                often response tokens are still labelled as the deleted topic, and how confidently.
  reordering    the response's topic spans are permuted out of the prompt's order.
  substitution  one span is swapped for a same-topic span from another record.

Removal, reordering and substitution also save a per-token dump of the after pass (role, gold,
prediction, confidence and the 7-way probabilities) unless ``--no-dump`` is given.

    python evals/perturbations/score_perturbations.py --kind insertion --model-size 1b
    python evals/perturbations/score_perturbations.py --kind removal --model-size 4b --arm dynamic
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

# local imports
from common.data import seed_everything
from common.labels import label_record, strip_tags
from common.metrics import metrics
from common.paths import PROJECT_ROOT
from evals.decode.decode import BACKBONES, Featurizer, load_detector, load_saes
from models import PAD, TOPICS
from preprocessing.build_static_cache import find_backbone

DEVICE = "cuda"
RESPONSE_ROLE = 2
INPUTS = {  # the perturbed sets; insertion and confidence both read the insertion output
    "insertion": "ablation/abstention/inject_offtopic_1500.jsonl",
    "confidence": "ablation/abstention/inject_offtopic_1500.jsonl",
    "removal": "ablation/abstention/truncation_1500.jsonl",
    "reordering": "ablation/abstention/reorder_1500.jsonl",
    "substitution": "ablation/abstention/substitution_1500.jsonl",
}
ARMS = {  # arm -> size -> (cache holding the feature index, detector checkpoints)
    "summation": {
        "1b": ("cache/sae1500_10k_1b_pr", "results/gemma1b_baselines_sae150_10k_pr"),
        "4b": ("cache/sae1500_10k_gemma4b_pr", "results/gemma4b_baselines_sae150_pr"),
    },
    "fstat": {
        "1b": ("cache/sae1500_10k_1b_pr_fclassif", "results/gemma1b_sae150_10k_pr_fclassif"),
        "4b": ("cache/sae1500_10k_4b_pr_fclassif", "results/gemma4b_sae150_10k_pr_fclassif"),
    },
    "lasso": {
        "1b": ("cache/sae1500_10k_1b_pr_probe_gsparse", "results/gemma1b_sae150_10k_pr_probe_gsparse"),
        "4b": ("cache/sae1500_10k_4b_pr_probe_gsparse", "results/gemma4b_sae150_10k_pr_probe_gsparse"),
    },
    "dynamic": {
        "1b": ("cache/dyn150_sparse_10k_1b", "results/dyn150_10k_1b_pr"),
        "4b": ("cache/dyn150_sparse_10k_4b", "results/dyn150_10k_4b_pr"),
    },
}
DETECTORS = {  # run subfolder of each detector, static arms vs dynamic
    "static": {"GRU": "gru_lr3e4", "Transformer": "transformer_wide", "ConvNeXt": "convnext_window505"},
    "dynamic": {
        "GRU": "gru/gru_lr3e4",
        "Transformer": "transformer/transformer_wide",
        "ConvNeXt": "tcn/convnext",
    },
}
TAUS = [0.3, 0.5, 0.7, 0.9, 0.95, 0.99]  # confidence thresholds for the abstention sweep
ASIDE_TAG = re.compile(r"</?(?:Joke|Neutral)>")
ALL_TAGS = re.compile(r"<(/?)(" + "|".join(re.escape(t) for t in [*TOPICS, "Joke", "Neutral"]) + r")>")


# ---------------------------------------------------------------- encoding


def hidden_states(backbone, ids: list[int], layers: list[int]) -> dict[int, torch.Tensor]:
    """One teacher-forced pass; each layer's output, read as ``hidden_states[L + 1]``."""
    out = backbone(input_ids=torch.tensor([ids], device=DEVICE), use_cache=False, output_hidden_states=True)
    return {L: out.hidden_states[L + 1][0].float() for L in layers}


def aside_token_mask(tok, ids, prompt: str, response: str, asides: list[str]) -> np.ndarray:
    """Mark the tokens of ``ids`` that came from an inserted aside, aligned like label_record's offsets."""
    templated = tok.apply_chat_template(
        [
            {"role": "user", "content": strip_tags(prompt)},
            {"role": "assistant", "content": strip_tags(response)},
        ],
        tokenize=False,
    )
    enc = tok(templated, add_special_tokens=False, return_offsets_mapping=True)
    ids2, offs = enc["input_ids"], enc["offset_mapping"]
    shift = 1 if (len(ids2) == len(ids) + 1 and ids2[1:] == list(ids)) else 0
    ranges = []
    for text in asides:
        start = templated.find(text)
        while start != -1:
            ranges.append((start, start + len(text)))
            start = templated.find(text, start + 1)
    mask = np.zeros(len(ids), dtype=bool)
    for k, (s, e) in enumerate(offs):
        i = k - shift
        if 0 <= i < len(ids) and e > s and any(not (e <= a or s >= b) for a, b in ranges):
            mask[i] = True
    return mask


def encode(tok, prompt: str, response: str, asides: list[str] | None = None):
    """Canonical prompt+response (ids, roles, labels); inserted-aside tokens, if given, become PAD."""
    ids, roles, labels = label_record({"prompt": prompt, "response": response}, tok, supervise_prompt=True)
    labels = np.asarray(labels, dtype=np.int64)
    if asides:
        labels[aside_token_mask(tok, ids, prompt, response, asides)] = PAD
    return list(ids), np.asarray(roles, dtype=np.int64), labels


def strip_with_asides(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Strip every tag; return the clean text and the character ranges that were inside an aside."""
    clean, asides, stack, pos, cursor = [], [], [], 0, 0
    for m in [*ALL_TAGS.finditer(text), None]:
        seg = text[cursor : m.start()] if m else text[cursor:]
        if seg:
            clean.append(seg)
            if stack and stack[-1] in ("Joke", "Neutral"):
                asides.append((pos, pos + len(seg)))
            pos += len(seg)
        if m is None:
            break
        closing, name = m.group(1), m.group(2)
        if not closing:
            stack.append(name)
        elif stack and stack[-1] == name:
            stack.pop()
        cursor = m.end()
    return "".join(clean), asides


def encode_for_confidence(tok, record: dict):
    """The abstention run's encoding: prompt with a generation prompt, then the response tokens.

    Returns (ids, index of the first response token, aside mask, keep mask) over response tokens,
    or None for an empty response. Tokens with an empty character span are left out of both groups.
    """
    prompt, _ = strip_with_asides(record["inj_prompt"])
    response, aside_ranges = strip_with_asides(record["inj_response"])
    p_ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True
    )
    p_ids = list(p_ids[0].ids if p_ids and not isinstance(p_ids[0], int) else p_ids)
    enc = tok(response, add_special_tokens=False, return_offsets_mapping=True)
    if not enc["input_ids"]:
        return None
    offs = enc["offset_mapping"]
    keep = np.array([e > s for s, e in offs], dtype=bool)
    aside = np.array(
        [e > s and any(not (e <= a or s >= b) for a, b in aside_ranges) for s, e in offs], dtype=bool
    )
    return p_ids + list(enc["input_ids"]), len(p_ids), aside, keep


# ---------------------------------------------------------------- summaries


def joined(values: list) -> np.ndarray:
    """Concatenate per-record arrays, or an empty array when there are none."""
    return np.concatenate(values) if values else np.array([])


def percentiles(values: np.ndarray) -> dict:
    """Count, mean, 5th / 50th / 95th percentile and share at or above 0.9."""
    if not len(values):
        return {}
    p5, p50, p95 = np.percentile(values, [5, 50, 95])
    return {
        "n": int(len(values)),
        "mean": round(float(values.mean()), 4),
        "p5": round(float(p5), 4),
        "p50": round(float(p50), 4),
        "p95": round(float(p95), 4),
        "frac_ge_0.9": round(float((values >= 0.9).mean()), 4),
    }


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score(pos) > score(neg)) from tie-averaged ranks (Mann-Whitney U)."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    _, inv, counts = np.unique(np.concatenate([pos, neg]), return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    ranks = ((csum - counts + 1 + csum) / 2.0)[inv]
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def abstention(conf_topic: np.ndarray, conf_aside: np.ndarray, pred_aside: np.ndarray) -> dict:
    """Confidence on topic vs aside tokens, their AUROC, a threshold sweep and where asides are sent."""
    out = {
        "n_topic_tokens": int(len(conf_topic)),
        "n_offtopic_tokens": int(len(conf_aside)),
        "mean_conf_topic": float(conf_topic.mean()) if len(conf_topic) else None,
        "mean_conf_offtopic": float(conf_aside.mean()) if len(conf_aside) else None,
        "median_conf_topic": float(np.median(conf_topic)) if len(conf_topic) else None,
        "median_conf_offtopic": float(np.median(conf_aside)) if len(conf_aside) else None,
        "auroc_topic_vs_offtopic": auroc(conf_topic, conf_aside),
        "tau_sweep": {
            f"{t}": {
                "offtopic_confident_frac": float((conf_aside >= t).mean()) if len(conf_aside) else None,
                "topic_confident_frac": float((conf_topic >= t).mean()) if len(conf_topic) else None,
            }
            for t in TAUS
        },
    }
    if len(pred_aside):
        counts = np.bincount(pred_aside, minlength=len(TOPICS))
        out["offtopic_predicted_topic_mix"] = {TOPICS[i]: int(c) for i, c in enumerate(counts) if c}
    return out


def before_after(rows: dict) -> dict:
    """Per-topic and macro F1 / IoU before and after, from the metrics() of each phase."""
    mb, ma = metrics(rows["before"]), metrics(rows["after"])
    per = {}
    for t in TOPICS:
        b, a = mb["per_topic"][t], ma["per_topic"][t]
        iou_b, iou_a = mb["topic_overlap"]["per_topic"][t]["iou"], ma["topic_overlap"]["per_topic"][t]["iou"]
        per[t] = {
            "support_before": int(b["support"]),
            "support_after": int(a["support"]),
            "precision_before": round(b["precision"], 4),
            "recall_before": round(b["recall"], 4),
            "f1_before": round(b["f1-score"], 4),
            "precision_after": round(a["precision"], 4),
            "recall_after": round(a["recall"], 4),
            "f1_after": round(a["f1-score"], 4),
            "f1_delta": round(a["f1-score"] - b["f1-score"], 4),
            "iou_before": round(iou_b, 4),
            "iou_after": round(iou_a, 4),
            "iou_delta": round(iou_a - iou_b, 4),
        }
    return {
        "per_topic": per,
        "macro_f1_before": round(mb["macro_f1"], 4),
        "macro_f1_after": round(ma["macro_f1"], 4),
        "macro_iou_before": round(mb["topic_overlap"]["macro_iou"], 4),
        "macro_iou_after": round(ma["topic_overlap"]["macro_iou"], 4),
    }


# ---------------------------------------------------------------- scoring one arm


@torch.inference_mode()
def score_arm(kind, tok, backbone, featurize, detectors, records, layers, dump) -> dict:
    """Run every record through the detectors of one arm and summarise the kind's measures."""
    n_topics = len(TOPICS)
    if kind == "confidence":
        acc = {d: {"topic": [], "aside": [], "pred": []} for d in detectors}
        for n, rec in enumerate(records, 1):
            encoded = encode_for_confidence(tok, rec)
            if encoded is None:
                continue
            ids, r0, aside, keep = encoded
            x = featurize(hidden_states(backbone, ids, layers))[None]
            for d, det in detectors.items():
                conf, pred = torch.softmax(det(x)[0].float(), dim=-1).max(dim=-1)
                conf, pred = conf[r0:].cpu().numpy(), pred[r0:].cpu().numpy()
                acc[d]["topic"].append(conf[~aside & keep])
                acc[d]["aside"].append(conf[aside & keep])
                acc[d]["pred"].append(pred[aside & keep])
            if n % 200 == 0:
                print(f"    {n}/{len(records)}", flush=True)
        return {
            d: abstention(joined(a["topic"]), joined(a["aside"]), joined(a["pred"]).astype(np.int64))
            for d, a in acc.items()
        }

    rows = {d: {"before": [], "after": []} for d in detectors}
    conf_by_topic = {
        d: {p: {t: {"conf": [], "ptrue": []} for t in range(n_topics)} for p in rows[d]} for d in detectors
    }
    confusion = {d: np.zeros((n_topics, n_topics), dtype=np.int64) for d in detectors}
    phantom = {d: {t: {"resp": 0, "hits": 0, "conf": []} for t in range(n_topics)} for d in detectors}
    for n, rec in enumerate(records, 1):
        asides = [i["text"] for i in rec.get("injections", [])]
        after = (
            (ASIDE_TAG.sub("", rec["inj_prompt"]), ASIDE_TAG.sub("", rec["inj_response"]), asides)
            if kind == "insertion"
            else (rec["inj_prompt"], rec["inj_response"], None)
        )
        for phase, (prompt, response, masked) in (
            ("before", (rec["orig_prompt"], rec["orig_response"], None)),
            ("after", after),
        ):
            ids, roles, labels = encode(tok, prompt, response, masked)
            x = featurize(hidden_states(backbone, ids, layers))[None]
            real = np.flatnonzero(labels != PAD)
            for d, det in detectors.items():
                probs = torch.softmax(det(x)[0].float(), dim=-1)
                conf, pred = (t.cpu().numpy() for t in probs.max(dim=-1))
                rows[d][phase].append({"labels": labels, "predictions": pred})
                if kind == "insertion" and real.size:
                    gold = labels[real]
                    if phase == "after":
                        np.add.at(confusion[d], (gold, pred[real]), 1)
                    p_true = probs.cpu().numpy()[real, gold]
                    for t in np.unique(gold):
                        conf_by_topic[d][phase][t]["conf"].append(conf[real][gold == t])
                        conf_by_topic[d][phase][t]["ptrue"].append(p_true[gold == t])
                if phase == "after" and kind == "removal":
                    resp = roles == RESPONSE_ROLE
                    ph = phantom[d][rec["dropped_topic_id"]]
                    hit = pred[resp] == rec["dropped_topic_id"]
                    ph["resp"] += int(resp.sum())
                    ph["hits"] += int(hit.sum())
                    ph["conf"].append(conf[resp][hit])
                if phase == "after" and dump is not None:
                    for key, value in (
                        ("rec", np.full(len(labels), rec["index"], np.int32)),
                        ("pos", np.arange(len(labels), dtype=np.int32)),
                        ("role", roles.astype(np.int8)),
                        ("gold", labels.astype(np.int16)),
                        ("pred", pred.astype(np.int8)),
                        ("conf", conf.astype(np.float16)),
                        ("probs", probs.cpu().numpy().astype(np.float16)),
                    ):
                        dump[d][key].append(value)
        if n % 200 == 0:
            print(f"    {n}/{len(records)}", flush=True)

    out = {}
    for d in detectors:
        out[d] = before_after(rows[d])
        if kind == "insertion":
            for t, name in enumerate(TOPICS):
                topic = out[d]["per_topic"][name]
                for phase in ("before", "after"):
                    topic[f"confidence_{phase}"] = percentiles(joined(conf_by_topic[d][phase][t]["conf"]))
                    topic[f"true_class_prob_{phase}"] = percentiles(
                        joined(conf_by_topic[d][phase][t]["ptrue"])
                    )
                topic["confusion_after"] = {TOPICS[j]: int(c) for j, c in enumerate(confusion[d][t]) if c}
        if kind == "removal":
            dropped = Counter(TOPICS[r["dropped_topic_id"]] for r in records)
            all_conf = joined([c for p in phantom[d].values() for c in p["conf"]])
            resp = sum(p["resp"] for p in phantom[d].values())
            out[d]["phantom_by_dropped_topic"] = {
                name: {
                    "records_dropped": dropped.get(name, 0),
                    "resp_tokens": phantom[d][t]["resp"],
                    "phantom_tokens": phantom[d][t]["hits"],
                    "phantom_rate": round(phantom[d][t]["hits"] / phantom[d][t]["resp"], 4)
                    if phantom[d][t]["resp"]
                    else None,
                    "phantom_conf": percentiles(joined(phantom[d][t]["conf"])),
                }
                for t, name in enumerate(TOPICS)
            }
            out[d]["phantom_rate_overall"] = (
                round(sum(p["hits"] for p in phantom[d].values()) / resp, 4) if resp else None
            )
            out[d]["phantom_conf_overall"] = percentiles(all_conf)
    return out


def main() -> None:
    """Score each requested arm on one perturbed set and write one json (and dumps) per arm."""
    parser = argparse.ArgumentParser(description="Score the detectors on a perturbed test set.")
    parser.add_argument("--kind", choices=sorted(INPUTS), required=True)
    parser.add_argument("--model-size", choices=sorted(BACKBONES), required=True)
    parser.add_argument("--arm", choices=[*ARMS, "all"], default="all")
    parser.add_argument("--input", default=None, help="perturbed jsonl (default: the recorded run's path)")
    parser.add_argument("--out", default=None, help="output directory")
    parser.add_argument("--no-dump", action="store_true", help="skip the per-token dump")
    parser.add_argument("--limit", type=int, default=0, help="only the first N records")
    args = parser.parse_args()

    def resolve(path: str) -> Path:
        return Path(path) if Path(path).is_absolute() else PROJECT_ROOT / path

    model_id, release, layers = BACKBONES[args.model_size]
    arms = list(ARMS) if args.arm == "all" else [args.arm]
    out_dir = resolve(args.out or f"ablation/perturbations/{args.model_size}")
    out_dir.mkdir(parents=True, exist_ok=True)
    latest = {}
    for line in resolve(args.input or INPUTS[args.kind]).open(encoding="utf-8"):
        record = json.loads(line)
        latest[record["index"]] = record  # the insertion run is resumable; keep the last row per record
    records = [r for r in latest.values() if r.get("_status") == "ok"][: args.limit or None]
    print(f"{args.kind} | {model_id} | arms {arms} | {len(records)} records -> {out_dir}", flush=True)

    seed_everything(42, deterministic=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    tok = AutoTokenizer.from_pretrained(model_id)
    # The abstention run used eager attention; the others the cache builder's FlashAttention.
    attention = "eager" if args.kind == "confidence" else "flash_attention_2"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, attn_implementation=attention
        )
    except (ImportError, ValueError) as e:
        print(f"{attention} unavailable ({e}); using eager", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, attn_implementation="eager"
        )
    backbone = find_backbone(model.to(DEVICE).eval())
    saes = load_saes(release, layers)

    for arm in arms:
        cache, ckpt = (PROJECT_ROOT / p for p in ARMS[arm][args.model_size])
        features = "dynamic" if arm == "dynamic" else "static"
        featurize = Featurizer(features, release, layers, cache, saes=saes)
        detectors = {
            d: load_detector(ckpt / sub / "checkpoint_best.pt") for d, sub in DETECTORS[features].items()
        }
        dump = None
        if args.kind in ("removal", "reordering", "substitution") and not args.no_dump:
            dump = {
                d: {k: [] for k in ("rec", "pos", "role", "gold", "pred", "conf", "probs")} for d in detectors
            }
        started = time.time()
        result = score_arm(args.kind, tok, backbone, featurize, detectors, records, layers, dump)
        report = {"kind": args.kind, "model_size": args.model_size, "arm": arm, "records": len(records)}
        (out_dir / f"{args.kind}_{arm}.json").write_text(json.dumps(report | {"detectors": result}, indent=2))
        if dump is not None:
            for d, columns in dump.items():
                arrays = {k: np.concatenate(v) for k, v in columns.items()}
                np.savez_compressed(out_dir / f"{args.kind}_{arm}_{d}_pertoken.npz", **arrays)
        print(f"  {arm} done in {time.time() - started:.0f}s -> {args.kind}_{arm}.json", flush=True)
        for d, r in result.items():
            if args.kind == "confidence":
                print(f"    {d:11s} AUROC {r['auroc_topic_vs_offtopic']:.3f}", flush=True)
            else:
                print(
                    f"    {d:11s} macro-F1 {r['macro_f1_before']:.4f} -> {r['macro_f1_after']:.4f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
