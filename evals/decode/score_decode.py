#!/usr/bin/env python3
r"""Score decode runs on the prompt-topic proxies, so feature inputs and backbones can be compared.

Generated text has no per-token labels: the only thing known about it is which topics the prompt
asked for. Every proxy below therefore compares a detector's response-token labels with the
prompt's topics. Run it on a ``decode.py --teacher`` output to get the same proxies on GPT's text,
the prefill side of the prefill-to-decode comparison.

  purity        share of response tokens labelled with one of the prompt's topics
  dominant_acc  the most frequent response label is one of the prompt's topics
  coverage      mean fraction of the prompt's topics holding at least TAU of the tokens
  full_cover    fraction of records where every prompt topic clears TAU
  switch_rate   label changes per token; lower means the detector stays on course
  segments      contiguous runs of at least MIN_RUN tokens with one label
  single_acc    single-topic records: token share equal to the prompt's topic
  set_recall    multi-topic records: share of the prompt's topics that appear at all
  set_precision multi-topic records: share of the labels that appear that the prompt asked for
  hit_cap       generated runs only: share of responses that reached the token cap
  gen_coverage  teacher runs only: coverage computed on GPT's true labels, i.e. whether the
                writer covered each topic, which separates "the detector missed it" from
                "the text never had it"

    python evals/decode/score_decode.py --out results/decode_1b.json \
        --run static=ablation/decode/1b_static_decode/per_record.jsonl \
        --run static_gpt=ablation/decode/1b_static_teacher/per_record.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

# local imports
from common.paths import PROJECT_ROOT
from models import PAD, TOPICS

TAU = 0.10  # a topic counts as covered when it holds at least this share of response tokens
MIN_RUN = 20  # a run of at least this many tokens with one label counts as a segment
METRICS = (
    "purity",
    "dominant_acc",
    "coverage",
    "full_cover",
    "switch_rate",
    "segments",
    "single_acc",
    "set_recall",
    "set_precision",
    "hit_cap",
    "gen_coverage",
)


def covered(labels: np.ndarray, truth: list[int]) -> int:
    """How many of the prompt's topics hold at least TAU of these labels."""
    return sum(bool(labels.size) and float((labels == t).mean()) >= TAU for t in truth)


def record_stats(record: dict, family: str, cap: int | None) -> dict | None:
    """Proxy statistics for one record under one detector, or None when there is nothing to score."""
    truth = [TOPICS.index(topic) for topic in record["topics"]]
    pred = np.asarray(record["pred"].get(family, []))
    if pred.size == 0:
        return None
    runs = np.diff(np.flatnonzero(np.diff(np.r_[-1, pred, -1]) != 0))  # lengths of constant-label runs
    n_covered = covered(pred, truth)
    stats = {
        "n_true": len(truth),
        "purity": float(np.isin(pred, truth).mean()),
        "dominant_acc": float(Counter(pred.tolist()).most_common(1)[0][0] in truth),
        "coverage": n_covered / len(truth),
        "full_cover": float(n_covered == len(truth)),
        "switch_rate": float((pred[1:] != pred[:-1]).sum()) / pred.size,
        "segments": float((runs >= MIN_RUN).sum()),
    }
    if record.get("single_topic"):
        stats["single_acc"] = float((pred == record["assumed_label_id"]).mean())
    else:
        seen = set(pred.tolist())
        stats["set_recall"] = len(seen & set(truth)) / len(truth)
        stats["set_precision"] = len(seen & set(truth)) / max(1, len(seen))
    if cap is not None and "generated_tokens" in record:
        stats["hit_cap"] = float(record["generated_tokens"] >= cap)
    if "true_labels" in record:
        gold = np.asarray(record["true_labels"])
        stats["gen_coverage"] = covered(gold[gold != PAD], truth) / len(truth)
    return stats


def aggregate(stats: list[dict]) -> dict:
    """Mean of every metric present in at least one record (None when no record has it)."""
    out = {"n": len(stats)}
    for metric in METRICS:
        values = [s[metric] for s in stats if metric in s]
        out[metric] = float(np.mean(values)) if values else None
    return out


def score_run(path: Path) -> dict:
    """Aggregate every detector in one decode run, overall and by the prompt's topic count."""
    records = [json.loads(line) for line in path.open()]
    summary = path.parent / "summary.json"
    cap = json.loads(summary.read_text()).get("max_new_tokens") if summary.exists() else None
    if cap is None and any("generated_tokens" in r for r in records):  # older runs: infer the cap
        cap = max(r.get("generated_tokens", 0) for r in records)
    families = sorted({family for r in records for family in r["pred"]})
    out = {"records": len(records), "detectors": {}}
    for family in families:
        stats = [s for r in records if (s := record_stats(r, family, cap))]
        out["detectors"][family] = aggregate(stats) | {
            "by_topic_count": {
                str(k): aggregate(group)
                for k in (1, 2, 3, 4)
                if (group := [s for s in stats if s["n_true"] == k])
            }
        }
    return out


def main() -> None:
    """Score every named run and write one comparison json."""
    parser = argparse.ArgumentParser(description="Score decode runs on the prompt-topic proxies.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="a per_record.jsonl from decode.py; repeat for each run",
    )
    parser.add_argument("--out", required=True, help="where to write the comparison json")
    args = parser.parse_args()

    results = {}
    for item in args.run:
        name, _, raw = item.partition("=")
        path = Path(raw) if Path(raw).is_absolute() else PROJECT_ROOT / raw
        results[name] = score_run(path)
        print(f"{name}: {results[name]['records']} records", flush=True)
        for family, v in results[name]["detectors"].items():
            print(
                f"    {family:12s} purity {v['purity']:.3f}  dominant {v['dominant_acc']:.3f}  "
                f"coverage {v['coverage']:.3f}  full {v['full_cover']:.3f}",
                flush=True,
            )

    out = Path(args.out) if Path(args.out).is_absolute() else PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"tau": TAU, "min_run": MIN_RUN, "runs": results}, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
