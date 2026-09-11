#!/usr/bin/env python3
"""Score decode-transfer runs on the same proxies, so selection arms can be compared.

The decode setting has no ground-truth per-token labels: the backbone writes its own response, so
the only thing known about it is which topics the prompt asked for. The proxies below are the ones
the earlier decode reports use, computed here for any number of runs at once.

  purity        share of response tokens labelled with one of the prompt's topics
  dominant_acc  the majority response label is one of the prompt's topics
  coverage      mean fraction of the prompt's topics holding at least TAU of the tokens
  full_cover    fraction of records where every prompt topic clears TAU
  switch_rate   label changes per token; lower means the detector stays on course
  single_acc    single-topic records only: token share equal to the assumed label

Usage:
    python -m important_scripts.analysis.score_decode_transfer \\
        --run summation=ablation/gemma_arm/decode_1b_static_pr/per_record.jsonl \\
        --run f_statistic=ablation/gemma_arm/decode_1b_static_pr_fclassif/per_record.jsonl \\
        --out results/decode_transfer_selectors_1b.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

# local imports
from important_scripts.model.models import TOPICS
from important_scripts.paths import PROJECT_ROOT

TAU = 0.10  # a topic counts as covered when it holds at least this share of response tokens


def record_stats(record: dict, family: str) -> dict | None:
    """Proxy statistics for one record under one detector, or None when nothing was generated."""
    truth = [TOPICS.index(topic) for topic in record["topics"]]
    predictions = np.asarray(record["pred"][family])
    if predictions.size == 0:
        return None
    shares = {topic: float((predictions == topic).mean()) for topic in truth}
    covered = sum(share >= TAU for share in shares.values())
    majority = Counter(predictions.tolist()).most_common(1)[0][0]
    stats = {
        "n_true": len(truth),
        "purity": float(np.isin(predictions, truth).mean()),
        "dominant": majority in truth,
        "coverage": covered / len(truth),
        "full_cover": covered == len(truth),
        "switch_rate": float((predictions[1:] != predictions[:-1]).sum()) / len(predictions),
    }
    if record.get("single_topic"):
        stats["single_acc"] = float((predictions == record["assumed_label_id"]).mean())
    return stats


def score_run(path: Path) -> dict:
    """Aggregate every detector in one decode run's per-record file."""
    records = [json.loads(line) for line in path.open()]
    families = list(records[0]["pred"])
    out = {"records": len(records), "detectors": {}}
    for family in families:
        stats = [s for record in records if (s := record_stats(record, family))]
        singles = [s["single_acc"] for s in stats if "single_acc" in s]
        out["detectors"][family] = {
            "purity": float(np.mean([s["purity"] for s in stats])),
            "dominant_acc": float(np.mean([s["dominant"] for s in stats])),
            "coverage": float(np.mean([s["coverage"] for s in stats])),
            "full_cover": float(np.mean([s["full_cover"] for s in stats])),
            "switch_rate": float(np.mean([s["switch_rate"] for s in stats])),
            "single_acc": float(np.mean(singles)) if singles else None,
            "by_topic_count": {
                str(k): {
                    "n": len(group),
                    "coverage": float(np.mean([s["coverage"] for s in group])),
                    "purity": float(np.mean([s["purity"] for s in group])),
                }
                for k in (1, 2, 3, 4)
                if (group := [s for s in stats if s["n_true"] == k])
            },
        }
    return out


def main() -> None:
    """Score every named run and write one comparison json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="NAME=PATH",
                        help="a labelled per_record.jsonl; repeat for each arm")
    parser.add_argument("--out", required=True, help="where to write the comparison json")
    args = parser.parse_args()

    results = {}
    for item in args.run:
        name, _, raw = item.partition("=")
        path = Path(raw)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        results[name] = score_run(path)
        detectors = results[name]["detectors"]
        print(f"{name}: {results[name]['records']} records", flush=True)
        for family, values in detectors.items():
            print(f"    {family:12s} purity {values['purity']:.3f}  dominant {values['dominant_acc']:.3f}  "
                  f"coverage {values['coverage']:.3f}  full {values['full_cover']:.3f}", flush=True)

    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"tau": TAU, "runs": results}, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
