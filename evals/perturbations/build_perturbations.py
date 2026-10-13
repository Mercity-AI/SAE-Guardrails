#!/usr/bin/env python3
"""Build the removal, reordering and substitution test sets from the insertion run's originals.

Each perturbation changes only the response; the prompt is left as it was. Every set is built
from the tagged originals stored in the insertion output, so all four perturbations score the same
test records, and the perturbed text is re-encoded from scratch by ``score_perturbations.py``.

  removal       one topic's spans are deleted from the response, so a topic the prompt asks for
                is absent. Records need at least two response topics and, by default, the same
                topic set in prompt and response.
  reordering    the response's topic spans are permuted so that none stays in its prompt position.
  substitution  one topic's span is swapped for a span of the same topic from another test record.

Reordering and substitution use the same eligibility: prompt and response hold the same set of
distinct topics, each once, at least two. Every random choice is seeded per record.

    python evals/perturbations/build_perturbations.py --kind removal
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

# local imports
from common.labels import span_char_ranges, strip_tags, validate_topic_tags
from common.paths import PROJECT_ROOT
from models import TOPICS

SOURCE = (
    "ablation/abstention/inject_offtopic_1500.jsonl"  # insert_offtopic.py output; orig_* are the originals
)
OUTPUTS = {
    "removal": "ablation/abstention/truncation_1500.jsonl",
    "reordering": "ablation/abstention/reorder_1500.jsonl",
    "substitution": "ablation/abstention/substitution_1500.jsonl",
}
_OPEN = re.compile(r"<(" + "|".join(re.escape(t) for t in TOPICS) + r")>")
_EVENT = re.compile(r"</?(?:" + "|".join(re.escape(t) for t in TOPICS) + r")>|[^<]+|<")


def ordered_spans(tagged: str) -> list[tuple[int, str]]:
    """[(topic id, span text)] in order of appearance."""
    stripped = strip_tags(tagged)
    return [(t, stripped[a:b].strip()) for a, b, t in span_char_ranges(tagged)]


def topic_set(tagged: str) -> set[int]:
    """Topic ids present in a tagged string."""
    return {t for *_, t in span_char_ranges(tagged)}


def record_rng(seed: int, index: int) -> random.Random:
    """The per-record random stream every builder draws from."""
    return random.Random(seed * 1_000_003 + index)


def distinct_matched(record: dict, skips: Counter) -> tuple[list[int], list[tuple[int, str]]] | None:
    """Prompt order and response spans if prompt and response hold the same >=2 distinct topics, once each."""
    p_ord = [t for t, _ in ordered_spans(record["orig_prompt"])]
    r_spans = ordered_spans(record["orig_response"])
    r_ord = [t for t, _ in r_spans]
    if len(set(r_ord)) < 2:
        skips["single_topic"] += 1
    elif len(p_ord) != len(set(p_ord)) or len(r_ord) != len(set(r_ord)):
        skips["repeated_topic"] += 1
    elif set(p_ord) != set(r_ord):
        skips["set_mismatch"] += 1
    else:
        return p_ord, r_spans
    return None


def tag_spans(spans: list[tuple[int, str]]) -> str:
    """Rebuild a tagged response from (topic id, text) spans, one span per line."""
    return "\n".join(f"<{TOPICS[t]}>{txt}</{TOPICS[t]}>" for t, txt in spans)


# ---------------------------------------------------------------- removal


def remove_topic(tagged: str, drop_id: int) -> str:
    """Return the tagged string with every span of ``drop_id`` (its tags and text) removed.

    Walks the same (tag | text | stray '<') events as span_char_ranges, tracking the active topic,
    and emits everything except the tags and text belonging to the dropped topic. Handles the
    dataset's implicit-close and missing-final-close conventions, because emission is gated purely
    on the currently active topic.
    """
    parts: list[str] = []
    cur: int | None = None
    for m in _EVENT.finditer(tagged):
        seg = m.group(0)
        op = _OPEN.fullmatch(seg)
        if op:  # opening tag: switches the active topic
            cur = TOPICS.index(op.group(1))
            if cur != drop_id:
                parts.append(seg)
        elif seg.startswith("</"):  # closing tag: ends the active topic
            name = seg[2:-1]
            tid = TOPICS.index(name) if name in TOPICS else None
            if tid != drop_id:
                parts.append(seg)
            cur = None
        elif cur != drop_id:  # plain text (or a stray '<'): belongs to the active topic
            parts.append(seg)
    return "".join(parts).strip()


def build_removal(records: list[dict], seed: int, eligibility: str, skips: Counter) -> list[dict]:
    """Drop one requested topic from each eligible response; skipped records are kept with a reason."""
    rows = []
    for r in records:
        idx, op, orr = r["index"], r["orig_prompt"], r["orig_response"]
        Tp, Tr = topic_set(op), topic_set(orr)
        topics = {
            "prompt_topics": sorted(TOPICS[t] for t in Tp),
            "response_topics": sorted(TOPICS[t] for t in Tr),
        }
        if len(Tr) < 2:
            skips["single_or_empty_response"] += 1
            rows.append({"index": idx, "_status": "skip", "reason": "single_response_topic", **topics})
            continue
        if eligibility == "matched":
            eligible, candidates = Tp == Tr, sorted(Tr)
        else:  # response: drop a response topic that also lives in the prompt
            candidates = sorted(Tr & Tp)
            eligible = len(candidates) >= 1
        if not eligible:
            skips["prompt_response_mismatch"] += 1
            rows.append({"index": idx, "_status": "skip", "reason": "prompt_response_mismatch", **topics})
            continue

        drop_id = record_rng(seed, idx).choice(candidates)
        new_resp = remove_topic(orr, drop_id)
        new_Tr = topic_set(new_resp)  # sanity: the topic is gone, at least one other survives
        errs = validate_topic_tags({"prompt": op, "response": new_resp})
        if drop_id in new_Tr or len(new_Tr) < 1 or not new_resp or errs:
            skips["construct_failed"] += 1
            detail = errs or f"drop_in_resp={drop_id in new_Tr} n_surv={len(new_Tr)}"
            rows.append({"index": idx, "_status": "skip", "reason": "construct_failed", "detail": detail})
            continue
        rows.append(
            {
                "index": idx,
                "_status": "ok",
                "dropped_topic": TOPICS[drop_id],
                "dropped_topic_id": drop_id,
                "still_in_prompt": drop_id in Tp,
                "orig_prompt": op,
                "orig_response": orr,
                "inj_prompt": op,
                "inj_response": new_resp,
                "injections": [],
                "prompt_topics": topics["prompt_topics"],
                "response_topics_before": topics["response_topics"],
                "response_topics_after": sorted(TOPICS[t] for t in new_Tr),
            }
        )
    return rows


# ---------------------------------------------------------------- reordering


def derangement(k: int, rng: random.Random) -> list[int]:
    """A permutation of range(k) with no fixed point (k >= 2)."""
    idx = list(range(k))
    while True:
        p = idx[:]
        rng.shuffle(p)
        if all(p[i] != i for i in range(k)):
            return p


def build_reordering(records: list[dict], seed: int, skips: Counter) -> list[dict]:
    """Permute each eligible response's spans so that no topic stays in its prompt position."""
    rows = []
    for r in records:
        if (matched := distinct_matched(r, skips)) is None:
            continue
        p_ord, r_spans = matched
        idx, op = r["index"], r["orig_prompt"]
        texts = dict(r_spans)
        perm = derangement(
            len(p_ord), record_rng(seed, idx)
        )  # response position i <- prompt position perm[i]
        after_order = [p_ord[perm[i]] for i in range(len(p_ord))]
        new_resp = tag_spans([(t, texts[t]) for t in after_order])
        if validate_topic_tags({"prompt": op, "response": new_resp}) or topic_set(new_resp) != set(texts):
            skips["construct_failed"] += 1
            continue
        rows.append(
            {
                "index": idx,
                "_status": "ok",
                "orig_prompt": op,
                "orig_response": r["orig_response"],
                "inj_prompt": op,
                "inj_response": new_resp,
                "injections": [],
                "prompt_order": [TOPICS[t] for t in p_ord],
                "resp_order_before": [TOPICS[t] for t, _ in r_spans],
                "resp_order_after": [TOPICS[t] for t in after_order],
                "k_topics": len(p_ord),
            }
        )
    return rows


# ---------------------------------------------------------------- substitution


def build_substitution(records: list[dict], seed: int, donor_split: str, skips: Counter) -> list[dict]:
    """Swap one topic's span for a same-topic span from another test record, keeping its position.

    Donors come from every other record (``none``), or, with ``disjoint``, from a seeded half of
    the eligible records while only the other half is evaluated.
    """
    # The donor bank keeps the source file's order: the seeded donor draw indexes into it.
    bank: dict[int, list[tuple[int, str]]] = defaultdict(list)  # topic -> [(index, span text)]
    for r in records:
        for t, txt in ordered_spans(r["orig_response"]):
            if txt:
                bank[t].append((r["index"], txt))
    ordered = sorted(records, key=lambda r: r["index"])
    eligible = [(r, m) for r in ordered if (m := distinct_matched(r, skips)) is not None]

    donors = None  # indices allowed as donors; None = all
    if donor_split == "disjoint":
        idxs = [r["index"] for r, _ in eligible]
        random.Random(seed).shuffle(idxs)
        donors, recipients = set(idxs[: len(idxs) // 2]), set(idxs[len(idxs) // 2 :])
        eligible = [e for e in eligible if e[0]["index"] in recipients]
        print(f"disjoint split: {len(donors)} donors, {len(eligible)} recipients", flush=True)

    rows = []
    for r, (_, r_spans) in eligible:
        idx, op = r["index"], r["orig_prompt"]
        r_topics = [t for t, _ in r_spans]
        rng = record_rng(seed, idx)
        sub_id = rng.choice(sorted(set(r_topics)))
        pool = [(di, dt) for di, dt in bank[sub_id] if di != idx and (donors is None or di in donors)]
        if not pool:
            skips["no_donor"] += 1
            continue
        donor_index, donor_text = rng.choice(pool)
        new_resp = tag_spans([(t, donor_text if t == sub_id else txt) for t, txt in r_spans])
        if validate_topic_tags({"prompt": op, "response": new_resp}) or topic_set(new_resp) != set(r_topics):
            skips["construct_failed"] += 1
            continue
        rows.append(
            {
                "index": idx,
                "_status": "ok",
                "substituted_topic": TOPICS[sub_id],
                "substituted_topic_id": sub_id,
                "donor_index": donor_index,
                "orig_prompt": op,
                "orig_response": r["orig_response"],
                "inj_prompt": op,
                "inj_response": new_resp,
                "injections": [],
                "response_topics": sorted(TOPICS[t] for t in set(r_topics)),
            }
        )
    return rows


def main() -> None:
    """Build one perturbation set and write it next to the insertion output."""
    parser = argparse.ArgumentParser(description="Build a removal, reordering or substitution test set.")
    parser.add_argument("--kind", choices=sorted(OUTPUTS), required=True)
    parser.add_argument("--source", default=SOURCE, help="insert_offtopic.py output holding the originals")
    parser.add_argument("--out", default=None, help="output jsonl (default: the recorded run's path)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eligibility",
        choices=["matched", "response"],
        default="matched",
        help="removal only. matched: prompt and response topic sets are equal; "
        "response: any record with >=2 response topics, dropping one that is also in the prompt",
    )
    parser.add_argument(
        "--donor-split", choices=["none", "disjoint"], default="none", help="substitution only (see above)"
    )
    args = parser.parse_args()

    source = Path(args.source) if Path(args.source).is_absolute() else PROJECT_ROOT / args.source
    latest = {}
    for line in source.open(encoding="utf-8"):
        r = json.loads(line)
        latest[r["index"]] = r  # the insertion run is resumable; keep the last row per record
    in_file_order = [r for r in latest.values() if r.get("_status") == "ok"]
    records = sorted(in_file_order, key=lambda r: r["index"])
    print(f"source ok records: {len(records)}", flush=True)

    skips: Counter = Counter()
    if args.kind == "removal":
        rows = build_removal(records, args.seed, args.eligibility, skips)
    elif args.kind == "reordering":
        rows = build_reordering(records, args.seed, skips)
    else:
        rows = build_substitution(in_file_order, args.seed, args.donor_split, skips)

    out = Path(args.out or OUTPUTS[args.kind])
    out = out if out.is_absolute() else PROJECT_ROOT / out
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    ok = [r for r in rows if r["_status"] == "ok"]
    key = {"removal": "dropped_topic", "substitution": "substituted_topic", "reordering": "k_topics"}[
        args.kind
    ]
    print(f"\nwrote {out}\nok: {len(ok)}   skipped: {sum(skips.values())} {dict(skips)}")
    print(f"by {key}: {dict(Counter(r[key] for r in ok).most_common())}")


if __name__ == "__main__":
    main()
