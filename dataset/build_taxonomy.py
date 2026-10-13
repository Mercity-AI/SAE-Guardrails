#!/usr/bin/env python3
"""Hand-author taxonomy.json + strategies.json for the MULTI-topic (1-4) topic-axis run.

Extends the earlier 1-2 topic taxonomy to 1-4. Same principle: the topic *label*
is the controlled axis (uniform per-slot leaves) so per-label balance is set here and nowhere else;
persona/tone/task stay free. New here:

  * Topic slots 2,3,4 each None|Present(7 leaves). CONTIGUITY (never_combine) forbids a later slot
    being Present while an earlier one is None, so counts are exactly 1/2/3/4 with no gaps.
  * DISTINCTNESS (never_combine) forbids any two occupied slots sharing a label -> up to 4 distinct
    topics, order preserved.
  * BOUNDARY consistency unchanged: single-topic => No boundary, multi-topic => a real boundary
    style applied at EACH pivot.

Count distribution is set by the per-slot None/Present weights below (PRESENT_WEIGHT). Because
sample_mix redraws the WHOLE mix when distinctness/contiguity is violated, the *accepted* count
distribution is skewed away from the raw weights (4 distinct labels from 7 collide often), so the
weights are tuned against simula's real sampler in the audit -- not guessed. Run this and read the
FREE AUDIT before any paid generation.

Run from the repository root:  python dataset/build_taxonomy.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

LABELS = [
    "Enterprise documents",
    "General news & content",
    "Customer service",
    "Legal",
    "Financial",
    "HR & people operations",
    "Healthcare",
]

# Ordered optional slots beyond the always-present Primary. Their None/Present weights set the
# 1/2/3/4 count mix (tuned against the audit; see note in the docstring).
SLOTS = ["Second topic", "Third topic", "Fourth topic"]
# P(Present) BEFORE distinctness/contiguity rejection, per slot. Later slots lower so the natural
# taper 1>2>3>4 holds after rejection skews it further. Tune by reading the audit.
# Tuned against the audit (simula's real sampler) so single-topic is a controlled ~20% minority and
# multi-topic (2-4) is the 80% majority: ~1:20% 2:22% 3:32% 4:27%, mean 2.65 topics/record.
PRESENT_WEIGHT = {"Second topic": 0.56, "Third topic": 0.67, "Fourth topic": 0.60}


def leaf(name: str, path: list[str], weight: float, description: str = "") -> dict:
    """Return a taxonomy leaf node in the dict format Simula reads."""
    return {
        "name": name,
        "description": description,
        "weight": weight,
        "level": len(path) - 1,
        "path": path,
        "children": [],
    }


def branch(name: str, path: list[str], weight: float, children: list[dict], description: str = "") -> dict:
    """Return a taxonomy branch node holding ``children``, in the dict format Simula reads."""
    return {
        "name": name,
        "description": description,
        "weight": weight,
        "level": len(path) - 1,
        "path": path,
        "children": children,
    }


def topic_leaves(parent: list[str]) -> list[dict]:
    """Return one leaf per topic label under ``parent``, all with equal weight."""
    return [leaf(lbl, parent + [lbl], 1.0) for lbl in LABELS]


def optional_topic_slot(slot: str, order_word: str) -> dict:
    """None|Present(7 uniform) branch for one optional ordered topic slot."""
    p_present = PRESENT_WEIGHT[slot]
    return branch(
        slot,
        [slot],
        1.0,
        [
            leaf(
                "None",
                [slot, "None"],
                1.0 - p_present,
                f"No {order_word} topic. The record stops before this slot (its count is one smaller).",
            ),
            branch(
                "Present",
                [slot, "Present"],
                p_present,
                topic_leaves([slot, "Present"]),
                f"A distinct {order_word} span: after the previous topic is covered, pivot once and "
                f"cover this label fully. Must differ from every earlier topic (enforced).",
            ),
        ],
        f"None => the record has fewer topics; Present => a distinct {order_word} label. Only legal "
        f"when every earlier slot is Present (contiguity enforced).",
    )


def build_taxonomy(description: str) -> dict:
    """Build the hand-authored taxonomy: primary topic, three optional ordered slots, and the style factors."""
    factors = [
        branch(
            "Primary topic",
            ["Primary topic"],
            1.0,
            topic_leaves(["Primary topic"]),
            "The label of the first (or only) span. Uniform across the seven domains; this is the "
            "one axis that sets per-label balance, so nothing else may dictate the topic.",
        ),
        optional_topic_slot("Second topic", "second"),
        optional_topic_slot("Third topic", "third"),
        optional_topic_slot("Fourth topic", "fourth"),
        branch(
            "Boundary difficulty",
            ["Boundary difficulty"],
            1.0,
            [
                leaf(
                    "No boundary",
                    ["Boundary difficulty", "No boundary"],
                    1.0,
                    "Single-topic only: there is no topic change to detect.",
                ),
                branch(
                    "Applicable",
                    ["Boundary difficulty", "Applicable"],
                    1.0,
                    [
                        leaf(
                            "Explicit natural connective",
                            ["Boundary difficulty", "Applicable", "Explicit natural connective"],
                            0.40,
                            'An ordinary linking sentence signals each shift ("On a separate note, ...").',
                        ),
                        leaf(
                            "Light or neutral connective",
                            ["Boundary difficulty", "Applicable", "Light or neutral connective"],
                            0.35,
                            "A faint pivot; the reader feels a mild change of subject at each boundary.",
                        ),
                        leaf(
                            "Unmarked or shared-entity pivot",
                            ["Boundary difficulty", "Applicable", "Unmarked or shared-entity pivot"],
                            0.25,
                            "No connective at all: the next sentence simply starts the new topic, or the pivot rides on a person/company/object the spans share.",
                        ),
                    ],
                    "Multi-topic. The chosen style applies at EVERY pivot. Never a visible heading, XML label, or topic name as the cue.",
                ),
            ],
            "How detectable each topic change is.",
        ),
        branch(
            "Surface structure",
            ["Surface structure"],
            1.0,
            [
                leaf("Continuous prose", ["Surface structure", "Continuous prose"], 0.30, ""),
                leaf("Multi-paragraph prose", ["Surface structure", "Multi-paragraph prose"], 0.30, ""),
                leaf("Bullets or checklist", ["Surface structure", "Bullets or checklist"], 0.15, ""),
                leaf("Numbered steps", ["Surface structure", "Numbered steps"], 0.15, ""),
                leaf("Mixed prose and list", ["Surface structure", "Mixed prose and list"], 0.07, ""),
            ],
            "Mechanical layout only, independent of topic. A visible structural break must sometimes "
            "sit INSIDE a stable topic span so layout is never a boundary shortcut.",
        ),
    ]
    return {"description": description, "factors": factors}


def build_strategies() -> dict:
    """Build the sampling strategy, including the never-combine rules that keep slots contiguous and distinct."""
    never_combine = []
    # Boundary consistency: single-topic <-> no boundary; any multi-topic <-> a real boundary.
    never_combine.append(["Second topic/None", "Boundary difficulty/Applicable"])
    never_combine.append(["Second topic/Present", "Boundary difficulty/No boundary"])
    # Contiguity: a later slot Present requires every earlier optional slot Present.
    never_combine.append(["Third topic/Present", "Second topic/None"])
    never_combine.append(["Fourth topic/Present", "Second topic/None"])
    never_combine.append(["Fourth topic/Present", "Third topic/None"])
    # Distinctness: no two occupied slots share a label. Slots that can hold a label:
    #   Primary topic/<L>   and   <slot>/Present/<L>
    holders = [("Primary topic", ["Primary topic"])] + [(s, [s, "Present"]) for s in SLOTS]
    for lbl in LABELS:
        for i in range(len(holders)):
            for j in range(i + 1, len(holders)):
                a = "/".join(holders[i][1] + [lbl])
                b = "/".join(holders[j][1] + [lbl])
                never_combine.append([a, b])
    return {
        "strategies": [
            {
                "id": "balanced_all",
                "description": "Uniformly sample the topic axes and free-vary everything else; never_combine "
                "enforces single/multi boundary applicability, contiguous slot occupancy, and "
                "distinct labels across all occupied slots (up to four).",
                "taxonomy_roots": [],
                "weight": 1.0,
                "never_combine": never_combine,
            }
        ]
    }


def _count_topics(mix: dict) -> int:
    """Number of topics in a sampled mix: the primary plus each contiguous Present slot."""
    n = 1
    for s in SLOTS:
        if mix[s]["path"][-1] != "None":
            n += 1
        else:
            break
    return n


def audit(taxonomy: dict, strategies: dict, n: int = 40000) -> None:
    """Sample ``n`` mixes with Simula's own sampler and print topic, slot and boundary balance before a paid run."""
    try:
        from simula.taxonomy import sample_mix
    except ImportError as e:
        raise SystemExit("Simula is not installed; see dataset/README.md") from e

    rng = random.Random(42)
    strat = strategies["strategies"][0]
    counts = Counter()
    primary = Counter()
    span_label = Counter()
    collisions = 0
    bad_single_boundary = 0
    bad_multi_noboundary = 0
    gap = 0
    pairs = set()
    for _ in range(n):
        mix = {m["factor"]: m for m in sample_mix(taxonomy, strat, rng)}
        k = _count_topics(mix)
        counts[k] += 1
        labels = [mix["Primary topic"]["node"]]
        for s in SLOTS:
            if mix[s]["path"][-1] != "None":
                labels.append(mix[s]["path"][-1])
        primary[labels[0]] += 1
        for l in labels:
            span_label[l] += 1
        if len(set(labels)) != len(labels):
            collisions += 1
        # contiguity check (no Present after a None)
        seen_none = False
        for s in SLOTS:
            present = mix[s]["path"][-1] != "None"
            if seen_none and present:
                gap += 1
            if not present:
                seen_none = True
        bd = mix["Boundary difficulty"]["path"]
        if k == 1 and "Applicable" in bd:
            bad_single_boundary += 1
        if k > 1 and "No boundary" in bd:
            bad_multi_noboundary += 1
        for a, b in zip(labels, labels[1:]):
            pairs.add((a, b))

    print(f"\n=== FREE AUDIT ({n:,} sampled mixes, simula sample_mix) ===")
    print("topic-count distribution:")
    for k in (1, 2, 3, 4):
        print(f"  {k}-topic: {counts[k] / n:6.1%}")
    print(f"  mean topics/record: {sum(k * c for k, c in counts.items()) / n:.2f}")
    print("\nPrimary-topic marginal (target 14.3% each):")
    for l in LABELS:
        print(f"  {primary[l] / n:6.1%}  {l}")
    tot = sum(span_label.values())
    print("\nTOTAL span share per label (all slots) — must stay ~uniform 14.3%:")
    for l in LABELS:
        print(f"  {span_label[l] / tot:6.1%}  {l}")
    print(f"\nordered adjacent pairs seen: {len(pairs)}/42")
    print(
        f"label collisions: {collisions}   contiguity gaps: {gap}   "
        f"single-with-boundary: {bad_single_boundary}   multi-without-boundary: {bad_multi_noboundary}"
    )
    assert collisions == 0 and gap == 0 and bad_single_boundary == 0 and bad_multi_noboundary == 0, (
        "CONSTRAINT VIOLATED"
    )
    print("\nAll hard constraints hold. Files are safe to run.")


def main() -> None:
    """Write taxonomy.json and strategies.json to the run's output directory and audit them."""
    parser = argparse.ArgumentParser(description="Hand-author the taxonomy and strategies, then audit them.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/GPT_5.6_10k_2108"),
        help="the YAML's project.output_dir (default matches topic_shift_10k.yaml)",
    )
    out = parser.parse_args().out
    description = (
        "Topic-drift detector dataset: prompt+response pairs with 1-4 topic-labelled spans (see YAML)."
    )
    taxonomy = build_taxonomy(description)
    strategies = build_strategies()
    out.mkdir(parents=True, exist_ok=True)
    (out / "taxonomy.json").write_text(json.dumps(taxonomy, ensure_ascii=False, indent=1))
    (out / "strategies.json").write_text(json.dumps(strategies, ensure_ascii=False, indent=1))
    print(
        f"wrote {out / 'taxonomy.json'} and {out / 'strategies.json'} "
        f"({len(strategies['strategies'][0]['never_combine'])} never_combine rules)"
    )
    audit(taxonomy, strategies)


if __name__ == "__main__":
    main()
