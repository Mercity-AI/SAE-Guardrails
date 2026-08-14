#!/usr/bin/env python3
"""Hand-author taxonomy.json + strategies.json for the topic-axis prefill run, then verify the
distribution for free with simula's own sampler before any paid generation.

Why hand-authored: the whole failure of runs/1500 was that the topic *label* was never a controlled
variable — it leaked in through the persona axis, so "consumer seeking help" collapsed onto Customer
service (29.7% of spans; 64.6% of the single-topic negatives). Here the label IS the axis:

  * Primary topic        -> the first (or only) span's label   [7 leaves, uniform]
  * Second topic         -> None (single-topic) OR one of 7    [None vs Present branch]

Independence of the two factors makes the per-label marginal exactly uniform and reaches all 42
ordered pairs; never_combine forbids (a) Primary == Second collisions and (b) single-topic records
being handed a real boundary leaf. Persona / tone / task are deliberately NOT factors — they stay
free variation so they can never again dictate the label.

Run:  python build_topic_axis_taxonomy.py            # writes files + prints the free audit
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIRECTORY = PROJECT_ROOT / "runs" / "topic_axis"

LABELS = [
    "Enterprise documents",
    "General news & content",
    "Customer service",
    "Legal",
    "Financial",
    "HR & people operations",
    "Healthcare",
]


def leaf(name: str, path: list[str], weight: float, description: str = "") -> dict:
    """Create one weighted taxonomy leaf."""
    return {
        "name": name,
        "description": description,
        "weight": weight,
        "level": len(path) - 1,
        "path": path,
        "children": [],
    }


def branch(
    name: str,
    path: list[str],
    weight: float,
    children: list[dict],
    description: str = "",
) -> dict:
    """Create one weighted taxonomy branch."""
    return {
        "name": name,
        "description": description,
        "weight": weight,
        "level": len(path) - 1,
        "path": path,
        "children": children,
    }


def topic_leaves(parent: list[str]) -> list[dict]:
    """Return equally weighted leaves for all supported topics."""
    # Uniform weight -> the marginal is set here and nowhere else.
    return [leaf(lbl, parent + [lbl], 1.0) for lbl in LABELS]


def build_taxonomy(description: str) -> dict:
    """Build the complete controlled topic-axis taxonomy."""
    factors = [
        branch(
            "Primary topic",
            ["Primary topic"],
            1.0,
            topic_leaves(["Primary topic"]),
            "The label of the first (or only) span. Uniform across the seven domains; this is the "
            "one axis that sets per-label balance, so nothing else may dictate the topic.",
        ),
        branch(
            "Second topic",
            ["Second topic"],
            1.0,
            [
                leaf(
                    "None",
                    ["Second topic", "None"],
                    1.0,
                    "SINGLE-TOPIC record: there is no second topic and no boundary. The whole exchange "
                    "stays on the Primary topic. These are the negatives that teach the detector not to "
                    "invent a switch.",
                ),
                branch(
                    "Present",
                    ["Second topic", "Present"],
                    1.0,
                    topic_leaves(["Second topic", "Present"]),
                    "TWO-TOPIC record: the response covers Primary fully, then pivots once to this "
                    "second label and covers it fully. Must differ from Primary (enforced).",
                ),
            ],
            "None => single-topic; Present => a distinct second label. The None/Present weight sets "
            "the single-vs-two ratio; the label weights under Present are uniform.",
        ),
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
                # Skewed easier: most boundaries carry a clear/soft connective (readily detectable); the
                # hard no-connective cases are a smaller minority.
                branch(
                    "Applicable",
                    ["Boundary difficulty", "Applicable"],
                    1.0,
                    [
                        leaf(
                            "Explicit natural connective",
                            [
                                "Boundary difficulty",
                                "Applicable",
                                "Explicit natural connective",
                            ],
                            0.40,
                            'An ordinary linking sentence signals the shift ("On a separate note, ...").',
                        ),
                        leaf(
                            "Light or neutral connective",
                            [
                                "Boundary difficulty",
                                "Applicable",
                                "Light or neutral connective",
                            ],
                            0.35,
                            "A faint pivot; the reader feels a mild change of subject.",
                        ),
                        leaf(
                            "Unmarked or shared-entity pivot",
                            [
                                "Boundary difficulty",
                                "Applicable",
                                "Unmarked or shared-entity pivot",
                            ],
                            0.25,
                            "No connective at all: the next sentence simply starts the new topic, or the pivot rides on a person/company/object the two topics share.",
                        ),
                    ],
                    "Two-topic only. Never a visible heading, XML label, or topic name as the cue.",
                ),
            ],
            "How detectable the single topic change is.",
        ),
        branch(
            "Surface structure",
            ["Surface structure"],
            1.0,
            [
                leaf(
                    "Continuous prose",
                    ["Surface structure", "Continuous prose"],
                    0.30,
                    "",
                ),
                leaf(
                    "Multi-paragraph prose",
                    ["Surface structure", "Multi-paragraph prose"],
                    0.30,
                    "",
                ),
                leaf(
                    "Bullets or checklist",
                    ["Surface structure", "Bullets or checklist"],
                    0.15,
                    "",
                ),
                leaf(
                    "Numbered steps", ["Surface structure", "Numbered steps"], 0.15, ""
                ),
                leaf(
                    "Mixed prose and list",
                    ["Surface structure", "Mixed prose and list"],
                    0.07,
                    "",
                ),
            ],
            "Mechanical layout only, independent of topic. A visible structural break must sometimes "
            "sit INSIDE a stable topic span so layout is never a boundary shortcut.",
        ),
    ]
    return {"description": description, "factors": factors}


def build_strategies() -> dict:
    """Build the intentionally unpinned sampling strategy."""
    # One strategy: sample every factor fully. never_combine does all the work.
    never_combine = [
        # single-topic <-> no boundary consistency (clean lineage)
        ["Second topic/None", "Boundary difficulty/Applicable"],
        ["Second topic/Present", "Boundary difficulty/No boundary"],
    ]
    # Primary != Second (no A->A "pair")
    for lbl in LABELS:
        never_combine.append([f"Primary topic/{lbl}", f"Second topic/Present/{lbl}"])
    return {
        "strategies": [
            {
                "id": "balanced_all",
                "description": "Uniformly sample the topic axes and free-vary everything else; never_combine "
                "enforces single/two applicability and forbids A->A pairs.",
                "taxonomy_roots": [],
                "weight": 1.0,
                "never_combine": never_combine,
            }
        ]
    }


def audit(taxonomy: dict, strategies: dict, n: int = 40000) -> None:
    """Sample the taxonomy and print balance and coverage diagnostics."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Simula"))
    from simula.taxonomy import sample_mix  # noqa

    rng = random.Random(42)
    strat = strategies["strategies"][0]
    primary = Counter()
    second = Counter()
    single = 0
    pairs = Counter()
    bad_single_boundary = 0
    bad_two_noboundary = 0
    collisions = 0
    for _ in range(n):
        mix = {m["factor"]: m for m in sample_mix(taxonomy, strat, rng)}
        p = mix["Primary topic"]["node"]
        s = mix["Second topic"][
            "path"
        ]  # ["Second topic","None"] or [...,"Present",label]
        primary[p] += 1
        is_single = s[-1] == "None"
        bd = mix["Boundary difficulty"]["path"]
        if is_single:
            single += 1
            if "Applicable" in bd:
                bad_single_boundary += 1
        else:
            sec = s[-1]
            second[sec] += 1
            if sec == p:
                collisions += 1
            pairs[(p, sec)] += 1
            if "No boundary" in bd:
                bad_two_noboundary += 1

    print(f"\n=== FREE AUDIT ({n:,} sampled mixes) ===")
    print(f"single-topic: {single/n:6.1%}   two-topic: {1-single/n:6.1%}")
    print("\nPrimary-topic marginal (target 14.3% each):")
    for label in LABELS:
        print(f"  {primary[label]/n:6.1%}  {label}")
    all_span = Counter(primary)
    for label in LABELS:
        all_span[label] += second[label]
    tot = sum(all_span.values())
    print(
        "\nTOTAL span share per label (primary+second) — the number that was 29.7% CS before:"
    )
    for label in LABELS:
        print(f"  {all_span[label]/tot:6.1%}  {label}")
    print(
        f"\nordered pairs seen: {len(pairs)}/42   (min {min(pairs.values())}, max {max(pairs.values())} of {sum(pairs.values())})"
    )
    print(
        f"illegal single-topic-with-boundary: {bad_single_boundary}   two-topic-without-boundary: {bad_two_noboundary}   A->A collisions: {collisions}"
    )
    assert (
        len(pairs) == 42
        and bad_single_boundary == 0
        and bad_two_noboundary == 0
        and collisions == 0
    ), "CONSTRAINT VIOLATED"
    print("\nAll hard constraints hold. Files are safe to run.")


def main() -> None:
    """Write the taxonomy files and run the free sampling audit."""
    # Description is injected by simula at build time; keep a short stand-in here for the taxonomy file.
    description = "Topic-drift detector dataset: prompt+response pairs with topic-labelled spans (see YAML)."
    taxonomy = build_taxonomy(description)
    strategies = build_strategies()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIRECTORY / "taxonomy.json").write_text(
        json.dumps(taxonomy, ensure_ascii=False, indent=1)
    )
    (OUTPUT_DIRECTORY / "strategies.json").write_text(
        json.dumps(strategies, ensure_ascii=False, indent=1)
    )
    print(
        f"wrote {OUTPUT_DIRECTORY/'taxonomy.json'} and {OUTPUT_DIRECTORY/'strategies.json'}"
    )
    audit(taxonomy, strategies)


if __name__ == "__main__":
    main()
