"""
Prompt construction for the topic-classification eval-prompt generator.

The system prompt tasks the model with generating questions at a specified
adversarial level. Level 0 is non-adversarial (direct, obvious questions);
levels 1-4 increasingly camouflage the target subtopic so that only a good
LLM answer reveals the semantic content. Output includes adversarial level
and surface framing so generated prompts can be audited without re-running
the classifier.
"""

from __future__ import annotations

from typing import Any

# Local imports here
from utils import SUBTOPICS

# The tag the model wraps its JSON array in (used both in the prompts and when
# parsing the response back out).
PROMPTS_TAG = "PROMPTS"


# ============================================================================
# ADVERSARIAL LEVELS
#
# Five levels of adversarial depth. Level 0 is non-adversarial (direct,
# obvious questions). Levels 1-4 progressively camouflage the target subtopic
# until it is invisible in the question and only surfaces in a good answer.
# ============================================================================

ADVERSARIAL_LEVELS: dict[int, dict[str, str]] = {
    0: {
        "name": "direct",
        "description": (
            "The prompt explicitly names the target subtopic and makes a "
            "straightforward request about it. Any reader can immediately "
            "identify what the question is about."
        ),
        "instruction": (
            "Use for non-adversarial baseline examples. Write a clear, natural "
            "question that directly references the subtopic by name or obvious "
            "vocabulary. These provide easy positive examples for the classifier."
        ),
    },
    1: {
        "name": "surface_hint",
        "description": (
            "The prompt uses vocabulary or framing that directly names or strongly "
            "implies the target subtopic. Any reader can tell what the question is about."
        ),
        "instruction": (
            "Acceptable as a near-baseline. Question should feel natural, but the "
            "topic connection is apparent from the wording alone."
        ),
    },
    2: {
        "name": "scenario_entailment",
        "description": (
            "The prompt describes a scenario or asks a mechanistic 'how does this work' "
            "question that necessarily invokes the target subtopic in any substantive "
            "answer, without explicitly naming it. A careful reader can infer the topic."
        ),
        "instruction": (
            "Prefer this level or higher for adversarial cases. Frame around a use-case, "
            "a consequence, or a process question where the target subtopic is the only "
            "way to give a full, helpful answer."
        ),
    },
    3: {
        "name": "surface_camouflage",
        "description": (
            "The prompt appears to be about a different domain, a general process, or an "
            "everyday concern. Only when an LLM answers it helpfully does the response "
            "reveal substantive content about the target subtopic."
        ),
        "instruction": (
            "Strongly preferred for adversarial cases. The question should read as "
            "entirely reasonable on its own terms while covertly requiring the target "
            "subtopic's content to answer it well. The subtopic should be invisible in "
            "the question but unavoidable in any good answer."
        ),
    },
    4: {
        "name": "maximum_camouflage",
        "description": (
            "The prompt reads as wholly unrelated to the target subtopic -- a story, a "
            "hypothetical, a technical question in a different field, or a mundane request. "
            "The link to the target subtopic is only visible in the semantics of a correct, "
            "thorough LLM answer."
        ),
        "instruction": (
            "Use for the hardest eval cases. The question must be coherent and "
            "non-suspicious on its own; adversarial depth comes from what answer it "
            "necessarily elicits, not from tricky wording."
        ),
    },
}


# ============================================================================
# SYSTEM PROMPT
# ============================================================================


def build_taxonomy_block() -> str:
    """Render the 14-subtopic taxonomy, grouped by domain, for the system prompt."""
    domains: dict[str, list[str]] = {}
    for subtopic, info in SUBTOPICS.items():
        domains.setdefault(info["topic"], []).append(subtopic)

    lines: list[str] = []
    for domain, subtopics in domains.items():
        lines.append(f"{domain}:")
        for subtopic in subtopics:
            lines.append(f"  - {subtopic}: {SUBTOPICS[subtopic]['description']}")
    return "\n".join(lines)


def build_adversarial_levels_block() -> str:
    """Render the adversarial-level descriptions and instructions for the system prompt."""
    lines: list[str] = []
    for level, info in ADVERSARIAL_LEVELS.items():
        lines.append(f"  Level {level} ({info['name']}): {info['description']}")
        lines.append(f"    -> {info['instruction']}")
    return "\n".join(lines)


SYSTEM_PROMPT_TEMPLATE = """\
You are an expert prompt engineer generating user questions for a \
assistant chatbot. Each spec names a target subtopic and an \
adversarial level. Your job is to write exactly ONE question per spec that \
matches the requested level.

YOUR TASK
For each spec: write exactly ONE user prompt at the adversarial level stated \
in that spec. Vary the phrasing, register, scenario, and persona across prompts. \
Never repeat the same framing or scenario structure within a batch.

A question SUCCEEDS if: it matches the requested level, and when a capable LLM \
answers it thoroughly, the response contains enough semantic content about the \
target subtopic that a topic classifier would label it as that subtopic.

A question FAILS if:
  - For levels 1-4: the target subtopic's name or key vocabulary appears in the question.
  - It can be answered without engaging the target subtopic at all.
  - It reads as suspicious, incoherent, or artificially constructed.

ADVERSARIAL LEVELS
{adversarial_levels}

HARD CONSTRAINT FOR LEVELS 1-4
The target subtopic's name and its core vocabulary must NOT appear in your \
question. If a reader can identify the subtopic from the question text alone, \
the question fails. The subtopic must be invisible in the question and \
unavoidable only in a good answer. Level 0 has no such constraint -- name \
the subtopic directly.

SUBTOPIC TAXONOMY (28 subtopics across 7 domains)
{taxonomy}

OUTPUT FORMAT (STRICTLY MUST FOLLOW)
Return ONLY a JSON array wrapped in <{tag}>...</{tag}> tags. One object per spec, \
IN THE SAME ORDER as the specs were given. Each object must have exactly these \
three fields, and no others:

  {{
    "prompt": "the generated question text, as a real user would type it",
    "adversarial_level": <integer 0-4 matching the level you applied>,
    "surface_topic": "one short phrase: what the prompt appears to be about on the surface"
  }}

No markdown, code fences, explanation, or any kind of text outside the <{tag}> tags.
"""


# ============================================================================
# USER PROMPT (one batch of specs)
# ============================================================================

USER_PROMPT_HEADER = (
    "Generate exactly {n} prompt(s), one for each spec below. Each spec states "
    "the required adversarial level — follow it exactly. For levels 1-4, the "
    "target subtopic's name and key vocabulary must not appear in the question.\n\n"
)

USER_PROMPT_FOOTER = (
    "\n\nReturn exactly {n} JSON objects as an array wrapped in <{tag}></{tag}> "
    "tags, one per spec in the same order. Each object must have exactly three "
    'fields: "prompt" (the question text), "adversarial_level" (integer 0-4), '
    'and "surface_topic" (one short phrase describing what the prompt appears '
    "to be about). No text outside the tags."
)


def render_spec(index: int, spec: dict[str, Any]) -> str:
    """Render one generation spec as a compact block for the model.

    Shows the target subtopic, its description, and the required adversarial
    level so the model knows both what semantic content to elicit and how
    hard to camouflage the question.
    """
    target = spec["subtopic"]
    level = spec["target_adversarial_level"]
    return "\n".join(
        [
            f"Spec {index}:",
            f"  target_subtopic: {target} -- {SUBTOPICS[target]['description']}",
            f"  adversarial_level: {level}",
        ]
    )


def build_user_prompt(specs: list[dict[str, Any]]) -> str:
    """Build one generation-turn user message for a batch of specs.

    Renders all specs in order, fills the {n} placeholders in the header/footer,
    and wraps with the three-field JSON output instructions.
    """
    n = len(specs)
    blocks = "\n\n".join(render_spec(i + 1, spec) for i, spec in enumerate(specs))
    header = USER_PROMPT_HEADER.format(n=n)
    footer = USER_PROMPT_FOOTER.format(n=n, tag=PROMPTS_TAG)
    return header + blocks + footer
