"""Prompt overrides for the MULTI-topic (1-4) topic-axis prefill run.

Generalizes the earlier 1-2 topic prompt set to 1-4 topics, and adds explicit
response-length control. Only the PARAMETER NAMES must match simula/prompts.py (validate enforces
this). We override:

  * meta_prompt_prompt -> read the ordered topic slots (Primary + Second/Third/Fourth), make the
    ordered topic sequence MANDATORY and label-defining, spell out the single-vs-multi rule, and set
    a whole-response LENGTH TARGET (~300-400 words total, spread across the spans).
  * critique_prompt    -> reject the structural defects, now for up to FOUR distinct ordered spans:
    bad/foreign tag names, malformed/overlapping tags, tag order != topics list, duplicate or >4
    topics, any label appearing in more than one span (interleaving / return), and dirty prose.
"""

from __future__ import annotations

import json
from typing import Any

LABELS = [
    "Enterprise documents",
    "General news & content",
    "Customer service",
    "Legal",
    "Financial",
    "HR & people operations",
    "Healthcare",
]

# Ordered optional slots beyond Primary; must match build_taxonomy.py.
SLOTS = ["Second topic", "Third topic", "Fourth topic"]

# Whole-response length target (tags stripped). The 2k teacher averaged ~143 words; this run aims
# for ~600 words. Total is targeted (not per span), so a 1-topic record fills it with one long span
# and a 4-topic record splits it across four (~150 words/span).
LEN_LO, LEN_HI = 550, 650


def schema_text(schema: dict[str, Any] | None) -> str:
    """Render the output JSON schema for the prompt, or say the output is free-form."""
    return "free-form text" if schema is None else json.dumps(schema, ensure_ascii=False, indent=1)


def mix_lines(mix: list[dict[str, Any]]) -> str:
    """Render a sampled taxonomy mix as one bullet per factor, with its node and description."""
    lines = []
    for m in mix:
        desc = f" — {m['description']}" if m.get("description") else ""
        lines.append(f"- {m['factor']}: {m['node']}{desc}")
    return "\n".join(lines)


def ordered_topics(mix: list[dict[str, Any]]) -> list[str]:
    """Primary, then each Present slot in order; stop at the first absent slot (contiguity)."""
    by_factor = {m["factor"]: m for m in mix}
    topics = [by_factor.get("Primary topic", {}).get("node", "")]
    for s in SLOTS:
        path = by_factor.get(s, {}).get("path", [])
        if path[-1:] == ["None"] or "Present" not in path:
            break
        topics.append(path[-1])
    return [t for t in topics if t]


def _tag_sequence(topics: list[str]) -> str:
    """Render the required tag order, e.g. ``<Legal>...</Legal> then <Financial>...</Financial>``."""
    return " then ".join(f"<{t}>...</{t}>" for t in topics)


def meta_prompt_prompt(
    description: str, schema: dict[str, Any] | None, mix: list[dict[str, Any]], k: int
) -> str:
    """Build the teacher prompt for one topic-axis generation request (1-4 topics)."""
    topics = ordered_topics(mix)
    n = len(topics)
    topics_list = json.dumps(topics, ensure_ascii=False)

    if n == 1:
        t = topics[0]
        topic_rule = (
            f'THIS IS A SINGLE-TOPIC RECORD. The one and only topic is "{t}". The topics list MUST be '
            f'exactly {topics_list}. The entire prompt and response stay on "{t}" with NO boundary and '
            f"NO other topic. Ignore any boundary-difficulty ingredient — it does not apply. Wrap the "
            f"whole prompt in one <{t}>...</{t}> tag and the whole response in one <{t}>...</{t}> tag."
        )
    else:
        topic_rule = (
            f"THIS IS A {n}-TOPIC RECORD. The topics, in order, are {topics_list}. The topics list MUST be "
            f"exactly {topics_list}, in that order, in BOTH prompt and response. Cover each topic fully, "
            f"then make ONE clean pivot to the next — {n - 1} pivot(s) total — never interleaving, never "
            f"returning to an earlier topic, never repeating a label. Each label appears in exactly one "
            f"contiguous span. Wrap each span in its own exact tag, in order: {_tag_sequence(topics)}. "
            f"Apply the sampled boundary-difficulty style at EVERY pivot."
        )

    length_rule = (
        f"LENGTH — the assistant RESPONSE (all spans together, tags removed) should total roughly "
        f"{LEN_LO}-{LEN_HI} words. Spread it across the {n} span(s): with more topics each span is "
        f"shorter, with fewer topics each span is longer, but the WHOLE response stays in that band. "
        f"Every span must still be substantive — never a token gesture just to fit the count. The user "
        f"PROMPT stays short and natural regardless."
    )

    return f"""
Dataset description:
{description}

Output JSON Schema:
{schema_text(schema)}

Sampled ingredients for this record:
{mix_lines(mix)}

TOPIC RULE — this is an important rule and overrides any topic the persona/style/task might suggest:
{topic_rule}

{length_rule}

The ONLY legal tag names are the seven exact labels: {LABELS}. Never emit any other tag
(no <strong>, <b>, <table>, <br>, <li>, markdown, or headings) inside the text; removing the topic
tags must leave clean, natural prose. Every boundary must be a felt sentence-level pivot, never a
visible heading or a phrase that names the topic. Use the remaining ingredients (surface structure,
difficulty) to vary the writing, and freely invent a realistic persona, tone, and task — vary those
hard, but never let them change the topic labels fixed above.

PROMPT REGISTER — keep it natural. The user prompt is simply a normal message a real person would
send to an assistant chatbot: they state their situation and ask for what they want in plain
language. Real users do not enumerate every sub-question or dictate the shape of the answer
("please include A, B, C, and also D") — leave the detail to the response. Some users are polished,
some are terse, casual, or a little messy. The response is likewise just a normal, helpful
assistant reply.

Generate {k} diverse meta-prompts, each telling a generator exactly what record to create. Keep each
meta-prompt itself compact and plain — a few sentences of instruction, not an elaborate scenario
brief. Do not add requirements beyond the ingredients and rules above.
Return JSON:
{{"meta_prompts": ["...", "..."]}}
""".strip()


def critique_prompt(
    description: str, schema: dict[str, Any], meta_prompt: str, record: dict[str, Any]
) -> str:
    """Build the critic prompt used to validate one generated record (1-4 ordered spans)."""
    return f"""
You are a strict validator for a topic-segmentation dataset. Reject on ANY violation below.

Dataset description:
{description}

Meta-prompt the record must satisfy:
{meta_prompt}

Generated record:
{json.dumps(record, ensure_ascii=False)}

REJECT if any of these hold (be literal):
1. TAG NAMES: any XML tag in the prompt or response is not one of exactly {LABELS}. Any <strong>,
   <b>, <em>, <table>, <tr>, <td>, <br>, <li>, <p>, markdown, or heading tag => REJECT.
2. MALFORMED: tags are not cleanly opened and closed, or they nest/overlap, or a tag is left
   unclosed.
3. MATCHES TOPICS: the ordered tag labels in the response are not exactly equal to the topics list;
   or the prompt tag order differs from the response tag order.
4. COUNT: the topics list has fewer than one or more than FOUR entries, or any two entries are
   identical (labels must be distinct).
5. NO RETURN / NO INTERLEAVE: any label appears in more than one separate span, or the spans
   interleave rather than running as contiguous, non-overlapping spans in the listed order (cover
   topic 1 fully, pivot once, cover topic 2 fully, ... never go back).
6. CLEAN PROSE: removing all tags does not leave natural, coherent text; or any boundary is marked
   by a visible heading, a bullet label, or text that literally names a topic/domain.
7. FAITHFUL: prompt and response cover different topics, drift to an untagged topic, or the content
   of a span does not match its label per the domain Includes/Excludes rules in the description.

Do NOT reject solely for response length; length is a soft target, not a hard rule.

Return JSON:
{{"verdict": "accept" | "reject", "explanation": "..."}}
""".strip()
