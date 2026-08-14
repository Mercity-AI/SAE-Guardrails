"""Prompt overrides for the topic-axis prefill run.

Only the PARAMETER NAMES must match simula/prompts.py (validate enforces this). We override two
functions:

  * meta_prompt_prompt  -> make the sampled Primary/Second topic MANDATORY and label-defining, so the
    topic never drifts back to whatever the persona implies (the runs/1500 failure mode), and make
    the single-vs-two logic explicit from the "Second topic = None" ingredient.
  * critique_prompt     -> reject the exact structural defects that slipped through before: malformed/
    unbalanced tags, tag names that are not one of the seven labels (kills <strong>/<table> leakage),
    tags that disagree with the topics list, prompt/response order mismatch, and any span that returns to a
    finished topic.
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


def schema_text(schema: dict[str, Any] | None) -> str:
    """Serialize the requested record schema for prompt inclusion."""
    return (
        "free-form text"
        if schema is None
        else json.dumps(schema, ensure_ascii=False, indent=1)
    )


def mix_lines(mix: list[dict[str, Any]]) -> str:
    """Render sampled taxonomy choices as readable prompt lines."""
    lines = []
    for m in mix:
        desc = f" — {m['description']}" if m.get("description") else ""
        lines.append(f"- {m['factor']}: {m['node']}{desc}")
    return "\n".join(lines)


def meta_prompt_prompt(
    description: str, schema: dict[str, Any] | None, mix: list[dict[str, Any]], k: int
) -> str:
    """Build the teacher prompt for one topic-axis generation request."""
    by_factor = {m["factor"]: m for m in mix}
    primary = by_factor.get("Primary topic", {}).get("node", "")
    second_path = by_factor.get("Second topic", {}).get("path", [])
    is_single = (second_path[-1:] == ["None"]) if second_path else True
    second = "" if is_single else second_path[-1]

    if is_single:
        topic_rule = (
            f'THIS IS A SINGLE-TOPIC RECORD. The one and only topic is "{primary}". The topics list MUST be '
            f'exactly ["{primary}"]. The entire prompt and response stay on "{primary}" with NO '
            f"boundary and NO second topic. Ignore any boundary-difficulty ingredient — it does not "
            f"apply. Wrap the whole prompt in one <{primary}>...</{primary}> "
            f"tag and the whole response in one <{primary}>...</{primary}> tag."
        )
    else:
        topic_rule = (
            f'THIS IS A TWO-TOPIC RECORD. First topic "{primary}", then a single clean pivot to '
            f'"{second}". The topics list MUST be exactly ["{primary}", "{second}"], in that order, in '
            f'both prompt and response. Cover "{primary}" fully, then transition once and cover '
            f'"{second}" fully — never return to "{primary}". Wrap each span in its own exact tag: '
            f"<{primary}>...</{primary}> then <{second}>...</{second}>."
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

The ONLY legal tag names are the seven exact labels: {LABELS}. Never emit any other tag
(no <strong>, <b>, <table>, <br>, <li>, markdown, or headings) inside the text; removing the topic
tags must leave clean, natural prose. The boundary must be a felt sentence-level pivot, never a
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
    """Build the critic prompt used to validate one generated record."""
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
3. MATCHES TOPICS: the ordered tag labels in the response are not exactly equal to the topics list; or
   the prompt tag order differs from the response tag order.
4. COUNT: the topics list has more than two entries, or the two entries are identical.
5. NO RETURN: for a two-topic record the response (or prompt) returns to the first topic after the
   second has begun, i.e. a label appears in more than one separate span, or the topics interleave.
6. CLEAN PROSE: removing all tags does not leave natural, coherent text; or the boundary is marked
   by a visible heading, a bullet label, or text that literally names a topic/domain.
7. FAITHFUL: prompt and response cover different topics, drift to an untagged topic, or the content
   of a span does not match its label per the domain Includes/Excludes rules in the description.

Return JSON:
{{"verdict": "accept" | "reject", "explanation": "..."}}
""".strip()
