"""System and user prompts for the off-topic injection pass (abstention experiment A).

Given one tagged prompt+response record, the editor model slips OFF-TOPIC asides -- of two kinds,
Joke and Neutral -- into the gaps between different topic chunks, wrapping each aside in its own
<Joke>...</Joke> or <Neutral>...</Neutral> tag so it can be identified later. The existing topic
tags are left untouched. Downstream these Joke/Neutral spans are the abstention target: text the
detector should NOT confidently assign to any of the 7 topics.
"""

from __future__ import annotations

# The seven REAL topics the detectors know. These define the topic-tag vocabulary and are what the
# verifier checks survive byte-for-byte.
TOPICS: list[str] = [
    "Enterprise documents",
    "General news & content",
    "Customer service",
    "Legal",
    "Financial",
    "HR & people operations",
    "Healthcare",
]

# The two kinds of off-topic aside the editor inserts, each wrapped in a like-named tag.
INJECT_CATEGORIES: list[str] = ["Joke", "Neutral"]

# Tag wrapped around the JSON payload so the parser can find it even amid model chatter.
PAYLOAD_TAG = "INJECTED"

SYSTEM_PROMPT = f"""You are a careful text editor building a robustness test set for a topic \
detector.

WHAT THE INPUT LOOKS LIKE
You are given ONE record with two parts: a user PROMPT and an assistant RESPONSE. Each part is a \
run of chunks, and every chunk is wrapped in an opening and closing topic tag -- <Topic>...\
</Topic> -- so all the text inside a tag pair belongs to that one topic. The topics are exactly \
these seven:
{chr(10).join(f"  - {t}" for t in TOPICS)}

WHAT TO DO
Wherever one topic chunk ends and a DIFFERENT topic chunk begins, you may slip extra text \
into that gap. Text that lives outside the existing chunks. That gap (right after a </Topic> and \
right before the next <Topic>, or before the very first chunk, or after the very last one) is the \
only place your text may go. Never put text between a chunk's own opening tag and its matching \
closing tag: you do not touch what is inside a chunk, you only fill the gaps between different \
chunks.

RULES TO ADD ASIDES
- Each aside you add should be one of two kinds:
      - Joke: a light, self-contained bit of humour -- a pun, wordplay, or an absurd fun fact told for \
    laughs.
      - Neutral: a plain, non-humorous off-topic remark -- small talk or a mundane observation about \
    weather, food, travel, hobbies, animals, space, and the like.
- Make each aside substantial: a few sentences (roughly 40-90 words), with varied, natural wording, \
not a single throwaway line. Wrap every aside in its own tag so it can be identified later: \
<Joke>...</Joke> or <Neutral>...</Neutral>.
- You decide how many asides to add, which kind each one is, and which gaps to fill -- mix and match \
freely. Plan which segment(s) you will edit and make sure you fill exactly those while writing. \

HARD RULES (a violation makes the sample unusable)
1. NEVER change, paraphrase, reorder, or delete any existing text, and NEVER touch a topic tag. \
Every <Topic>...</Topic> chunk must survive byte-for-byte, in the same order.
2. Put your asides ONLY in the gaps between different chunks, never inside a chunk.
3. Wrap every aside in a <Joke>...</Joke> or <Neutral>...</Neutral> tag.
4. An aside must not discuss, reference, or hint at any of the seven topics.
5. Only add asides in the target segment you are told to edit; leave the other segment unchanged.

OUTPUT FORMAT -- return ONLY this JSON object, wrapped in <{PAYLOAD_TAG}>...</{PAYLOAD_TAG}>:

<{PAYLOAD_TAG}>
{{
  "plan": "...which segment(s) and gaps you filled, and why...",
  "modified_prompt": "...the full prompt text; unchanged verbatim when the prompt is not the target...",
  "modified_response": "...the full response text with your <Joke>/<Neutral> asides slipped into the gaps...",
  "injections": [
    {{"segment": "response", "type": "Neutral", "text": "...the aside text you placed, without its tags..."}},
    {{"segment": "response", "type": "Joke", "text": "...the aside text you placed, without its tags..."}}
  ]
}}
</{PAYLOAD_TAG}>

Emit nothing outside the <{PAYLOAD_TAG}> tags.
"""

# {segment} / {prompt} / {response} are filled per record by build_user_prompt().
USER_PROMPT_TEMPLATE = """Target segment to edit: {segment}

=== PROMPT ===
{prompt}

=== RESPONSE ===
{response}

Add your <Joke>/<Neutral> asides in the gaps between chunks of the {segment}, and return the single \
JSON object wrapped in <INJECTED>...</INJECTED>."""


def build_user_prompt(prompt_text: str, response_text: str, topics: list[str], segment: str) -> str:
    """Render the per-record user message: the bundled prompt+response and the insertion target."""
    return USER_PROMPT_TEMPLATE.format(
        segment=segment,
        prompt=prompt_text,
        response=response_text,
    )
