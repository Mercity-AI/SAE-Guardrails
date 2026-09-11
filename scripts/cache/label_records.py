"""Per-token labeler: tagged {prompt,response,topics} record -> (input_ids, roles, labels).

Reproduces the 2k cache convention (validated exactly against sae1500 labels.npy/role_ids.npy):
  * input_ids = apply_chat_template(user=strip(prompt), assistant=strip(response)), tags removed.
  * roles: 1 = prompt/template tokens, 2 = assistant response tokens (= everything from the
    prompt-with-generation-prompt prefix length onward).
  * labels: PAD(-100) for role-1 tokens; for each response token, the topic id of the span it falls
    in. Response tokens are aligned to strip(response) via offset mapping; trailing template tokens
    (end_of_turn) inherit the last span's topic (matches the single-topic cache behavior).
"""

from __future__ import annotations

import re
from typing import Any

# local imports
from important_scripts.model.models import PAD, TOPICS

TAG = re.compile(r"</?(?:" + "|".join(re.escape(t) for t in TOPICS) + r")>")
_OPEN = re.compile(r"<(" + "|".join(re.escape(t) for t in TOPICS) + r")>")
_ANY_TAG = re.compile(r"<(/?)([^<>]+)>")


def strip_tags(s: str) -> str:
    """Remove every ``<Topic>``/``</Topic>`` marker, leaving the plain text."""
    return TAG.sub("", s)


def validate_topic_tags(rec: dict[str, Any]) -> list[str]:
    """Validate the dataset's paired-tag and open-tag-as-transition conventions.

    A new opening tag may implicitly end the previous span, and the final span may omit its
    closing tag. Those forms occur intentionally in the accepted dataset. A closing tag with no
    matching active topic, or an unknown tag, is invalid.
    """
    errors = []
    known = set(TOPICS)
    for field in ("prompt", "response"):
        active: str | None = None
        for match in _ANY_TAG.finditer(rec[field]):
            closing, name = match.groups()
            literal = match.group(0)
            if name not in known:
                errors.append(f"{field}: unrecognized tag {literal!r}")
                continue
            if not closing:
                active = name
            elif active is None:
                errors.append(f"{field}: closing tag {literal!r} has no opening tag")
            elif active != name:
                errors.append(
                    f"{field}: closing tag {literal!r} does not match <{active}>"
                )
                active = None
            else:
                active = None
    return errors


def _ids(x: Any) -> list[int]:
    """Normalize a tokenizer return (list[int] or a BatchEncoding) to a flat id list."""
    return x if (not x or isinstance(x[0], int)) else x[0].ids


def span_char_ranges(tagged: str) -> list[tuple[int, int, int]]:
    """Walk a tagged string, return [(stripped_start, stripped_end, topic_id)] per span, in order.

    The stripped positions index into strip_tags(tagged). Text OUTSIDE any tag (e.g. whitespace
    between </A> and <B>) is attributed to the most recent open span, or the next one if none yet.
    """
    pos = 0  # position in the STRIPPED string
    cur_topic = None
    out: list[list[int]] = []
    # Tokenize the tagged string into (text | tag) events, tracking stripped offsets.
    for m in re.finditer(r"</?(" + "|".join(re.escape(t) for t in TOPICS) + r")>|[^<]+|<", tagged):
        seg = m.group(0)
        tag_open = _OPEN.fullmatch(seg)
        tag_close = seg.startswith("</")
        if tag_open:
            cur_topic = TOPICS.index(tag_open.group(1))
            out.append([pos, pos, cur_topic])
        elif tag_close:
            cur_topic = None
        else:
            # plain text contributes to the stripped string
            seg_len = len(seg)
            if out and out[-1][2] is not None and cur_topic is not None:
                out[-1][1] = pos + seg_len
            elif out and cur_topic is None:
                # trailing/inter-span text: extend the previous span
                out[-1][1] = pos + seg_len
            # else: leading text before any tag is attributed to the first span (handled on open)
            pos += seg_len
    # merge: ensure contiguous coverage [0,total) mapped to nearest span
    return [(a, b, t) for a, b, t in out if b > a]


def _region(ranges):
    """Return (span_starts, first_start, topic_at) helpers for one tagged region's ranges."""
    starts = {a for a, b, t in ranges}
    first = ranges[0][0] if ranges else None

    def topic_at(ch: int) -> int:
        for a, b, t in ranges:
            if a <= ch < b:
                return t
        return ranges[-1][2] if ranges else PAD

    return starts, first, topic_at


def label_record(
    rec: dict[str, Any], tok, supervise_prompt: bool = False
) -> tuple[list[int], list[int], list[int]]:
    """Per-token labels for a tagged {prompt,response,topics} record.

    Default (supervise_prompt=False): RESPONSE-only supervision — response topic tokens labeled,
    everything else (prompt, template, neutral) PAD. This is the current/validated configuration.

    supervise_prompt=True: also label the prompt's own <topic>...</topic> spans (one per sub-request);
    template markers and inter-span neutral text stay PAD. The scripts were originally intended for
    this prompt+response mode; it is kept available behind the flag but OFF by default so response-to-
    response decode comparisons stay clean. roles: 1 = prompt/template tokens, 2 = response tokens.
    """
    prompt, response = strip_tags(rec["prompt"]), strip_tags(rec["response"])
    full = _ids(
        tok.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
            tokenize=True,
            add_generation_prompt=False,
        )
    )
    prefix = _ids(
        tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True
        )
    )
    prefix_str = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    nprefix = len(prefix)
    T = len(full)
    roles = [1] * nprefix + [2] * (T - nprefix)
    labels = [PAD] * T

    prompt_ranges = span_char_ranges(rec["prompt"])
    resp_ranges = span_char_ranges(rec["response"])
    p_starts, p_first, p_topic = _region(prompt_ranges)
    r_starts, r_first, r_topic = _region(resp_ranges)

    # Offsets on the TEMPLATED STRING (same tokenization as `full`). Response content begins exactly
    # at len(prefix_str) (validated). Prompt content is located by substring search; the chat template
    # trims leading/trailing whitespace, so we add back the stripped prompt's leading-whitespace count.
    templated = tok.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
        tokenize=False,
    )
    enc = tok(templated, add_special_tokens=False, return_offsets_mapping=True)
    ids2, offs = enc["input_ids"], enc["offset_mapping"]
    shift = 1 if (len(ids2) == T + 1 and ids2[1:] == list(full)) else 0

    resp_lo = len(prefix_str)
    resp_hi = resp_lo + len(response)
    p_content = prompt.strip()
    lead_p = len(prompt) - len(prompt.lstrip())
    p_lo = templated.find(p_content)
    p_hi = p_lo + len(p_content) if p_lo >= 0 else -1

    for k in range(len(ids2)):
        idx = k - shift
        if idx < 0 or idx >= T:
            continue
        s, e = offs[k]
        if resp_lo <= s < resp_hi:  # response CONTENT token (role 2)
            rel = s - resp_lo
            labels[idx] = PAD if (rel in r_starts and rel != r_first) else r_topic(rel)
        elif (
            supervise_prompt and idx < nprefix and p_lo >= 0 and p_lo <= s < p_hi
        ):  # prompt CONTENT token (role 1)
            rel = (s - p_lo) + lead_p
            labels[idx] = PAD if (rel in p_starts and rel != p_first) else p_topic(rel)
        # else: chat-template markers / inter-span neutral text -> stay PAD
    return full, roles, labels
