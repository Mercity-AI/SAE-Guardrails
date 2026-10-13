#!/usr/bin/env python3
"""Abstention experiment A -- off-topic injection over the 1500-sample eval set.

For each eval record we bundle the tagged PROMPT + RESPONSE, ask the editor model (via OpenRouter)
to slip substantial off-topic asides -- each wrapped in a <Joke>...</Joke> or <Neutral>...</Neutral>
tag -- into the gaps BETWEEN different topic chunks (never inside a chunk, never altering tagged
text), and store the original + modified bundle plus the parsed asides. Downstream the cache
pipeline strips the topic tags and re-encodes the modified text; the Joke/Neutral spans are the
abstention targets -- text the detector should not confidently assign to any of the 7 topics.

This script ONLY produces the injected eval set. Re-encoding + scoring is a separate step.

The defaults below are the recorded run; the command-line flags override the ones that vary
between runs. --dry-run prints the system prompt and one rendered user prompt without any API
calls. Output is resumable: rows already in the output file (by index) are skipped on a re-run.

    python evals/perturbations/insert_offtopic.py [--limit 5] [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from common.paths import PROJECT_ROOT
from evals.perturbations.injection_prompts import (
    INJECT_CATEGORIES,
    PAYLOAD_TAG,
    SYSTEM_PROMPT,
    TOPICS,
    build_user_prompt,
)

# ============================================================================
# CONFIG  (defaults for the recorded run; main() overrides them from the command line)
# ============================================================================

REPO_ROOT = PROJECT_ROOT

# --- data ---
MANIFEST = Path(__file__).resolve().parent / "manifest_test_full1500.jsonl"  # ships with the code
DATASET = REPO_ROOT / "runs/GPT_5.6_10k_2108/dataset.final.jsonl"  # response joined by index
OUTPUT = REPO_ROOT / "ablation/abstention/inject_offtopic_1500.jsonl"

# --- what to inject where ---
INJECT_TARGET = "both"  # "response" | "prompt" | "both"; A scores response tokens
LIMIT = 0  # 0 = all 1500; small value for a smoke test

# --- OpenRouter transport ---
BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV = "OPENROUTER_API_KEY"
MODEL = "openai/gpt-5.6-terra"
TEMPERATURE: float | None = None  # None = omit (reasoning models reject it); mechanical task
MAX_TOKENS = 100000
# OpenRouter passthrough. Prefer the cheapest provider route (the "flex"-ish option here) and
# ask for a reasoning trace so we can capture the editor's placement rationale.
EXTRA_BODY: dict[str, Any] = {
    "provider": {"sort": "price"},
    "reasoning": {"effort": "medium"},
}
EXTRA_HEADERS: dict[str, str] = {
    "HTTP-Referer": "https://local/abstention-inject",
    "X-Title": "abstention-offtopic-injection",
}

# --- concurrency / retries ---
MAX_CONCURRENT = 96
MAX_RETRIES = 5
BACKOFF_MIN = 2.0
BACKOFF_MAX = 45.0
REQUEST_TIMEOUT = 180.0
SEED = 42

DRY_RUN = False

# 7-topic tag matcher, built exactly like the cache pipeline (escaped, close-or-open).
TAG_RE = re.compile(r"</?(?:" + "|".join(re.escape(t) for t in TOPICS) + r")>")
SPAN_RE = re.compile(r"<(" + "|".join(re.escape(t) for t in TOPICS) + r")>(.*?)</\1>", re.DOTALL)
# Injected-aside matcher: <Joke>...</Joke> / <Neutral>...</Neutral> (the abstention spans).
INJECT_TAG_RE = re.compile(
    r"<(" + "|".join(re.escape(c) for c in INJECT_CATEGORIES) + r")>(.*?)</\1>", re.DOTALL
)


# ============================================================================
# LOADING
# ============================================================================


def load_records() -> list[dict[str, Any]]:
    """Load the 1500 eval manifest and join each prompt with its tagged response by dataset index."""
    manifest = [json.loads(line) for line in MANIFEST.open(encoding="utf-8")]
    dataset = [json.loads(line) for line in DATASET.open(encoding="utf-8")]
    records: list[dict[str, Any]] = []
    for row in manifest:
        idx = int(row["index"])
        records.append(
            {
                "index": idx,
                "topics": row["topics"],
                "single_topic": row["single_topic"],
                "assumed_label_id": row["assumed_label_id"],
                "prompt": row["prompt"],
                "response": dataset[idx]["record"]["response"],
            }
        )
    return records


def already_done(path: Path) -> set[int]:
    """Return indices whose LAST recorded status is 'ok' (resume retries non-ok rows on re-run)."""
    if not path.exists():
        return set()
    last_status: dict[int, str] = {}
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            last_status[int(row["index"])] = row.get("_status", "")
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    return {idx for idx, status in last_status.items() if status == "ok"}


# ============================================================================
# API CALL  (retry / backoff)
# ============================================================================


def backoff_delay(attempt_index: int) -> float:
    """Exponential backoff with jitter: BACKOFF_MIN * 2^attempt + U(0, BACKOFF_MIN), capped."""
    base = min(BACKOFF_MAX, BACKOFF_MIN * (2**attempt_index))
    return min(BACKOFF_MAX, base + random.uniform(0, BACKOFF_MIN))


def is_retryable(exc: Exception) -> bool:
    """True if the exception is a transient error worth retrying."""
    if isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)):
        return True
    msg = str(exc).lower()
    return any(
        tok in msg
        for tok in (
            "429",
            "rate limit",
            "timeout",
            "timed out",
            "overloaded",
            "service unavailable",
            "internal server",
            "temporar",
        )
    )


def extract_reasoning(resp: Any) -> str:
    """Pull the editor model's reasoning trace out of an OpenRouter chat response, or '' if none."""
    try:
        message = resp.choices[0].message
    except (AttributeError, IndexError):
        return ""
    # OpenRouter surfaces reasoning as message.reasoning and/or message.reasoning_details.
    reasoning = getattr(message, "reasoning", None)
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()
    details = getattr(message, "reasoning_details", None)
    if isinstance(details, list):
        parts = [d.get("text", "") for d in details if isinstance(d, dict)]
        joined = "\n".join(p for p in parts if p)
        if joined.strip():
            return joined.strip()
    return ""


async def call_model(client: AsyncOpenAI, user_prompt: str) -> tuple[str, str]:
    """Send one system+user chat request to OpenRouter, retrying transient errors; return (text, reasoning)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    kwargs: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "extra_body": EXTRA_BODY,
        "extra_headers": EXTRA_HEADERS,
    }
    if TEMPERATURE is not None:
        kwargs["temperature"] = TEMPERATURE

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = await client.chat.completions.create(**kwargs)
            return (resp.choices[0].message.content or "").strip(), extract_reasoning(resp)
        except Exception as exc:  # broad: keep one bad record from killing the run
            last_error = exc
            if is_retryable(exc) and attempt < MAX_RETRIES:
                await asyncio.sleep(backoff_delay(attempt))
                continue
            raise
    raise last_error if last_error else RuntimeError("request_failed_without_exception")


# ============================================================================
# PARSE + VERIFY  (lexical only; flag, never repair)
# ============================================================================


def extract_payload(raw_text: str) -> dict[str, Any] | None:
    """Pull the JSON object out of <INJECTED>...</INJECTED>, falling back to the first {..last }."""
    if not raw_text:
        return None
    match = re.search(rf"<{PAYLOAD_TAG}>\s*(.*?)\s*</{PAYLOAD_TAG}>", raw_text, re.DOTALL | re.IGNORECASE)
    candidate = match.group(1) if match else None
    if candidate is None:
        start, end = raw_text.find("{"), raw_text.rfind("}")
        if start == -1 or end <= start:
            return None
        candidate = raw_text[start : end + 1]
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def span_char_ranges(text: str) -> list[tuple[int, int]]:
    """Character (start, end) ranges of each <Topic>...</Topic> span, tags included."""
    return [(m.start(), m.end()) for m in SPAN_RE.finditer(text)]


def spans_preserved(original: str, modified: str) -> bool:
    """True if every original <Topic>...</Topic> block survives verbatim and in order in modified."""
    orig_spans = [m.group(0) for m in SPAN_RE.finditer(original)]
    cursor = 0
    for span in orig_spans:
        found = modified.find(span, cursor)
        if found == -1:
            return False
        cursor = found + len(span)
    return True


def tag_topics(text: str) -> list[str]:
    """Ordered list of the real topics whose <Topic> spans appear in the text (opening tags)."""
    return [m.group(1) for m in SPAN_RE.finditer(text)]


def parse_injections(modified: str, segment: str) -> tuple[list[dict[str, Any]], bool]:
    """Parse <Joke>/<Neutral> asides from one edited segment; return (asides, any_inside_topic_span).

    Each aside is {segment, type, text}. `any_inside_topic_span` is True if a Joke/Neutral tag sits
    inside a real topic span (an infill -- forbidden; must be a gap between chunks).
    """
    topic_ranges = span_char_ranges(modified)
    asides: list[dict[str, Any]] = []
    inside = False
    for m in INJECT_TAG_RE.finditer(modified):
        start, end = m.start(), m.end()
        if any(not (end <= s or start >= e) for s, e in topic_ranges):
            inside = True
        asides.append({"segment": segment, "type": m.group(1), "text": m.group(2).strip()})
    return asides, inside


def verify(record: dict[str, Any], payload: dict[str, Any] | None) -> tuple[dict[str, Any], str]:
    """Run lexical checks on the model output; return (annotated_row, status). Never repairs."""
    row = {
        "index": record["index"],
        "topics": record["topics"],
        "single_topic": record["single_topic"],
        "assumed_label_id": record["assumed_label_id"],
        "orig_prompt": record["prompt"],
        "orig_response": record["response"],
    }
    if payload is None:
        return {**row, "_status": "parse_error"}, "parse_error"

    mod_prompt = payload.get("modified_prompt")
    mod_response = payload.get("modified_response")
    if not isinstance(mod_prompt, str) or not isinstance(mod_response, str):
        return {**row, "_status": "bad_fields", "_raw_payload": payload}, "bad_fields"

    # the 7 topic chunks must survive byte-for-byte and in order in every segment
    if not spans_preserved(record["prompt"], mod_prompt):
        return {**row, "_status": "prompt_span_altered", "inj_prompt": mod_prompt}, "prompt_span_altered"
    if not spans_preserved(record["response"], mod_response):
        return {
            **row,
            "_status": "response_span_altered",
            "inj_response": mod_response,
        }, "response_span_altered"

    # authoritative injections = the <Joke>/<Neutral> tags actually present in the edited text
    prompt_asides, p_inside = parse_injections(mod_prompt, "prompt")
    response_asides, r_inside = parse_injections(mod_response, "response")
    if p_inside or r_inside:
        return (
            {
                **row,
                "_status": "injection_inside_span",
                "inj_prompt": mod_prompt,
                "inj_response": mod_response,
            },
            "injection_inside_span",
        )

    # every targeted segment must actually receive at least one aside
    if INJECT_TARGET in ("response", "both") and not response_asides:
        return {
            **row,
            "_status": "response_no_injection",
            "inj_response": mod_response,
        }, "response_no_injection"
    if INJECT_TARGET in ("prompt", "both") and not prompt_asides:
        return {**row, "_status": "prompt_no_injection", "inj_prompt": mod_prompt}, "prompt_no_injection"

    injections = prompt_asides + response_asides

    # original vs new topic tags: the real-topic tag sequence must be unchanged after injection
    orig_tag_topics = tag_topics(record["prompt"]) + tag_topics(record["response"])
    new_tag_topics = tag_topics(mod_prompt) + tag_topics(mod_response)
    topics_block = {
        "orig_topics": record["topics"],
        "orig_tag_topics": orig_tag_topics,
        "new_tag_topics": new_tag_topics,
        "tag_topics_match": orig_tag_topics == new_tag_topics,
        "injected_categories": sorted({inj["type"] for inj in injections}),
        "n_injections": len(injections),
        "injected_words": sum(len(inj["text"].split()) for inj in injections),
    }
    if not topics_block["tag_topics_match"]:
        return {
            **row,
            **topics_block,
            "inj_prompt": mod_prompt,
            "inj_response": mod_response,
            "injections": injections,
            "_status": "tag_topics_changed",
        }, "tag_topics_changed"

    return (
        {
            **row,
            **topics_block,
            "plan": payload.get("plan", ""),
            "model_injections": payload.get("injections", []),
            "inj_prompt": mod_prompt,
            "inj_response": mod_response,
            "injections": injections,
            "_status": "ok",
        },
        "ok",
    )


# ============================================================================
# ORCHESTRATION
# ============================================================================


async def process_record(
    client: AsyncOpenAI, record: dict[str, Any], sem: asyncio.Semaphore
) -> dict[str, Any]:
    """Inject off-topic text into one record and verify it; returns the annotated output row."""
    user_prompt = build_user_prompt(record["prompt"], record["response"], record["topics"], INJECT_TARGET)
    async with sem:
        try:
            raw, reasoning = await call_model(client, user_prompt)
        except Exception as exc:
            return {
                "index": record["index"],
                "topics": record["topics"],
                "orig_prompt": record["prompt"],
                "orig_response": record["response"],
                "_status": "request_failed",
                "_error_type": type(exc).__name__,
                "_error": str(exc)[:1000],
                "_meta": {
                    "model": MODEL,
                    "target": INJECT_TARGET,
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                },
            }
    payload = extract_payload(raw)
    row, status = verify(record, payload)
    # keep the full raw model text whenever parsing/verification did not cleanly succeed
    if status != "ok":
        row["_raw_output"] = raw
    row["reasoning"] = reasoning
    row["_meta"] = {
        "model": MODEL,
        "target": INJECT_TARGET,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    return row


async def run(records: list[dict[str, Any]]) -> dict[str, int]:
    """Run all records concurrently (bounded), streaming each result to OUTPUT; return status counts."""
    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        raise ValueError(f"{API_KEY_ENV} not set in environment")
    client = AsyncOpenAI(api_key=api_key, base_url=BASE_URL, timeout=REQUEST_TIMEOUT, max_retries=0)
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    tasks = [asyncio.create_task(process_record(client, r, sem)) for r in records]
    counts: dict[str, int] = {}
    done = 0
    with OUTPUT.open("a", encoding="utf-8") as fout:
        for coro in asyncio.as_completed(tasks):
            row = await coro
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            counts[row["_status"]] = counts.get(row["_status"], 0) + 1
            done += 1
            if done % 25 == 0 or done == len(tasks):
                ok = counts.get("ok", 0)
                print(f"{done}/{len(tasks)}  ok={ok}  other={done - ok}", flush=True)
    return counts


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    """Entry point: load eval records, (dry-run preview or) inject off-topic text, write JSONL."""
    global \
        MANIFEST, \
        DATASET, \
        OUTPUT, \
        INJECT_TARGET, \
        LIMIT, \
        MODEL, \
        MAX_CONCURRENT, \
        SEED, \
        DRY_RUN, \
        BASE_URL, \
        API_KEY_ENV
    parser = argparse.ArgumentParser(description="Insert tagged off-topic asides into the test records.")
    parser.add_argument("--manifest", default=str(MANIFEST), help="1,500-record test manifest")
    parser.add_argument("--dataset", default=str(DATASET), help="dataset.final.jsonl (responses by index)")
    parser.add_argument("--output", default=str(OUTPUT), help="injected JSONL to write (resumable)")
    parser.add_argument("--target", choices=["response", "prompt", "both"], default=INJECT_TARGET)
    parser.add_argument(
        "--limit", type=int, default=LIMIT, help="0 = all records; small value for a smoke test"
    )
    parser.add_argument("--model", default=MODEL, help="editor model on the OpenAI-compatible endpoint")
    parser.add_argument("--base-url", default=BASE_URL, help="any OpenAI-compatible endpoint")
    parser.add_argument("--api-key-env", default=API_KEY_ENV, help="environment variable holding the API key")
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true", help="print the prompts, make no API calls")
    args = parser.parse_args()

    def resolve(path: str) -> Path:
        return Path(path) if Path(path).is_absolute() else REPO_ROOT / path

    MANIFEST, DATASET, OUTPUT = resolve(args.manifest), resolve(args.dataset), resolve(args.output)
    INJECT_TARGET, LIMIT, MODEL = args.target, args.limit, args.model
    BASE_URL, API_KEY_ENV = args.base_url, args.api_key_env
    MAX_CONCURRENT, SEED, DRY_RUN = args.max_concurrent, args.seed, args.dry_run

    random.seed(SEED)
    records = load_records()
    if LIMIT:
        records = records[:LIMIT]

    if DRY_RUN:
        sample = records[0]
        print("=" * 70 + "\nSYSTEM PROMPT\n" + "=" * 70)
        print(SYSTEM_PROMPT)
        print(
            "\n" + "=" * 70 + f"\nUSER PROMPT (index {sample['index']}, target={INJECT_TARGET})\n" + "=" * 70
        )
        print(build_user_prompt(sample["prompt"], sample["response"], sample["topics"], INJECT_TARGET))
        print("\n" + "=" * 70)
        print(f"records={len(records)}  output={OUTPUT}  (DRY_RUN: no API calls made)")
        return

    done = already_done(OUTPUT)
    if done:
        before = len(records)
        records = [r for r in records if r["index"] not in done]
        print(f"resume: {len(done)} already done, {len(records)}/{before} remaining")

    if not records:
        print("nothing to do")
        return

    print(f"model={MODEL} via {BASE_URL}  target={INJECT_TARGET}  records={len(records)}  -> {OUTPUT}")
    counts = asyncio.run(run(records))
    print("\n=== done ===")
    for status, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {status:24s} {n}")
    print(f"output: {OUTPUT}")


if __name__ == "__main__":
    main()
