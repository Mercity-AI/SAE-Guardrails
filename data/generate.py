#!/usr/bin/env python3
"""
Topic-classification prompt generation pipeline.

Generates short user prompts spread across 28 subtopics (7 domains x 4 subtopics
each — see utils.SUBTOPICS) using the OpenAI Responses API. Each prompt is tagged
with its target subtopic/topic, adversarial level, and surface framing, then saved
as JSONL for downstream use in the SAE training pipeline (training/main.py).

Adversarial levels (0–4) control how much the target topic is camouflaged in the
question; higher levels produce prompts where only the LLM's response reveals the
true topic.

Pipeline:
  1. Read config.yaml for model settings, batch sizes, and adversarial levels
  2. Build a spec pool: target_per_subtopic specs per subtopic, each assigned
     an adversarial level drawn from the config
  3. Chunk the pool into batches of samples_per_call specs
  4. For each batch (concurrent): ask the model for one prompt per spec
  5. Parse each batch's JSON array, attach metadata, write to prompts.jsonl

Usage:
    1. Set OPENAI_API_KEY in environment or .env file
    2. Edit data/config.yaml to configure the run
    3. python data/generate.py
    4. Set dry_run: true in config.yaml to preview prompts without API calls
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import types
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

load_dotenv()

# Local imports here
from prompts import (
    ADVERSARIAL_LEVELS,
    PROMPTS_TAG,
    SYSTEM_PROMPT_TEMPLATE,
    build_adversarial_levels_block,
    build_taxonomy_block,
    build_user_prompt,
)
from utils import (
    SUBTOPICS,
    backoff_delay,
    build_temperature_plan,
    extract_json_array,
    is_flex_tier_issue,
    is_retryable_error,
    is_temperature_unsupported_error,
    parse_selection_mode,
    parse_temperature_weights,
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in environment")

SCRIPT_DIR = Path(__file__).resolve().parent


# ============================================================================
# SPEC SAMPLING
# ============================================================================


def build_spec_pool(
    rng: random.Random, args: types.SimpleNamespace
) -> list[dict[str, Any]]:
    """Build the full list of generation specs: target_per_subtopic per subtopic.

    Each spec carries the target subtopic, its parent topic, and a randomly
    sampled adversarial level drawn from args.adversarial_levels. The pool is
    shuffled so batches mix subtopics and levels rather than running one at a time.
    """
    level_pool = args.adversarial_levels
    specs: list[dict[str, Any]] = []
    for subtopic, info in SUBTOPICS.items():
        for _ in range(args.target_per_subtopic):
            specs.append(
                {
                    "subtopic": subtopic,
                    "topic": info["topic"],
                    "target_adversarial_level": rng.choice(level_pool),
                }
            )
    rng.shuffle(specs)
    return specs


def chunk_specs(specs: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    """Split the spec pool into batches of at most `size` specs each."""
    return [specs[i : i + size] for i in range(0, len(specs), size)]


def build_system_prompt() -> str:
    """Render the full system prompt with taxonomy and adversarial levels filled in."""
    return SYSTEM_PROMPT_TEMPLATE.format(
        taxonomy=build_taxonomy_block(),
        adversarial_levels=build_adversarial_levels_block(),
        tag=PROMPTS_TAG,
    )


# ============================================================================
# OPENAI CALL  (retry / backoff / service-tier fallback)
# ============================================================================


def safe_model_dump(obj: Any) -> dict[str, Any]:
    """Best-effort conversion of an SDK response object to a plain dict."""
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except TypeError:
            return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return {"raw": str(obj)}


def extract_output_text(resp_obj: Any, resp_json: dict[str, Any]) -> str:
    """
    Pull the assistant text out of a Responses API result.

    Prefers the convenience `output_text`; otherwise walks the `output` list and
    concatenates text pieces from `message`-type items (ignores reasoning items).
    """
    output_text = getattr(resp_obj, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    outputs = resp_json.get("output")
    if not isinstance(outputs, list):
        return ""
    parts: list[str] = []
    for item in outputs:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if isinstance(content, list):
            for piece in content:
                if isinstance(piece, dict) and isinstance(piece.get("text"), str):
                    if piece["text"].strip():
                        parts.append(piece["text"].strip())
        elif isinstance(content, str) and content.strip():
            parts.append(content.strip())
    return "\n".join(parts).strip()


def extract_reasoning_text(resp_json: dict[str, Any]) -> str | None:
    """
    Pull reasoning summary text out of a Responses API result.

    Walks the `output` list for `reasoning`-type items and concatenates all
    `summary_text` pieces in order. Returns None when no reasoning output is present
    (e.g. reasoning_summary: none, or a non-reasoning model).
    """
    outputs = resp_json.get("output")
    if not isinstance(outputs, list):
        return None
    parts: list[str] = []
    for item in outputs:
        if not isinstance(item, dict) or item.get("type") != "reasoning":
            continue
        # Each reasoning item has a `summary` list of {type: "summary_text", text: "..."}
        for chunk in item.get("summary", []):
            if isinstance(chunk, dict) and isinstance(chunk.get("text"), str):
                if chunk["text"].strip():
                    parts.append(chunk["text"].strip())
    return "\n\n".join(parts) if parts else None


async def call_with_tier_fallback(
    client: AsyncOpenAI,
    request_input: list[dict[str, Any]],
    args: types.SimpleNamespace,
    temperature_value: float | None,
) -> tuple[Any, float | None]:
    """
    Call the Responses API, retrying transient errors and falling back from the
    primary service tier (flex) to the fallback tier (default) when needed.

    Returns (response, effective_temperature). effective_temperature may differ
    from temperature_value if the model rejected it as unsupported (reasoning
    models return a 400 for temperature -- we drop it and retry transparently).
    Raises if every tier/retry is exhausted.
    """
    effective_temperature = temperature_value

    tiers = [args.primary_tier]
    if args.fallback_tier and args.fallback_tier != args.primary_tier:
        tiers.append(args.fallback_tier)

    last_error: Exception | None = None
    for tier_idx, tier in enumerate(tiers):
        for attempt in range(args.max_retries_per_tier + 1):
            try:
                resp = await client.responses.create(
                    model=args.model,
                    input=request_input,
                    service_tier=tier,
                    max_output_tokens=args.max_output_tokens,
                    reasoning={
                        "effort": args.reasoning_effort,
                        "summary": args.reasoning_summary,
                    },
                    **(
                        {"temperature": effective_temperature}
                        if effective_temperature is not None
                        else {}
                    ),
                )
                return resp, effective_temperature
            except (
                Exception
            ) as exc:  # broad catch: keep one bad batch from killing the run
                last_error = exc
                if (
                    effective_temperature is not None
                    and is_temperature_unsupported_error(exc)
                ):
                    effective_temperature = None
                    continue
                if (
                    tier == args.primary_tier
                    and tier_idx == 0
                    and args.fallback_tier
                    and is_flex_tier_issue(exc)
                    and attempt == 0
                ):
                    print(
                        f"[tier-fallback] {args.primary_tier} -> {args.fallback_tier}: {exc}"
                    )
                    break
                if is_retryable_error(exc) and attempt < args.max_retries_per_tier:
                    await asyncio.sleep(
                        backoff_delay(attempt, args.backoff_min, args.backoff_max)
                    )
                    continue
                if tier_idx + 1 < len(tiers):
                    break  # this tier exhausted -> try next tier
                raise

    if last_error is not None:
        raise last_error
    raise RuntimeError("request_failed_without_exception")


# ============================================================================
# GENERATION  (one batch of specs -> one API call -> one record per spec)
# ============================================================================


async def run_one_batch(
    client: AsyncOpenAI,
    batch_index: int,
    system_prompt: str,
    specs: list[dict[str, Any]],
    args: types.SimpleNamespace,
    temperature_value: float | None,
) -> list[dict[str, Any]]:
    """Send one batch of specs to the model and return one record per spec.

    On success, each record carries the generated `prompt` string and
    `_status: "ok"`. On request failure, parse failure, or count mismatch,
    every record in the batch carries `prompt: None` and a `_status` describing
    the failure so post-processing can account for every requested spec. A
    per-item bad entry (missing/empty "prompt") only marks that one record.
    """
    request_id = uuid.uuid4().hex
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_prompt(specs)},
    ]

    def build_records(
        prompt_texts: list[str | None],
        statuses: list[str],
        raw_output: str | None = None,
        reasoning: str | None = None,
    ) -> list[dict[str, Any]]:
        records = []
        for spec, prompt_text, status in zip(specs, prompt_texts, statuses):
            record = dict(spec)
            record["prompt"] = prompt_text
            record["_status"] = status
            gen_meta: dict[str, Any] = {
                "batch_index": batch_index,
                "request_id": request_id,
                "temperature": temperature_value,
                "model": args.model,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            }
            # Only attach raw_output/reasoning on failures — for ok records the
            # parsed prompt is the output; storing the full batch response on every
            # record would duplicate it N times and bloat the file.
            if status != "ok":
                gen_meta["raw_output"] = raw_output
                gen_meta["reasoning"] = reasoning
            record["_gen"] = gen_meta
            records.append(record)
        return records

    try:
        response, effective_temp = await call_with_tier_fallback(
            client, messages, args, temperature_value
        )
        # effective_temp reflects what the API actually accepted (reasoning models
        # may reject the configured temperature) -- record that, not the plan value.
        temperature_value = effective_temp
    except Exception as e:
        print(f"[batch {batch_index}] request failed: {e}")
        return build_records([None] * len(specs), ["request_failed"] * len(specs))

    resp_json = safe_model_dump(response)
    raw_output = extract_output_text(response, resp_json)
    reasoning = extract_reasoning_text(resp_json)
    parsed, parse_err = extract_json_array(raw_output, PROMPTS_TAG)

    if parse_err:
        print(f"[batch {batch_index}] parse error: {parse_err}")
        return build_records(
            [None] * len(specs),
            [f"parse_error:{parse_err}"] * len(specs),
            raw_output,
            reasoning,
        )

    if len(parsed) != len(specs):
        print(
            f"[batch {batch_index}] count mismatch: expected {len(specs)}, got {len(parsed)}"
        )
        return build_records(
            [None] * len(specs), ["count_mismatch"] * len(specs), raw_output, reasoning
        )

    prompt_texts: list[str | None] = []
    statuses: list[str] = []
    adversarial_levels: list[int | None] = []
    surface_topics: list[str | None] = []
    for item in parsed:
        if (
            isinstance(item, dict)
            and isinstance(item.get("prompt"), str)
            and item["prompt"].strip()
        ):
            prompt_texts.append(item["prompt"].strip())
            statuses.append("ok")
            raw_level = item.get("adversarial_level")
            adversarial_levels.append(
                int(raw_level) if isinstance(raw_level, (int, float)) else None
            )
            raw_surface = item.get("surface_topic")
            surface_topics.append(
                raw_surface.strip()
                if isinstance(raw_surface, str) and raw_surface.strip()
                else None
            )
        else:
            prompt_texts.append(None)
            statuses.append("bad_item")
            adversarial_levels.append(None)
            surface_topics.append(None)

    records = build_records(prompt_texts, statuses, raw_output, reasoning)
    for rec, level, surface in zip(records, adversarial_levels, surface_topics):
        rec["adversarial_level"] = level
        rec["surface_topic"] = surface
    return records


async def run_generation(
    system_prompt: str, args: types.SimpleNamespace, out_path: Path
) -> dict[str, Any]:
    """
    Orchestrate all batches concurrently (bounded by a semaphore), stream every
    produced record to `out_path` as JSONL, and return a summary.
    """
    client = AsyncOpenAI(api_key=OPENAI_API_KEY, timeout=args.timeout, max_retries=0)

    rng = random.Random(args.seed)
    specs = build_spec_pool(rng, args)
    batches = chunk_specs(specs, args.samples_per_call)
    temperature_plan = build_temperature_plan(args, len(batches))

    print(
        f"Specs   : {len(specs)} ({args.target_per_subtopic} per subtopic x {len(SUBTOPICS)} subtopics)"
    )
    print(f"Batches : {len(batches)} (samples_per_call={args.samples_per_call})")

    sem = asyncio.Semaphore(args.max_concurrent)

    async def worker(idx: int) -> list[dict[str, Any]]:
        async with sem:
            return await run_one_batch(
                client, idx, system_prompt, batches[idx], args, temperature_plan[idx]
            )

    tasks = [asyncio.create_task(worker(i)) for i in range(len(batches))]

    totals = Counter()
    ok_by_subtopic = Counter()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fout:
        pbar = tqdm(total=len(tasks), desc="Batches", unit="batch", dynamic_ncols=True)
        for coro in asyncio.as_completed(tasks):
            records = await coro
            for r in records:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                fout.flush()
                totals["total"] += 1
                totals[r["_status"]] += 1
                if r["_status"] == "ok":
                    ok_by_subtopic[r["subtopic"]] += 1
            pbar.update(1)
            pbar.set_postfix_str(
                f"ok={totals['ok']} bad={totals['total'] - totals['ok']}", refresh=False
            )
        pbar.close()

    return {
        "totals": dict(totals),
        "ok_by_subtopic": dict(ok_by_subtopic),
        "num_specs": len(specs),
        "num_batches": len(batches),
    }


# ============================================================================
# MAIN
# ============================================================================


def load_config() -> types.SimpleNamespace:
    """Load generation config from config.yaml, resolving local paths relative to the script directory."""
    config_path = SCRIPT_DIR / "config.yaml"
    with config_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    temperatures = data.get("temperatures")
    if temperatures is not None:
        if not isinstance(temperatures, list) or not temperatures:
            raise ValueError("temperatures must be a non-empty list when provided")
        data["temperature_selection_mode"] = parse_selection_mode(
            data.get("temperature_selection_mode", "random"),
            "temperature_selection_mode",
        )
        data["temperatures"] = [float(value) for value in temperatures]
        data["temperature_weights"] = parse_temperature_weights(
            data.get("temperature_weights"), len(data["temperatures"])
        )
    elif data.get("temperature") is not None:
        data["temperature"] = float(data["temperature"])

    data["target_per_subtopic"] = max(1, int(data.get("target_per_subtopic", 1)))
    data["samples_per_call"] = max(1, int(data.get("samples_per_call", 1)))

    raw_levels = data.get("adversarial_levels", list(ADVERSARIAL_LEVELS.keys()))
    if not isinstance(raw_levels, list) or not raw_levels:
        raise ValueError("adversarial_levels must be a non-empty list")
    invalid = [lv for lv in raw_levels if lv not in ADVERSARIAL_LEVELS]
    if invalid:
        raise ValueError(f"adversarial_levels contains unknown levels: {invalid}")
    data["adversarial_levels"] = [int(lv) for lv in raw_levels]

    ns = types.SimpleNamespace(**data)
    ns.output_dir = SCRIPT_DIR / ns.output_dir
    return ns


def main() -> None:
    args = load_config()
    system_prompt = build_system_prompt()

    if args.dry_run:
        rng = random.Random(args.seed)
        preview_specs = build_spec_pool(rng, args)[: args.samples_per_call]
        print("=" * 70 + "\nSYSTEM PROMPT\n" + "=" * 70)
        print(system_prompt)
        print(f"\n{'=' * 70}\nUSER PROMPT ({len(preview_specs)} spec(s))\n{'=' * 70}")
        print(build_user_prompt(preview_specs))
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"{args.run_directory}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = run_dir / "prompts.jsonl"
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"

    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config["run_dir"] = str(run_dir)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"Run dir : {run_dir}")
    print(f"Model   : {args.model}  (reasoning={args.reasoning_effort})")
    print(f"Tiers   : primary={args.primary_tier} fallback={args.fallback_tier}")

    summary = asyncio.run(run_generation(system_prompt, args, prompts_path))
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    t = summary["totals"]
    print("\n=== Done ===")
    print(
        f"total={t.get('total', 0)}  ok={t.get('ok', 0)}  bad={t.get('total', 0) - t.get('ok', 0)}"
    )
    print(f"ok by subtopic: {summary['ok_by_subtopic']}")
    print(f"\nArtifacts:\n  prompts: {prompts_path}\n  summary: {summary_path}")


if __name__ == "__main__":
    main()
