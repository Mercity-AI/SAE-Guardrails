"""
Shared constants and helpers for the topic-classification eval-prompt generator.

This module holds the parts of the pipeline that are reused or purely mechanical:
  * the subtopic taxonomy (28 subtopics across 7 domains) and the adversarial
    framing modes,
  * the weighted-sampling helpers (including confound-subtopic selection),
  * the OpenAI retry / service-tier predicates and backoff,
  * the model-output JSON extraction.

Prompt construction (system/user prompt templates) lives in prompts.py; threading
and orchestration live in main.py.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

ALLOWED_SELECTION_MODES = {"random", "uniform", "weighted"}

# ============================================================================
# SUBTOPIC TAXONOMY
#
# 28 leaf labels across 7 domains. Each becomes one training class for the
# downstream SAE topic classifier. The domain name is the high-level topic;
# the subtopic key is the fine-grained label.
# ============================================================================

SUBTOPICS: dict[str, dict[str, str]] = {
    # -- Enterprise documents -------------------------------------------------
    "Internal records & memos": {
        "topic": "Enterprise documents",
        "description": (
            "Intra-company documents such as board meeting minutes, executive "
            "memos, internal announcements, and organizational policy handbooks — "
            "records created and consumed inside a business, not shared externally."
        ),
    },
    "Operational procedures & guides": {
        "topic": "Enterprise documents",
        "description": (
            "Process documentation including standard operating procedures (SOPs), "
            "IT runbooks, system administration guides, employee how-to manuals, "
            "and technical reference documentation for internal operations."
        ),
    },
    "Project & planning documents": {
        "topic": "Enterprise documents",
        "description": (
            "Forward-looking business artifacts such as project proposals, business "
            "cases, product roadmaps, OKR trackers, project status reports, and "
            "post-mortem write-ups that plan, track, or review work."
        ),
    },
    "Commercial transaction records": {
        "topic": "Enterprise documents",
        "description": (
            "B2B paperwork that records or authorizes a commercial exchange: invoices, "
            "purchase orders, receipts, vendor quotes, rate cards, and statements of "
            "work accompanying the transfer of goods or services between businesses."
        ),
    },
    # -- General news & content -----------------------------------------------
    "Current affairs & geopolitics": {
        "topic": "General news & content",
        "description": (
            "Breaking news, election coverage, government decisions, diplomatic "
            "negotiations, military conflicts, and international relations as "
            "reported in the press and broadcast media."
        ),
    },
    "Science & technology news": {
        "topic": "General news & content",
        "description": (
            "Newly published research findings, technology product launches, space "
            "exploration milestones, AI and software developments, and cybersecurity "
            "incidents as covered in news and science journalism."
        ),
    },
    "Culture & sports": {
        "topic": "General news & content",
        "description": (
            "Sports match results and athlete profiles, film and music releases, "
            "celebrity news, awards coverage, and pop-culture commentary."
        ),
    },
    "Environment & society": {
        "topic": "General news & content",
        "description": (
            "Climate change reporting, environmental activism, social justice movements, "
            "education news, immigration stories, and community-level human-interest "
            "features as covered by journalists."
        ),
    },
    # -- Customer service -----------------------------------------------------
    "Billing & payment disputes": {
        "topic": "Customer service",
        "description": (
            "Support interactions about unexpected charges, invoice errors, failed "
            "payments, overcharges, refund eligibility, and disputes over what a "
            "customer was billed by a company."
        ),
    },
    "Account access & management": {
        "topic": "Customer service",
        "description": (
            "Support interactions about logging in, password resets, two-factor "
            "authentication issues, profile settings, subscription tier changes, "
            "and managing account preferences with a service provider."
        ),
    },
    "Technical support & troubleshooting": {
        "topic": "Customer service",
        "description": (
            "Support interactions about diagnosing and fixing product or software "
            "malfunctions, setup failures, compatibility problems, and error messages "
            "encountered while using a product or service."
        ),
    },
    "Shipping, returns & refunds": {
        "topic": "Customer service",
        "description": (
            "Support interactions about delivery status, delayed or lost packages, "
            "damaged goods, initiating a product return, processing a refund, and "
            "complaint escalation after a failed purchase experience."
        ),
    },
    # -- Legal ----------------------------------------------------------------
    "Contract clause analysis": {
        "topic": "Legal",
        "description": (
            "Legal interpretation of specific provisions in contracts: liability "
            "caps, indemnification obligations, IP ownership and assignment, "
            "non-compete and non-solicitation, force majeure, and arbitration clauses."
        ),
    },
    "Regulatory & data compliance": {
        "topic": "Legal",
        "description": (
            "Obligations imposed by data protection laws (GDPR, CCPA), privacy policy "
            "drafting requirements, consumer protection regulations, and the compliance "
            "programs organizations must implement to meet statutory duties."
        ),
    },
    "Employment & labor law": {
        "topic": "Legal",
        "description": (
            "Workers' legal rights, wrongful termination and discrimination claims, "
            "wage and hour disputes, union and collective bargaining rules, and "
            "employer obligations under employment statutes."
        ),
    },
    "Intellectual property & licensing": {
        "topic": "Legal",
        "description": (
            "Patents, trademarks, copyrights, trade secrets, software licensing terms, "
            "IP infringement claims, and legal frameworks governing ownership and "
            "commercialization of creative or inventive work."
        ),
    },
    # -- Financial ------------------------------------------------------------
    "Investor & market communications": {
        "topic": "Financial",
        "description": (
            "Earnings calls, shareholder letters, analyst equity research notes, "
            "forward guidance statements, IPO prospectuses, and investor-facing "
            "disclosures produced by public companies and financial analysts."
        ),
    },
    "Personal banking & credit": {
        "topic": "Financial",
        "description": (
            "Savings and checking accounts, personal loans, mortgage applications, "
            "credit score improvement, credit card management, and retail banking "
            "products for individual consumers."
        ),
    },
    "Insurance & risk management": {
        "topic": "Financial",
        "description": (
            "Insurance policy terms, premium calculations, claims filing and disputes, "
            "actuarial risk assessment, liability coverage decisions, and risk "
            "management frameworks for individuals and organizations."
        ),
    },
    "Tax & accounting": {
        "topic": "Financial",
        "description": (
            "Corporate and personal tax filing, bookkeeping practices, accounting "
            "standards (GAAP, IFRS), depreciation schedules, VAT/GST obligations, "
            "and financial statement preparation for reporting purposes."
        ),
    },
    # -- HR & people operations -----------------------------------------------
    "Talent acquisition & recruiting": {
        "topic": "HR & people operations",
        "description": (
            "Writing job descriptions, sourcing and screening candidates, conducting "
            "interviews, extending offer letters, running background checks, and "
            "managing the end-to-end hiring pipeline."
        ),
    },
    "Onboarding & learning development": {
        "topic": "HR & people operations",
        "description": (
            "New-hire orientation programs, employee training curricula, skills "
            "certification tracking, mentorship structures, and ongoing learning "
            "and development initiatives within an organization."
        ),
    },
    "Performance management": {
        "topic": "HR & people operations",
        "description": (
            "Performance review cycles, goal-setting frameworks (OKRs, MBOs), "
            "performance improvement plans (PIPs), promotion calibration, and "
            "disciplinary procedures for underperforming employees."
        ),
    },
    "Compensation, benefits & leave": {
        "topic": "HR & people operations",
        "description": (
            "Salary structures, payroll processing, benefits enrollment (health, "
            "dental, 401k), paid and unpaid leave policies, expense reimbursement "
            "procedures, and employee total-compensation queries."
        ),
    },
    # -- Healthcare -----------------------------------------------------------
    "Clinical documentation": {
        "topic": "Healthcare",
        "description": (
            "Patient-level physical health records: physician consultation notes, "
            "prescriptions, diagnostic lab and imaging results, surgical reports, "
            "discharge summaries, and specialist referral letters."
        ),
    },
    "Public health & epidemiology": {
        "topic": "Healthcare",
        "description": (
            "Population-level health management: disease outbreak surveillance, "
            "vaccination campaign planning, epidemiological trend reports, "
            "quarantine guidelines, and public health advisories from health agencies."
        ),
    },
    "Medical research & clinical trials": {
        "topic": "Healthcare",
        "description": (
            "Clinical trial protocols and results, drug safety and adverse event "
            "reports, research abstracts in life sciences, FDA regulatory submissions, "
            "and peer-reviewed biomedical study findings."
        ),
    },
    "Mental health & behavioral care": {
        "topic": "Healthcare",
        "description": (
            "Psychiatric evaluations, therapy session notes, DSM-based diagnoses, "
            "behavioral treatment plans, counseling referrals, and psychotropic "
            "medication management for mental and behavioral health conditions."
        ),
    },
}

# ============================================================================
# SAMPLING UTILITIES
# ============================================================================


def weighted_choice(rng: random.Random, weights: dict[str, int]) -> str:
    """Pick a key from `weights` with probability proportional to its weight."""
    keys = list(weights.keys())
    return rng.choices(keys, weights=[weights[k] for k in keys], k=1)[0]


def build_sampling_plan(
    values: list[Any],
    count: int,
    mode: str,
    rng: random.Random,
    weights: list[int] | None = None,
) -> list[Any]:
    """Build a deterministic sampling plan using random, uniform, or weighted mode."""
    if mode not in ALLOWED_SELECTION_MODES:
        allowed = ", ".join(sorted(ALLOWED_SELECTION_MODES))
        raise ValueError(f"selection mode must be one of: {allowed}")
    if not values:
        raise ValueError("values must contain at least one item")
    if count <= 0:
        return []

    if mode == "random":
        return [rng.choice(values) for _ in range(count)]

    if mode == "uniform":
        plan = [values[i % len(values)] for i in range(count)]
        rng.shuffle(plan)
        return plan

    if weights is None or len(weights) != len(values):
        raise ValueError("weighted mode requires weights matching values length")
    if any(weight < 0 for weight in weights):
        raise ValueError("weights must be non-negative")
    if sum(weights) <= 0:
        raise ValueError("weighted mode requires at least one positive weight")
    return rng.choices(values, weights=weights, k=count)


def backoff_delay(attempt_index: int, min_seconds: float, max_seconds: float) -> float:
    """Exponential backoff with jitter: min*2^attempt + U(0, min), capped at max."""
    base = min(max_seconds, min_seconds * (2**attempt_index))
    jitter = random.uniform(0, min_seconds)
    return min(max_seconds, base + jitter)


def is_retryable_error(exc: Exception) -> bool:
    """True if the exception is a transient error worth retrying."""
    if isinstance(
        exc, (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
    ):
        return True
    msg = str(exc).lower()
    return any(
        tok in msg
        for tok in (
            "429",
            "rate limit",
            "timeout",
            "timed out",
            "resource_unavailable",
            "service unavailable",
            "overloaded",
            "internal server",
            "temporar",
        )
    )


def is_flex_tier_issue(exc: Exception) -> bool:
    """True if the error looks specific to the flex service tier (-> fall back)."""
    msg = str(exc).lower()
    return (
        "service_tier" in msg
        or "flex" in msg
        or "resource_unavailable" in msg
        or "429" in msg
    )


def is_temperature_unsupported_error(exc: Exception) -> bool:
    """True if the model rejected temperature as an unsupported parameter (reasoning models).

    BadRequestError.param is set to "temperature" by the OpenAI SDK when the model
    does not support the parameter, which is more reliable than string-matching the message.
    """
    if isinstance(exc, BadRequestError):
        return getattr(exc, "param", None) == "temperature"
    return False


# ============================================================================
# CONFIG HELPERS
# ============================================================================


def parse_selection_mode(mode_value: Any, field_name: str) -> str:
    """Validate and normalize one selection mode string from config."""
    mode = str(mode_value).lower().strip()
    if mode not in ALLOWED_SELECTION_MODES:
        allowed = ", ".join(sorted(ALLOWED_SELECTION_MODES))
        raise ValueError(f"{field_name} must be one of: {allowed}")
    return mode


def parse_temperature_weights(
    raw_weights: Any,
    num_temperatures: int,
) -> list[int] | None:
    """Parse optional explicit temperature weights from config.

    Returns None when raw_weights is absent (uniform/random mode will be used),
    or a list of non-negative ints whose length matches num_temperatures.
    """
    if raw_weights is None:
        return None
    if len(raw_weights) != num_temperatures:
        raise ValueError("temperature_weights length must match temperatures length")

    parsed_weights: list[int] = []
    for raw_weight in raw_weights:
        weight = int(raw_weight)
        if weight < 0:
            raise ValueError("temperature_weights must be non-negative integers")
        parsed_weights.append(weight)
    return parsed_weights


def build_temperature_plan(args: Any, num_requests: int) -> list[float | None]:
    """Build one temperature value per API call.

    Draws from `args.temperatures` (list) using the configured selection mode and
    weights when present. Falls back to the scalar `args.temperature` if set, or
    None (no temperature kwarg sent to the API) if neither is configured.
    """
    temperatures = getattr(args, "temperatures", None)
    if temperatures:
        plan_rng = random.Random(args.seed)
        return [
            float(value)
            for value in build_sampling_plan(
                temperatures,
                num_requests,
                args.temperature_selection_mode,
                plan_rng,
                weights=args.temperature_weights,
            )
        ]

    temperature = getattr(args, "temperature", None)
    return [float(temperature) if temperature is not None else None] * num_requests


# ============================================================================
# PARSING
# ============================================================================


def extract_json_array(raw_text: str, tag: str) -> tuple[list[Any] | None, str]:
    """
    Pull the JSON array out of <tag>...</tag>. Falls back to the first '[' .. last
    ']' span if the tag is missing. Returns (array, "") or (None, error_reason).
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None, "empty_model_output"

    m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", raw_text, re.IGNORECASE | re.DOTALL)
    candidate = m.group(1).strip() if m else None
    if candidate is None:
        start, end = raw_text.find("["), raw_text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return None, "prompts_tag_not_found"
        candidate = raw_text[start : end + 1].strip()

    try:
        # raw_decode (rather than loads) tolerates trailing content after the
        # array -- models sometimes append stray text inside the tags.
        parsed, _ = json.JSONDecoder().raw_decode(candidate)
    except json.JSONDecodeError as e:
        return None, f"json_decode_error: {e}"
    if not isinstance(parsed, list):
        return None, "json_not_array"
    return parsed, ""
