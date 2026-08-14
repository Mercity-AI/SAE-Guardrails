"""Shared topic-sequence decoders used by training and evaluation.

The raw, minimum-duration, hysteresis, and transition-penalty decoders are
strictly causal. Sticky Viterbi is retained for offline analysis and marked
non-causal because backtracking uses the complete sequence.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


CAUSAL_DECODERS = {"raw", "minimum_duration", "hysteresis", "transition_penalty"}
OFFLINE_DECODERS = {"viterbi"}


def softmax(logits: np.ndarray) -> np.ndarray:
    """Convert logits to probabilities using a numerically stable softmax."""
    values = np.asarray(logits)
    shifted = values - values.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def decode_raw(logits: np.ndarray) -> np.ndarray:
    """Select the highest-logit topic independently at every token."""
    return np.asarray(logits).argmax(axis=-1).astype(np.int64, copy=False)


def decode_minimum_duration(logits: np.ndarray, duration: int = 1) -> np.ndarray:
    """Switch after a challenger remains strongest for a minimum duration."""
    raw = decode_raw(logits)
    if not len(raw) or duration <= 1:
        return raw
    output = np.empty_like(raw)
    state = int(raw[0])
    challenger, count = -1, 0
    output[0] = state
    for index in range(1, len(raw)):
        proposed = int(raw[index])
        if proposed == state:
            challenger, count = -1, 0
        elif proposed == challenger:
            count += 1
        else:
            challenger, count = proposed, 1
        if count >= duration:
            state, challenger, count = challenger, -1, 0
        output[index] = state
    return output


def decode_hysteresis(logits: np.ndarray, margin: float = 0.2) -> np.ndarray:
    """Switch when a challenger exceeds the current topic by a probability margin."""
    probabilities = softmax(logits)
    if not len(probabilities):
        return np.empty(0, dtype=np.int64)
    output = np.empty(len(probabilities), dtype=np.int64)
    state = int(probabilities[0].argmax())
    output[0] = state
    for index, distribution in enumerate(probabilities[1:], start=1):
        challenger = int(distribution.argmax())
        if challenger != state and distribution[challenger] - distribution[state] >= margin:
            state = challenger
        output[index] = state
    return output


def decode_transition_penalty(logits: np.ndarray, penalty: float = 1.0) -> np.ndarray:
    """Accumulate causal log evidence before paying a topic-switch penalty."""
    values = np.asarray(logits)
    log_probabilities = values - np.logaddexp.reduce(values, axis=-1, keepdims=True)
    if not len(log_probabilities):
        return np.empty(0, dtype=np.int64)
    output = np.empty(len(log_probabilities), dtype=np.int64)
    state = int(log_probabilities[0].argmax())
    challenger, evidence = -1, 0.0
    output[0] = state
    for index, distribution in enumerate(log_probabilities[1:], start=1):
        proposed = int(distribution.argmax())
        if proposed == state:
            challenger, evidence = -1, 0.0
        else:
            increment = float(distribution[proposed] - distribution[state])
            if proposed != challenger:
                challenger, evidence = proposed, 0.0
            evidence = max(0.0, evidence + increment)
            if evidence >= penalty:
                state, challenger, evidence = challenger, -1, 0.0
        output[index] = state
    return output


def decode_viterbi(logits: np.ndarray, penalty: float = 1.0) -> np.ndarray:
    """Find an offline sticky-Viterbi path with a penalty for label changes."""
    values = np.asarray(logits)
    if not len(values):
        return np.empty(0, dtype=np.int64)
    log_probabilities = values - np.logaddexp.reduce(values, axis=-1, keepdims=True)
    classes = log_probabilities.shape[-1]
    score = log_probabilities[0].copy()
    backpointers = np.zeros((len(log_probabilities), classes), dtype=np.int64)
    transitions = np.full((classes, classes), -float(penalty))
    np.fill_diagonal(transitions, 0.0)
    for index in range(1, len(log_probabilities)):
        candidates = score[:, None] + transitions
        best_previous = candidates.argmax(axis=0)
        score = candidates[best_previous, np.arange(classes)] + log_probabilities[index]
        backpointers[index] = best_previous
    path = np.empty(len(log_probabilities), dtype=np.int64)
    path[-1] = int(score.argmax())
    for index in range(len(path) - 1, 0, -1):
        path[index - 1] = backpointers[index, path[index]]
    return path


def normalize_decoder_config(config: str | Mapping[str, object] | None) -> dict[str, object]:
    """Normalize a decoder name or mapping into one validated configuration."""
    if config is None:
        normalized: dict[str, object] = {"method": "raw"}
    elif isinstance(config, str):
        normalized = {"method": config}
    else:
        normalized = dict(config)
    method = str(normalized.get("method", "raw"))
    if method not in CAUSAL_DECODERS | OFFLINE_DECODERS:
        raise ValueError(f"unknown decoder: {method}")
    normalized["method"] = method
    return normalized


def decode_logits(
    logits: np.ndarray,
    config: str | Mapping[str, object] | None = None,
    *,
    require_causal: bool = True,
) -> np.ndarray:
    """Decode one logit sequence according to a validated configuration."""
    decoder = normalize_decoder_config(config)
    method = str(decoder["method"])
    if require_causal and method not in CAUSAL_DECODERS:
        raise ValueError(f"decoder {method!r} is offline-only and cannot be used in causal training")
    if method == "raw":
        return decode_raw(logits)
    if method == "minimum_duration":
        return decode_minimum_duration(logits, int(decoder.get("duration", 1)))
    if method == "hysteresis":
        return decode_hysteresis(logits, float(decoder.get("margin", 0.2)))
    if method == "transition_penalty":
        return decode_transition_penalty(logits, float(decoder.get("penalty", 1.0)))
    return decode_viterbi(logits, float(decoder.get("penalty", 1.0)))
