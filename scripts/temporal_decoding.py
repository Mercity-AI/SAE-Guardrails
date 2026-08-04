#!/usr/bin/env python3
"""Role-aware boundary metrics and causal temporal decoders.

All decoders operate left-to-right. Parameter selection is performed on the
validation split; the test split is evaluated once with the selected value.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from utils import PAD, TOPICS, build_model, load_data, load_split

ROOT = Path(__file__).resolve().parent
PROMPT, RESPONSE = 1, 2


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=-1, keepdims=True)


def decode_raw(logits: np.ndarray, **_: object) -> np.ndarray:
    return logits.argmax(axis=-1)


def decode_minimum_duration(logits: np.ndarray, duration: int) -> np.ndarray:
    """Causal debounce: confirm a challenger for ``duration`` tokens."""
    raw = logits.argmax(axis=-1)
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
    """Switch only when the challenger exceeds the active class by margin."""
    probabilities = softmax(logits)
    if not len(probabilities):
        return np.empty(0, dtype=np.int64)
    output = np.empty(len(probabilities), dtype=np.int64)
    state = int(probabilities[0].argmax())
    output[0] = state
    for index, distribution in enumerate(probabilities[1:], start=1):
        challenger = int(distribution.argmax())
        if (
            challenger != state
            and distribution[challenger] - distribution[state] >= margin
        ):
            state = challenger
        output[index] = state
    return output


def decode_transition_penalty(logits: np.ndarray, penalty: float) -> np.ndarray:
    """Causal accumulated log-evidence decoder with a switch cost.

    Evidence is the challenger/current log-probability ratio. Weak evidence
    must persist; strong evidence can cross the penalty quickly.
    """
    logp = logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)
    if not len(logp):
        return np.empty(0, dtype=np.int64)
    output = np.empty(len(logp), dtype=np.int64)
    state = int(logp[0].argmax())
    challenger, evidence = -1, 0.0
    output[0] = state
    for index, distribution in enumerate(logp[1:], start=1):
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


DECODERS = {
    "raw": decode_raw,
    "minimum_duration": decode_minimum_duration,
    "hysteresis": decode_hysteresis,
    "transition_penalty": decode_transition_penalty,
}


def _boundary_positions(labels: np.ndarray) -> np.ndarray:
    positions = np.flatnonzero(labels != PAD)
    if len(positions) < 2:
        return np.empty(0, dtype=np.int64)
    values = labels[positions]
    return positions[1:][values[1:] != values[:-1]]


def _match(true_positions: np.ndarray, predicted_positions: np.ndarray, tolerance: int):
    # Match closest pairs first so every true and predicted boundary is used once.
    pairs = sorted(
        (
            (abs(int(p) - int(t)), int(t), int(p))
            for t in true_positions
            for p in predicted_positions
            if abs(int(p) - int(t)) <= tolerance
        ),
        key=lambda item: item[0],
    )
    used_true, used_pred, errors = set(), set(), []
    for _, truth, prediction in pairs:
        if truth not in used_true and prediction not in used_pred:
            used_true.add(truth)
            used_pred.add(prediction)
            errors.append(prediction - truth)
    return errors


def role_boundary_report(rows: list[dict], role: int | None, tolerance: int) -> dict:
    """Score prompt, response, or the chronological combined stream.

    Prompt/response metrics mask the other role. Combined intentionally keeps
    both roles and therefore includes a topic reset at assistant turn start.
    """
    matched = predicted = true = 0
    errors: list[int] = []
    for row in rows:
        labels = np.asarray(row["labels"])
        roles = np.asarray(row["roles"])
        keep = labels != PAD
        if role is not None:
            keep &= roles == role
        truth_full = np.full_like(labels, PAD)
        prediction_full = np.full_like(labels, PAD)
        truth_full[keep] = labels[keep]
        prediction_full[keep] = np.asarray(row["predictions"])[keep]
        truth_positions = _boundary_positions(truth_full)
        predicted_positions = _boundary_positions(prediction_full)
        found = _match(truth_positions, predicted_positions, tolerance)
        matched += len(found)
        predicted += len(predicted_positions)
        true += len(truth_positions)
        errors.extend(found)
    precision = matched / predicted if predicted else 0.0
    recall = matched / true if true else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "matched": matched,
        "predicted": predicted,
        "true": true,
        "median_absolute_error": float(np.median(np.abs(errors))) if errors else None,
    }


def role_metrics(rows: list[dict], tolerances: list[int]) -> dict:
    return {
        name: {
            f"at_{tolerance}": role_boundary_report(rows, role, tolerance)
            for tolerance in tolerances
        }
        for name, role in (("prompt", PROMPT), ("response", RESPONSE), ("combined", None))
    }


@torch.inference_mode()
def infer_logits(model, rows, roles, device: str) -> list[dict]:
    model.eval()
    result = []
    for (features, labels), role_ids in zip(rows, roles):
        tensor = torch.from_numpy(np.asarray(features, dtype=np.float32))[None].to(device)
        logits = model(tensor)[0].float().cpu().numpy()
        if len(role_ids) != len(labels):
            raise ValueError("role length does not match cached sequence")
        result.append({"labels": labels.copy(), "roles": role_ids.copy(), "logits": logits})
    return result


def apply_decoder(rows: list[dict], decoder: str, value: float | int | None) -> list[dict]:
    kwargs = {}
    if decoder == "minimum_duration":
        kwargs["duration"] = int(value)
    elif decoder == "hysteresis":
        kwargs["margin"] = float(value)
    elif decoder == "transition_penalty":
        kwargs["penalty"] = float(value)
    fn = DECODERS[decoder]
    return [
        {**row, "predictions": fn(row["logits"], **kwargs)}
        for row in rows
    ]


def load_roles(cache: Path, indices: np.ndarray) -> list[np.ndarray]:
    lengths = np.load(cache / "lengths.npy")
    flat = np.load(cache / "role_ids.npy", mmap_mode="r")
    starts = np.concatenate(([0], np.cumsum(lengths[:-1])))
    return [np.asarray(flat[s : s + lengths[i]]) for i, s in ((int(i), starts[int(i)]) for i in indices)]


def main(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text())
    cache = ROOT / config["data"]["cache_dir"]
    checkpoint_path = ROOT / config["checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["architecture"], checkpoint["model"])
    model.load_state_dict(checkpoint["state_dict"])
    device = config.get("device", "cuda")
    model.to(device)
    data = load_data(cache)
    partitions = load_split(cache, data)
    split = np.load(cache / "split_indices.npz")
    validation_logits = infer_logits(model, partitions["validation"], load_roles(cache, split["validation"]), device)
    test_logits = infer_logits(model, partitions["test"], load_roles(cache, split["test"]), device)
    tolerances = [int(x) for x in config["evaluation"]["boundary_tolerances"]]
    output = {"checkpoint": str(checkpoint_path), "selection_partition": "validation", "test_used_for_selection": False, "methods": {}}
    for name, specification in config["decoders"].items():
        decoder = specification["method"]
        candidates = specification.get("values", [specification.get("value")])
        scored = []
        for value in candidates:
            decoded = apply_decoder(validation_logits, decoder, value)
            reports = role_metrics(decoded, tolerances)
            score = reports["combined"][f"at_{tolerances[0]}"]["f1"]
            scored.append({"value": value, "selection_score": score, "metrics": reports})
        winner = max(scored, key=lambda row: row["selection_score"])
        test_rows = apply_decoder(test_logits, decoder, winner["value"])
        output["methods"][name] = {
            "method": decoder,
            "selected_value": winner["value"],
            "validation_candidates": scored,
            "test": role_metrics(test_rows, tolerances),
        }
    output_path = ROOT / config["output"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    (output_path.parent / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    main(parser.parse_args().config)
