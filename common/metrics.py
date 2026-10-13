"""Token-level evaluation: macro-F1, per-topic overlap, boundary matching and detection latency.

Every metric is computed over the labelled tokens only (``labels != PAD``), pooled across records,
in the original token coordinates.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

# local imports
from models import PAD, TOPICS


def flatten_labeled(rows):
    """Flatten only supervised labels and matching predictions across records."""
    truth, predicted = [], []
    for row in rows:
        keep = row["labels"] != PAD
        truth.append(row["labels"][keep])
        predicted.append(row["predictions"][keep])
    return np.concatenate(truth), np.concatenate(predicted)


def topic_overlap(truth, predicted):
    """Pooled token-wise one-vs-rest Dice/IoU for every topic class."""
    truth = np.asarray(truth)
    predicted = np.asarray(predicted)
    per_topic = {}
    for class_id, topic in enumerate(TOPICS):
        true_mask = truth == class_id
        predicted_mask = predicted == class_id
        intersection = int(np.count_nonzero(true_mask & predicted_mask))
        true_tokens = int(np.count_nonzero(true_mask))
        predicted_tokens = int(np.count_nonzero(predicted_mask))
        dice_denominator = true_tokens + predicted_tokens
        union = true_tokens + predicted_tokens - intersection
        per_topic[topic] = {
            "dice": 2.0 * intersection / dice_denominator if dice_denominator else 0.0,
            "iou": intersection / union if union else 0.0,
            "intersection_tokens": intersection,
            "true_tokens": true_tokens,
            "predicted_tokens": predicted_tokens,
        }
    return {
        "definition": "pooled labeled-token one-vs-rest overlap; macro is unweighted across topics",
        "macro_dice": float(np.mean([value["dice"] for value in per_topic.values()])),
        "macro_iou": float(np.mean([value["iou"] for value in per_topic.values()])),
        "per_topic": per_topic,
    }


def boundary_positions(labels):
    """Return transitions in original token coordinates, retaining masked gaps."""
    positions = np.flatnonzero(labels != PAD)
    if len(positions) < 2:
        return np.empty(0, dtype=np.int64)
    values = labels[positions]
    return positions[1:][values[1:] != values[:-1]]


def boundary_report(rows, tolerance: int):
    """Calculate one-to-one tolerant boundary precision, recall, and F1."""
    matched = predicted = true = 0
    errors = []
    for row in rows:
        labels = row["labels"]
        keep = labels != PAD
        predicted_full = np.full_like(labels, PAD)
        predicted_full[keep] = row["predictions"][keep]
        true_positions = boundary_positions(labels)
        predicted_positions = boundary_positions(predicted_full)
        true += len(true_positions)
        predicted += len(predicted_positions)
        unused = set(map(int, predicted_positions))
        for position in true_positions:
            candidates = [candidate for candidate in unused if abs(candidate - int(position)) <= tolerance]
            if candidates:
                candidate = min(candidates, key=lambda value: abs(value - int(position)))
                unused.remove(candidate)
                matched += 1
                errors.append(candidate - int(position))
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


def topic_detection_latency(rows, confirmation_tokens: int = 3):
    """Measure how far into each true topic span a correct prediction appears.

    Latency is one-based tokens consumed: an immediately correct prediction has latency 1.
    ``confirmation_tokens`` requires that many consecutive correct predictions, shortened to
    the span length for spans shorter than the requested confirmation window. Statistics over
    latency are conditional on detection; detection rate and misses are reported separately.
    """
    observations = {topic: [] for topic in TOPICS}
    for row in rows:
        keep = row["labels"] != PAD
        truth = np.asarray(row["labels"])[keep]
        predicted = np.asarray(row["predictions"])[keep]
        start = 0
        while start < len(truth):
            stop = start + 1
            while stop < len(truth) and truth[stop] == truth[start]:
                stop += 1
            class_id = int(truth[start])
            span_predictions = predicted[start:stop]
            span_length = stop - start
            required = min(int(confirmation_tokens), span_length)
            consumed = None
            for offset in range(span_length - required + 1):
                if np.all(span_predictions[offset : offset + required] == class_id):
                    consumed = offset + required
                    break
            observations[TOPICS[class_id]].append((span_length, consumed))
            start = stop

    def summarize(values):
        """Summarize detection coverage and conditional latency values."""
        detected = [(length, consumed) for length, consumed in values if consumed is not None]
        tokens = np.asarray([consumed for _, consumed in detected], dtype=np.float64)
        fractions = np.asarray([consumed / length for length, consumed in detected], dtype=np.float64)
        spans = len(values)
        found = len(detected)
        return {
            "spans": spans,
            "detected_spans": found,
            "missed_spans": spans - found,
            "detection_rate": found / spans if spans else 0.0,
            "mean_tokens_consumed": float(tokens.mean()) if found else None,
            "median_tokens_consumed": float(np.median(tokens)) if found else None,
            "p90_tokens_consumed": float(np.percentile(tokens, 90)) if found else None,
            "mean_fraction_consumed": float(fractions.mean()) if found else None,
            "median_fraction_consumed": float(np.median(fractions)) if found else None,
        }

    pooled = [value for values in observations.values() for value in values]
    return {
        "definition": (
            "one-based labeled topic tokens consumed from each true span start until the first "
            f"run of {confirmation_tokens} correct predictions; for shorter spans the required "
            "run equals span length; latency summaries are conditional on detection"
        ),
        "confirmation_tokens": int(confirmation_tokens),
        "overall": summarize(pooled),
        "per_topic": {topic: summarize(values) for topic, values in observations.items()},
    }


def metrics(rows, tolerances: list[int] | None = None):
    """Calculate token, overlap, boundary, and acquisition metrics."""
    tolerances = tolerances or [5, 10]
    truth, predicted = flatten_labeled(rows)
    report = classification_report(
        truth,
        predicted,
        labels=range(len(TOPICS)),
        target_names=TOPICS,
        output_dict=True,
        zero_division=0,
    )
    result = {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "topic_overlap": topic_overlap(truth, predicted),
        "per_topic": {topic: report[topic] for topic in TOPICS},
        "topic_detection_first_correct": topic_detection_latency(rows, confirmation_tokens=1),
        "topic_detection_confirmed_3": topic_detection_latency(rows, confirmation_tokens=3),
    }
    for tolerance in tolerances:
        result[f"boundary_at_{tolerance}"] = boundary_report(rows, tolerance)
    return result
