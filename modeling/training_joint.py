#!/usr/bin/env python3
"""Shared joint topic-and-boundary training engine."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from decoder_utils import decode_logits
from models import (
    PAD,
    TOPICS,
    build_model,
    seed_everything,
    topic_detection_latency,
    topic_overlap,
)
from utils import require_cache

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve(value: str) -> Path:
    """Resolve configuration paths relative to the project root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_rows(cache: Path):
    """Load cached features, topic labels, roles, and persisted split indices."""
    features = np.load(cache / "features.npy", mmap_mode="r")
    labels = np.load(cache / "labels.npy", mmap_mode="r")
    roles_path = cache / "role_ids.npy"
    if not roles_path.exists():
        raise FileNotFoundError(f"{roles_path} is required; run prepare_role_ids.py")
    roles = np.load(roles_path, mmap_mode="r")
    lengths = np.load(cache / "lengths.npy")
    if not (len(features) == len(labels) == len(roles) == int(lengths.sum())):
        raise ValueError("cache arrays have inconsistent token counts")
    offsets = np.r_[0, np.cumsum(lengths)]
    rows = []
    for start, end in zip(offsets[:-1], offsets[1:]):
        rows.append(
            (features[start:end], labels[start:end].astype(np.int64), roles[start:end])
        )
    split = np.load(cache / "split_indices.npz")
    return rows, {name: split[name] for name in ("train", "validation", "test")}


def boundary_targets(
    labels: np.ndarray, roles: np.ndarray, *, include_turn_transition: bool = False
) -> np.ndarray:
    """Build explicit boundary targets without accidentally crossing masked gaps."""
    targets = np.full(len(labels), PAD, dtype=np.int64)
    for role in (1, 2):
        positions = np.flatnonzero((labels != PAD) & (roles == role))
        if not len(positions):
            continue
        targets[positions] = 0
        changed = labels[positions[1:]] != labels[positions[:-1]]
        targets[positions[1:][changed]] = 1
    if include_turn_transition:
        prompt = np.flatnonzero((labels != PAD) & (roles == 1))
        response = np.flatnonzero((labels != PAD) & (roles == 2))
        if len(prompt) and len(response):
            targets[response[0]] = int(labels[prompt[-1]] != labels[response[0]])
    return targets


def collate(rows, indices, input_features, include_turn_transition):
    """Right-pad selected joint-training sequences into one batch."""
    selected = [rows[int(index)] for index in indices]
    length = max(len(row[0]) for row in selected)
    device = (
        selected[0][0].device if isinstance(selected[0][0], torch.Tensor) else "cpu"
    )
    x = torch.zeros(len(selected), length, input_features, device=device)
    topics = torch.full((len(selected), length), PAD, dtype=torch.long, device=device)
    boundaries = torch.full(
        (len(selected), length), PAD, dtype=torch.long, device=device
    )
    for index, (features, labels, roles) in enumerate(selected):
        size = len(labels)
        feature_tensor = (
            features
            if isinstance(features, torch.Tensor)
            else torch.from_numpy(np.array(features, dtype=np.float32, copy=True))
        )
        label_tensor = (
            labels
            if isinstance(labels, torch.Tensor)
            else torch.from_numpy(np.array(labels, copy=True))
        )
        role_tensor = (
            roles
            if isinstance(roles, torch.Tensor)
            else torch.from_numpy(np.array(roles, copy=True))
        )
        x[index, :size] = feature_tensor
        topics[index, :size] = label_tensor
        target = torch.full((size,), PAD, dtype=torch.long, device=device)
        for role in (1, 2):
            positions = torch.nonzero(
                (label_tensor != PAD) & (role_tensor == role), as_tuple=False
            ).flatten()
            if len(positions):
                target[positions] = 0
                changed = label_tensor[positions[1:]] != label_tensor[positions[:-1]]
                target[positions[1:][changed]] = 1
        if include_turn_transition:
            prompt = torch.nonzero(
                (label_tensor != PAD) & (role_tensor == 1), as_tuple=False
            ).flatten()
            response = torch.nonzero(
                (label_tensor != PAD) & (role_tensor == 2), as_tuple=False
            ).flatten()
            if len(prompt) and len(response):
                target[response[0]] = (
                    label_tensor[prompt[-1]] != label_tensor[response[0]]
                ).long()
        boundaries[index, :size] = target
    return x, topics, boundaries


@torch.inference_mode()
def predict(model, rows, indices, device, threshold, batch_size=1, decoder=None):
    """Predict topic labels and boundary scores for selected records."""
    model.eval()
    output = []
    for start in range(0, len(indices), batch_size):
        selected_indices = indices[start : start + batch_size]
        x, _, _ = collate(
            rows, selected_indices, rows[int(selected_indices[0])][0].shape[-1], True
        )
        topic_logits, boundary_logits = model(x.to(device))
        for offset, row_index in enumerate(selected_indices):
            features, labels, roles = rows[int(row_index)]
            size = len(labels)
            label_values = (
                labels.detach().cpu().numpy()
                if isinstance(labels, torch.Tensor)
                else labels
            )
            role_values = (
                roles.detach().cpu().numpy()
                if isinstance(roles, torch.Tensor)
                else np.asarray(roles)
            )
            output.append(
                {
                    "labels": label_values,
                    "roles": role_values,
                    "topic_predictions": decode_logits(
                        topic_logits[offset, :size].float().cpu().numpy(), decoder
                    ),
                    "boundary_predictions": (
                        boundary_logits[offset, :size].sigmoid() >= threshold
                    )
                    .cpu()
                    .numpy(),
                    "boundary_scores": boundary_logits[offset, :size]
                    .sigmoid()
                    .cpu()
                    .numpy(),
                }
            )
    return output


def role_metrics(rows, role=None):
    """Calculate topic, overlap, and exact-boundary metrics for one role scope."""
    truth_topics, pred_topics, truth_bounds, pred_bounds = [], [], [], []
    for row in rows:
        keep = row["labels"] != PAD
        if role is not None:
            keep &= row["roles"] == role
        truth_topics.extend(row["labels"][keep])
        pred_topics.extend(row["topic_predictions"][keep])
        targets = boundary_targets(
            row["labels"], row["roles"], include_turn_transition=role is None
        )
        boundary_keep = targets != PAD
        if role is not None:
            boundary_keep &= row["roles"] == role
        truth_bounds.extend(targets[boundary_keep])
        pred_bounds.extend(row["boundary_predictions"][boundary_keep])
    precision, recall, boundary_f1, _ = precision_recall_fscore_support(
        truth_bounds, pred_bounds, average="binary", zero_division=0
    )
    overlap = topic_overlap(truth_topics, pred_topics)
    latency_rows = []
    for row in rows:
        labels = np.asarray(row["labels"]).copy()
        if role is not None:
            labels[np.asarray(row["roles"]) != role] = PAD
        latency_rows.append({"labels": labels, "predictions": row["topic_predictions"]})
    return {
        "topic_accuracy": float(accuracy_score(truth_topics, pred_topics)),
        "topic_macro_f1": float(
            f1_score(truth_topics, pred_topics, average="macro", zero_division=0)
        ),
        "topic_overlap": overlap,
        "boundary_precision": float(precision),
        "boundary_recall": float(recall),
        "boundary_f1": float(boundary_f1),
        "true_boundaries": int(np.sum(truth_bounds)),
        "predicted_boundaries": int(np.sum(pred_bounds)),
        "topic_detection_first_correct": topic_detection_latency(latency_rows, 1),
        "topic_detection_confirmed_3": topic_detection_latency(latency_rows, 3),
    }


def evaluate(model, rows, indices, device, threshold, batch_size=1, decoder=None):
    """Evaluate prompt, response, and combined chronological scopes."""
    predictions = predict(
        model, rows, indices, device, threshold, batch_size=batch_size, decoder=decoder
    )
    return {
        "prompt": role_metrics(predictions, 1),
        "response": role_metrics(predictions, 2),
        # Combined is the chronological deployment stream and explicitly adds
        # the prompt-final -> response-first transition when the topics differ.
        "combined": role_metrics(predictions),
    }


def clone_state(model):
    """Copy a model state to CPU memory for checkpoint retention."""
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def train_joint_model(config_path: Path):
    """Train one joint model and retain topic-best, boundary-best, and final states."""
    config = yaml.safe_load(config_path.read_text())
    cache = resolve(config["data"]["cache_dir"])
    require_cache(cache, require_roles=True)
    rows, split = load_rows(cache)
    experiment = config["experiment"]
    training = config["training"]
    specification = config["model"]
    seed = int(experiment.get("seed", 42))
    seed_everything(seed, deterministic=bool(experiment.get("deterministic", True)))
    print(f"using training and decoder seed {seed}", flush=True)
    device = experiment.get("device", "cuda")
    model_config = {
        "input_features": int(config["data"]["input_features"]),
        "classes": len(TOPICS),
        **specification["parameters"],
    }
    architecture = specification["architecture"]
    if architecture not in {"joint_convnext", "joint_gru", "joint_transformer"}:
        raise ValueError(f"unsupported joint architecture: {architecture}")
    model = build_model(architecture, model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    topic_loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    include_turn = bool(training.get("train_turn_transition", True))
    train_targets = np.concatenate(
        [
            boundary_targets(
                rows[int(i)][1], rows[int(i)][2], include_turn_transition=include_turn
            )
            for i in split["train"]
        ]
    )
    positives = int((train_targets == 1).sum())
    negatives = int((train_targets == 0).sum())
    if positives == 0:
        raise ValueError("training split contains no boundaries")
    configured_weight = training.get("boundary_positive_weight", "auto")
    positive_weight = (
        negatives / positives
        if configured_weight == "auto"
        else float(configured_weight)
    )
    boundary_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, device=device)
    )
    if bool(experiment.get("gpu_resident_cache", False)):
        rows = [
            (
                torch.from_numpy(np.array(features, dtype=np.float32, copy=True)).to(
                    device
                ),
                torch.from_numpy(np.array(labels, dtype=np.int64, copy=True)).to(
                    device
                ),
                torch.from_numpy(np.array(roles, dtype=np.int64, copy=True)).to(device),
            )
            for features, labels, roles in rows
        ]
        print("loaded joint cache onto CUDA", flush=True)
    boundary_weight = float(training.get("boundary_loss_weight", 1.0))
    threshold = float(config["evaluation"].get("boundary_threshold", 0.5))
    decoder = config["evaluation"].get("decoder", {"method": "raw"})
    best_score, best_epoch, best_state = -1.0, 0, None
    best_boundary_score, best_boundary_epoch, best_boundary_state = -1.0, 0, None
    history = []
    batch_size = int(training["batch_size"])
    evaluation_batch_size = int(config["evaluation"].get("batch_size", batch_size))
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        permutation = np.random.permutation(split["train"])
        losses = []
        for start in range(0, len(permutation), batch_size):
            x, topics, boundaries = collate(
                rows,
                permutation[start : start + batch_size],
                model_config["input_features"],
                include_turn,
            )
            topic_logits, boundary_logits = model(x.to(device))
            topic_loss = topic_loss_fn(
                topic_logits.reshape(-1, len(TOPICS)),
                topics.reshape(-1).to(device),
            )
            flat_boundaries = boundaries.reshape(-1).to(device)
            keep = flat_boundaries != PAD
            boundary_loss = boundary_loss_fn(
                boundary_logits.reshape(-1)[keep], flat_boundaries[keep].float()
            )
            loss = topic_loss + boundary_weight * boundary_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = evaluate(
            model,
            rows,
            split["validation"],
            device,
            threshold,
            evaluation_batch_size,
            decoder,
        )
        score = validation["combined"]["topic_macro_f1"]
        history.append(
            {"epoch": epoch, "loss": float(np.mean(losses)), "validation": validation}
        )
        print(
            f"epoch {epoch:2d}: loss={np.mean(losses):.4f} "
            f"val_topic_f1={score:.4f} "
            f"val_boundary_f1={validation['combined']['boundary_f1']:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score, best_epoch, best_state = score, epoch, clone_state(model)
        boundary_score = validation["combined"]["boundary_f1"]
        if boundary_score > best_boundary_score:
            best_boundary_score = boundary_score
            best_boundary_epoch = epoch
            best_boundary_state = clone_state(model)
    final_state = clone_state(model)
    output = resolve(experiment["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    common = {
        "architecture": architecture,
        "model": model_config,
        "classes": TOPICS,
        "config": config,
    }
    torch.save(
        {**common, "epoch": best_epoch, "state_dict": best_state},
        output / "checkpoint_best_topic.pt",
    )
    torch.save(
        {**common, "epoch": best_boundary_epoch, "state_dict": best_boundary_state},
        output / "checkpoint_best_boundary.pt",
    )
    torch.save(
        {**common, "epoch": int(training["epochs"]), "state_dict": final_state},
        output / "checkpoint_final.pt",
    )
    model.load_state_dict(best_state)
    best_validation = evaluate(
        model,
        rows,
        split["validation"],
        device,
        threshold,
        evaluation_batch_size,
        decoder,
    )
    best_test = evaluate(
        model, rows, split["test"], device, threshold, evaluation_batch_size, decoder
    )
    model.load_state_dict(best_boundary_state)
    best_boundary_validation = evaluate(
        model,
        rows,
        split["validation"],
        device,
        threshold,
        evaluation_batch_size,
        decoder,
    )
    best_boundary_test = evaluate(
        model, rows, split["test"], device, threshold, evaluation_batch_size, decoder
    )
    model.load_state_dict(final_state)
    final_validation = evaluate(
        model,
        rows,
        split["validation"],
        device,
        threshold,
        evaluation_batch_size,
        decoder,
    )
    final_test = evaluate(
        model, rows, split["test"], device, threshold, evaluation_batch_size, decoder
    )
    checkpoint_metrics = {
        "checkpoint_best_topic": {
            "epoch": best_epoch,
            "validation": best_validation,
            "test": best_test,
        },
        "checkpoint_best_boundary": {
            "epoch": best_boundary_epoch,
            "validation": best_boundary_validation,
            "test": best_boundary_test,
        },
        "checkpoint_final": {
            "epoch": int(training["epochs"]),
            "validation": final_validation,
            "test": final_test,
        },
    }
    for checkpoint_name, values in checkpoint_metrics.items():
        (output / f"{checkpoint_name}.metrics.json").write_text(
            json.dumps({"checkpoint": f"{checkpoint_name}.pt", **values}, indent=2)
            + "\n"
        )
    results = {
        "best_epoch": best_epoch,
        "best_boundary_epoch": best_boundary_epoch,
        "boundary_positive_weight": positive_weight,
        "boundary_loss_weight": boundary_weight,
        "seed": seed,
        "decoder": decoder,
        "best_validation": best_validation,
        "best_test": best_test,
        "best_boundary_validation": best_boundary_validation,
        "best_boundary_test": best_boundary_test,
        "final_validation": final_validation,
        "final_test": final_test,
        "history": history,
    }
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    for checkpoint_name, values in checkpoint_metrics.items():
        combined = values["test"]["combined"]
        overlap = combined["topic_overlap"]
        print(
            f"{checkpoint_name} test: topic F1={combined['topic_macro_f1']:.4f} "
            f"Dice={overlap['macro_dice']:.4f} IoU={overlap['macro_iou']:.4f}",
            flush=True,
        )
    print(f"wrote {output}")
