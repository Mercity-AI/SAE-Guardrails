#!/usr/bin/env python3
"""Train the role-aware joint ConvNeXt topic and boundary model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from models import PAD, TOPICS, build_model

ROOT = Path(__file__).resolve().parent


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_rows(cache: Path):
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
        rows.append((features[start:end], labels[start:end].astype(np.int64), roles[start:end]))
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
    selected = [rows[int(index)] for index in indices]
    length = max(len(row[0]) for row in selected)
    x = torch.zeros(len(selected), length, input_features)
    topics = torch.full((len(selected), length), PAD, dtype=torch.long)
    boundaries = torch.full((len(selected), length), PAD, dtype=torch.long)
    for index, (features, labels, roles) in enumerate(selected):
        size = len(labels)
        x[index, :size] = torch.from_numpy(np.asarray(features, dtype=np.float32))
        topics[index, :size] = torch.from_numpy(labels)
        boundaries[index, :size] = torch.from_numpy(
            boundary_targets(labels, roles, include_turn_transition=include_turn_transition)
        )
    return x, topics, boundaries


@torch.inference_mode()
def predict(model, rows, indices, device, threshold):
    model.eval()
    output = []
    for index in indices:
        features, labels, roles = rows[int(index)]
        x = torch.from_numpy(np.asarray(features, dtype=np.float32))[None].to(device)
        topic_logits, boundary_logits = model(x)
        output.append({
            "labels": labels,
            "roles": np.asarray(roles),
            "topic_predictions": topic_logits[0].argmax(-1).cpu().numpy(),
            "boundary_predictions": (boundary_logits[0].sigmoid() >= threshold).cpu().numpy(),
            "boundary_scores": boundary_logits[0].sigmoid().cpu().numpy(),
        })
    return output


def role_metrics(rows, role=None):
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
    return {
        "topic_accuracy": float(accuracy_score(truth_topics, pred_topics)),
        "topic_macro_f1": float(f1_score(truth_topics, pred_topics, average="macro", zero_division=0)),
        "boundary_precision": float(precision),
        "boundary_recall": float(recall),
        "boundary_f1": float(boundary_f1),
        "true_boundaries": int(np.sum(truth_bounds)),
        "predicted_boundaries": int(np.sum(pred_bounds)),
    }


def evaluate(model, rows, indices, device, threshold):
    predictions = predict(model, rows, indices, device, threshold)
    return {
        "prompt": role_metrics(predictions, 1),
        "response": role_metrics(predictions, 2),
        # Combined is the chronological deployment stream and explicitly adds
        # the prompt-final -> response-first transition when the topics differ.
        "combined": role_metrics(predictions),
    }


def clone_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def main(config_path: Path):
    config = yaml.safe_load(config_path.read_text())
    cache = resolve(config["data"]["cache_dir"])
    rows, split = load_rows(cache)
    experiment = config["experiment"]
    training = config["training"]
    specification = config["model"]
    seed = int(experiment.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = experiment.get("device", "cuda")
    model_config = {"input_features": int(config["data"]["input_features"]), "classes": len(TOPICS), **specification["parameters"]}
    model = build_model("joint_convnext", model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    topic_loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    include_turn = bool(training.get("train_turn_transition", True))
    train_targets = np.concatenate([
        boundary_targets(
            rows[int(i)][1], rows[int(i)][2], include_turn_transition=include_turn
        )
        for i in split["train"]
    ])
    positives = int((train_targets == 1).sum())
    negatives = int((train_targets == 0).sum())
    if positives == 0:
        raise ValueError("training split contains no boundaries")
    configured_weight = training.get("boundary_positive_weight", "auto")
    positive_weight = negatives / positives if configured_weight == "auto" else float(configured_weight)
    boundary_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
    boundary_weight = float(training.get("boundary_loss_weight", 1.0))
    threshold = float(config["evaluation"].get("boundary_threshold", 0.5))
    best_score, best_epoch, best_state = -1.0, 0, None
    best_boundary_score, best_boundary_epoch, best_boundary_state = -1.0, 0, None
    history = []
    batch_size = int(training["batch_size"])
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        permutation = np.random.permutation(split["train"])
        losses = []
        for start in range(0, len(permutation), batch_size):
            x, topics, boundaries = collate(
                rows,
                permutation[start:start + batch_size],
                model_config["input_features"],
                include_turn,
            )
            topic_logits, boundary_logits = model(x.to(device))
            topic_loss = topic_loss_fn(topic_logits.reshape(-1, len(TOPICS)), topics.reshape(-1).to(device))
            flat_boundaries = boundaries.reshape(-1).to(device)
            keep = flat_boundaries != PAD
            boundary_loss = boundary_loss_fn(boundary_logits.reshape(-1)[keep], flat_boundaries[keep].float())
            loss = topic_loss + boundary_weight * boundary_loss
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"])); optimizer.step()
            losses.append(float(loss.detach()))
        validation = evaluate(model, rows, split["validation"], device, threshold)
        score = validation["combined"]["topic_macro_f1"]
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "validation": validation})
        print(f"epoch {epoch:2d}: loss={np.mean(losses):.4f} val_topic_f1={score:.4f} val_boundary_f1={validation['combined']['boundary_f1']:.4f}", flush=True)
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
    common = {"architecture": "joint_convnext", "model": model_config, "classes": TOPICS, "config": config}
    torch.save({**common, "epoch": best_epoch, "state_dict": best_state}, output / "checkpoint_best_topic.pt")
    torch.save(
        {**common, "epoch": best_boundary_epoch, "state_dict": best_boundary_state},
        output / "checkpoint_best_boundary.pt",
    )
    torch.save({**common, "epoch": int(training["epochs"]), "state_dict": final_state}, output / "checkpoint_final.pt")
    model.load_state_dict(best_state)
    best_validation = evaluate(model, rows, split["validation"], device, threshold)
    best_test = evaluate(model, rows, split["test"], device, threshold)
    model.load_state_dict(best_boundary_state)
    best_boundary_validation = evaluate(model, rows, split["validation"], device, threshold)
    best_boundary_test = evaluate(model, rows, split["test"], device, threshold)
    model.load_state_dict(final_state)
    final_validation = evaluate(model, rows, split["validation"], device, threshold)
    results = {
        "best_epoch": best_epoch,
        "best_boundary_epoch": best_boundary_epoch,
        "boundary_positive_weight": positive_weight,
        "boundary_loss_weight": boundary_weight,
        "best_validation": best_validation,
        "best_test": best_test,
        "best_boundary_validation": best_boundary_validation,
        "best_boundary_test": best_boundary_test,
        "final_validation": final_validation,
        "history": history,
    }
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    main(args.config)
