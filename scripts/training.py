#!/usr/bin/env python3
"""Run corrected causal TCN experiments declared entirely in YAML."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score

from utils import (
    PAD,
    TOPICS,
    build_model,
    collate,
    flatten_labeled,
    load_data,
    load_split,
    metrics,
    predict_sequences,
)

ROOT = Path(__file__).resolve().parent


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    required = {"experiment", "data", "training", "evaluation", "models"}
    missing = required - set(config or {})
    if missing:
        raise ValueError(f"missing config sections: {sorted(missing)}")
    if config["data"].get("split_file") != "split_indices.npz":
        raise ValueError("corrected runs require data.split_file: split_indices.npz")
    if config["data"].get("feature_selection") != "training_records_only":
        raise ValueError("corrected runs require train-only SAE feature selection")
    if config["evaluation"].get("report_partition") != "test":
        raise ValueError("corrected runs must report the untouched test partition")
    if config["training"].get("checkpoint_metric") != "validation_macro_f1":
        raise ValueError("checkpoint selection must use validation_macro_f1")
    if config["evaluation"].get("boundary_coordinates") != "original_token_positions":
        raise ValueError("boundary metrics must use original_token_positions")
    for name, specification in config["models"].items():
        if "architecture" not in specification:
            raise ValueError(f"model {name!r} has no architecture")
        build_model(
            specification["architecture"],
            {
                "input_features": int(config["data"]["input_features"]),
                "classes": len(TOPICS),
                **specification.get("model", {}),
            },
        )
    return config


def train_one(name, specification, config, partitions, output_dir, device):
    seed = int(config["experiment"]["seed"])
    training = config["training"]
    model_config = {
        "input_features": int(config["data"]["input_features"]),
        "classes": len(TOPICS),
        **specification.get("model", {}),
    }
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(specification["architecture"], model_config).to(device)
    learning_rate = float(specification.get("learning_rate", training["learning_rate"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(training["weight_decay"]),
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    best_score, best_epoch, best_state = -1.0, 0, None
    train_rows = partitions["train"]
    validation_rows = partitions["validation"]
    epochs = int(training["epochs"])
    batch_size = int(specification.get("batch_size", training["batch_size"]))
    consistency_weight = float(
        specification.get(
            "temporal_consistency_weight",
            training.get("temporal_consistency_weight", 0.0),
        )
    )
    consistency_tau = float(training.get("temporal_consistency_tau", 4.0))

    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.permutation(len(train_rows))
        for start in range(0, len(order), batch_size):
            x_batch, y_batch = collate(
                [train_rows[index] for index in order[start : start + batch_size]],
                input_features=model_config["input_features"],
            )
            logits = model(x_batch.to(device))
            targets = y_batch.to(device)
            loss = loss_fn(
                logits.reshape(-1, len(TOPICS)),
                targets.reshape(-1),
            )
            if consistency_weight:
                log_probabilities = logits.log_softmax(dim=-1)
                stable = (
                    (targets[:, 1:] != PAD)
                    & (targets[:, :-1] != PAD)
                    & (targets[:, 1:] == targets[:, :-1])
                )
                if stable.any():
                    # MS-TCN's truncated temporal MSE: suppress rapid changes
                    # inside a true segment without smoothing across boundaries.
                    probability_shift = (
                        log_probabilities[:, 1:]
                        - log_probabilities[:, :-1].detach()
                    ).abs().clamp(max=consistency_tau).square().mean(dim=-1)
                    loss = loss + consistency_weight * probability_shift[stable].mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            optimizer.step()

        rows = predict_sequences(model, validation_rows, device=device)
        truth, predicted = flatten_labeled(rows)
        score = f1_score(truth, predicted, average="macro", zero_division=0)
        print(f"{name} epoch {epoch:2d}/{epochs}: validation macro-F1={score:.4f}", flush=True)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    final_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    tolerances = [int(value) for value in config["evaluation"]["boundary_tolerances"]]
    final_validation_metrics = metrics(
        predict_sequences(model, validation_rows, device=device), tolerances
    )
    model.load_state_dict(best_state)
    validation_metrics = metrics(
        predict_sequences(model, validation_rows, device=device), tolerances
    )
    test_metrics = metrics(
        predict_sequences(model, partitions["test"], device=device), tolerances
    )
    best_checkpoint = {
        "state_dict": best_state,
        "architecture": specification["architecture"],
        "model": model_config,
        "classes": TOPICS,
        "config": config,
        "epoch": best_epoch,
        "selection": "best_validation_macro_f1",
    }
    final_checkpoint = {
        **best_checkpoint,
        "state_dict": final_state,
        "epoch": epochs,
        "selection": "final_epoch",
    }
    run_dir = output_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_checkpoint, run_dir / "checkpoint_best.pt")
    torch.save(final_checkpoint, run_dir / "checkpoint_final.pt")
    resolved_config = {
        **config,
        "model_run": {
            "name": name,
            "architecture": specification["architecture"],
            "model": model_config,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
        },
        "artifacts": {
            "best_checkpoint": "checkpoint_best.pt",
            "final_checkpoint": "checkpoint_final.pt",
            "results": "results.json",
            "sae_activations": str(resolve_path(config["data"]["cache_dir"])),
        },
    }
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=False)
    )
    result = {
        "architecture": specification["architecture"],
        "model": model_config,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "temporal_consistency_weight": consistency_weight,
        "temporal_consistency_tau": consistency_tau,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "validation": validation_metrics,
        "final_epoch": epochs,
        "final_validation": final_validation_metrics,
        "test": test_metrics,
    }
    (run_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run(config_path: Path):
    config = load_config(config_path)
    device = config["experiment"].get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by this experiment config")
    cache = resolve_path(config["data"]["cache_dir"])
    data = load_data(cache)
    partitions = load_split(cache, data)
    output_dir = resolve_path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "metrics.json"
    summary = {
        "config_path": str(config_path.resolve()),
        "experiment": config["experiment"],
        "data": {
            **config["data"],
            "records": len(data),
            "train_records": len(partitions["train"]),
            "validation_records": len(partitions["validation"]),
            "test_records": len(partitions["test"]),
        },
        "training": config["training"],
        "evaluation": config["evaluation"],
        "runs": {},
    }
    for name, specification in config["models"].items():
        print(f"\n=== {name} ===", flush=True)
        summary["runs"][name] = train_one(
            name,
            specification,
            config,
            partitions,
            output_dir,
            device,
        )
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    print(f"\nWrote {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        action="append",
        help="YAML configuration; repeat this option to run multiple config files",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate the YAML contract and model specifications without training",
    )
    args = parser.parse_args()
    for config_path in args.config:
        if args.validate_only:
            config = load_config(config_path)
            print(f"valid: {config_path} ({len(config['models'])} models)")
        else:
            run(config_path)


if __name__ == "__main__":
    main()
