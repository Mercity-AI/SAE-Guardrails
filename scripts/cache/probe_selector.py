"""Train per-layer linear topic probes and turn their weights into SAE feature rankings."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class ProbeConfig:
    """Hyperparameters shared by every independently fitted layer probe."""

    classes: int
    topk: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    group_lambda: float
    patience: int
    seeds: tuple[int, ...]
    device: str = "cuda"


def make_sample_plan(
    labels: np.ndarray,
    roles: np.ndarray,
    offsets: np.ndarray,
    train_records: np.ndarray,
    token_cap: int,
    seed: int,
    pad: int,
    balance: str = "natural",
    validation_fraction: float = 0.10,
) -> dict[str, np.ndarray]:
    """Sample labeled tokens from a nested record-level split of official training records."""
    rng = np.random.default_rng(seed)
    records = np.asarray(train_records, dtype=np.int64).copy()
    rng.shuffle(records)
    n_validation = max(1, int(round(len(records) * validation_fraction)))
    validation_records = np.sort(records[:n_validation])
    fit_records = np.sort(records[n_validation:])

    def labeled_rows(record_ids: np.ndarray) -> np.ndarray:
        parts = []
        for record in record_ids:
            lo, hi = int(offsets[record]), int(offsets[record + 1])
            local = np.flatnonzero(labels[lo:hi] != pad)
            if len(local):
                parts.append(local.astype(np.int64) + lo)
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)

    fit_candidates = labeled_rows(fit_records)
    validation_candidates = labeled_rows(validation_records)
    available = len(fit_candidates) + len(validation_candidates)
    wanted = available if token_cap <= 0 else min(token_cap, available)
    wanted_validation = min(
        len(validation_candidates), max(1, int(round(wanted * validation_fraction)))
    )
    wanted_fit = min(len(fit_candidates), wanted - wanted_validation)
    # If one side was smaller than its allocation, fill the remainder from the other side.
    shortfall = wanted - wanted_fit - wanted_validation
    if shortfall:
        extra_fit = min(shortfall, len(fit_candidates) - wanted_fit)
        wanted_fit += extra_fit
        wanted_validation += shortfall - extra_fit

    def choose(candidates: np.ndarray, wanted_rows: int) -> np.ndarray:
        if balance == "natural":
            return np.sort(rng.choice(candidates, wanted_rows, replace=False))
        if balance != "topic_role":
            raise ValueError(f"unknown probe sampling balance {balance!r}")
        strata = [
            candidates[(labels[candidates] == topic) & (roles[candidates] == role)]
            for role in np.unique(roles[candidates])
            for topic in np.unique(labels[candidates])
        ]
        strata = [values.copy() for values in strata if len(values)]
        for values in strata:
            rng.shuffle(values)
        picked = []
        remaining = wanted_rows
        active = strata
        # Water-fill scarce strata, then distribute the rest equally over those with capacity.
        while remaining and active:
            share = max(1, remaining // len(active))
            next_active = []
            for values in active:
                take = min(share, len(values))
                if take:
                    picked.append(values[:take])
                    values = values[take:]
                    remaining -= take
                if len(values):
                    next_active.append(values)
                if not remaining:
                    break
            active = next_active
        if remaining:
            raise ValueError(f"could not draw requested balanced probe sample; missing {remaining}")
        return np.sort(np.concatenate(picked))

    fit_rows = choose(fit_candidates, wanted_fit)
    validation_rows = choose(validation_candidates, wanted_validation)
    rows = np.concatenate([fit_rows, validation_rows])
    is_validation = np.concatenate(
        [np.zeros(len(fit_rows), dtype=bool), np.ones(len(validation_rows), dtype=bool)]
    )
    return {
        "rows": rows,
        "is_validation": is_validation,
        "fit_records": fit_records,
        "validation_records": validation_records,
    }


def feature_moments(
    features: np.ndarray,
    rows: np.ndarray,
    device: str,
    chunk_size: int = 4096,
):
    """Compute stable column means/scales with chunked GPU reductions and Welford merging."""
    mean = np.zeros(features.shape[1], dtype=np.float64)
    m2 = np.zeros(features.shape[1], dtype=np.float64)
    count = 0
    for start in range(0, len(rows), chunk_size):
        x = torch.as_tensor(
            np.asarray(features[rows[start : start + chunk_size]], dtype=np.float32),
            device=device,
        )
        chunk_var, chunk_mean = torch.var_mean(x, dim=0, correction=0)
        chunk_mean_np = chunk_mean.cpu().numpy().astype(np.float64)
        chunk_m2 = chunk_var.cpu().numpy().astype(np.float64) * len(x)
        new_count = count + len(x)
        delta = chunk_mean_np - mean
        m2 += chunk_m2 + delta * delta * count * len(x) / max(1, new_count)
        mean += delta * len(x) / max(1, new_count)
        count = new_count
    variance = np.maximum(m2 / max(1, count), 0.0)
    scale = np.sqrt(variance)
    usable = np.isfinite(scale) & (scale > 1e-8)
    scale[~usable] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32), usable


def macro_f1_from_predictions(truth: np.ndarray, predicted: np.ndarray, classes: int) -> float:
    """Unweighted one-vs-rest macro-F1 with an explicit, fixed class set."""
    values = []
    for class_id in range(classes):
        true = truth == class_id
        pred = predicted == class_id
        tp = int(np.count_nonzero(true & pred))
        fp = int(np.count_nonzero(~true & pred))
        fn = int(np.count_nonzero(true & ~pred))
        denominator = 2 * tp + fp + fn
        values.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(values))


@torch.inference_mode()
def predict(
    model: torch.nn.Linear,
    features: np.ndarray,
    rows: np.ndarray,
    mean: torch.Tensor,
    scale: torch.Tensor,
    batch_size: int,
    selected: torch.Tensor | None = None,
) -> np.ndarray:
    """Evaluate a full probe or the same probe with all but selected columns removed."""
    outputs = []
    weight = model.weight
    bias = model.bias
    for start in range(0, len(rows), batch_size):
        x = torch.as_tensor(
            np.asarray(features[rows[start : start + batch_size]], dtype=np.float32),
            device=mean.device,
        )
        z = (x - mean) / scale
        if selected is None:
            logits = torch.nn.functional.linear(z, weight, bias)
        else:
            logits = torch.nn.functional.linear(
                z.index_select(1, selected), weight.index_select(1, selected), bias
            )
        outputs.append(logits.argmax(-1).cpu().numpy())
    return np.concatenate(outputs)


def centered_feature_scores(weight: torch.Tensor, usable: np.ndarray) -> np.ndarray:
    """L2 norm across softmax classes after removing their unidentifiable shared component."""
    centered = weight - weight.mean(0, keepdim=True)
    scores = centered.norm(dim=0).detach().cpu().numpy().astype(np.float64)
    scores[~usable] = -np.inf
    return scores


def fit_one_probe(
    features: np.ndarray,
    labels: np.ndarray,
    fit_rows: np.ndarray,
    validation_rows: np.ndarray,
    config: ProbeConfig,
    method: str,
    seed: int,
    normalization: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, dict]:
    """Fit one 16,384-to-topic probe and return feature scores plus diagnostics."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mean_np, scale_np, usable = normalization
    mean = torch.from_numpy(mean_np).to(config.device)
    scale = torch.from_numpy(scale_np).to(config.device)
    model = torch.nn.Linear(features.shape[1], config.classes).to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    best_score = -1.0
    best_epoch = 0
    best_state = None
    stale = 0

    with torch.enable_grad():
        for epoch in range(1, config.epochs + 1):
            model.train()
            order = rng.permutation(fit_rows)
            for start in range(0, len(order), config.batch_size):
                rows = order[start : start + config.batch_size]
                x = torch.as_tensor(
                    np.asarray(features[rows], dtype=np.float32), device=config.device
                )
                target = torch.as_tensor(labels[rows], dtype=torch.long, device=config.device)
                logits = model((x - mean) / scale)
                loss = loss_fn(logits, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if method == "probe_group" and config.group_lambda > 0:
                    # Proximal group shrinkage removes complete SAE columns across topic classes.
                    with torch.no_grad():
                        weight = model.weight
                        centered = weight - weight.mean(0, keepdim=True)
                        norm = centered.norm(dim=0, keepdim=True)
                        shrink = (1.0 - config.learning_rate * config.group_lambda / norm.clamp_min(1e-12)).clamp_min(0.0)
                        weight.copy_(centered * shrink)

            model.eval()
            val_pred = predict(
                model, features, validation_rows, mean, scale, config.batch_size
            )
            val_score = macro_f1_from_predictions(
                labels[validation_rows], val_pred, config.classes
            )
            if val_score > best_score + 1e-6:
                best_score = val_score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= config.patience:
                    break

    if best_state is None:
        raise RuntimeError("probe training produced no checkpoint")
    model.load_state_dict(best_state)
    scores = centered_feature_scores(model.weight, usable)
    centered_norm = (
        model.weight - model.weight.mean(0, keepdim=True)
    ).norm(dim=0).detach().cpu().numpy()
    selected_np = np.argsort(scores)[::-1][: config.topk].astype(np.int64)
    selected = torch.as_tensor(selected_np, device=config.device)
    reduced_pred = predict(
        model, features, validation_rows, mean, scale, config.batch_size, selected=selected
    )
    diagnostics = {
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_macro_f1_full": best_score,
        "validation_macro_f1_selected_without_refit": macro_f1_from_predictions(
            labels[validation_rows], reduced_pred, config.classes
        ),
        "usable_features": int(usable.sum()),
        "dead_or_constant_features": int((~usable).sum()),
        "nonzero_feature_groups": int(np.count_nonzero(centered_norm > 1e-8)),
        "selected": np.sort(selected_np).tolist(),
    }
    return scores, diagnostics


def fit_layer_probes(
    layer_features: dict[int, np.ndarray],
    sample_labels: np.ndarray,
    is_validation: np.ndarray,
    config: ProbeConfig,
    method: str,
    output_dir: Path,
) -> tuple[dict[int, np.ndarray], dict]:
    """Fit every layer and choose consensus top-K indices from normalized per-seed ranks."""
    fit_rows = np.flatnonzero(~is_validation)
    validation_rows = np.flatnonzero(is_validation)
    selected: dict[int, np.ndarray] = {}
    score_payload = {}
    diagnostics = {
        "method": method,
        "sample_tokens": int(len(sample_labels)),
        "fit_tokens": int(len(fit_rows)),
        "validation_tokens": int(len(validation_rows)),
        "class_counts_fit": np.bincount(
            sample_labels[fit_rows], minlength=config.classes
        ).tolist(),
        "class_counts_validation": np.bincount(
            sample_labels[validation_rows], minlength=config.classes
        ).tolist(),
        "seeds": list(config.seeds),
        "layers": {},
    }
    for layer, features in layer_features.items():
        normalization = feature_moments(features, fit_rows, config.device)
        seed_scores = []
        seed_diagnostics = []
        for seed in config.seeds:
            scores, diag = fit_one_probe(
                features,
                sample_labels,
                fit_rows,
                validation_rows,
                config,
                method,
                seed,
                normalization,
            )
            seed_scores.append(scores)
            seed_diagnostics.append(diag)
        # Rank aggregation avoids coefficient-scale differences between independently fitted seeds.
        rank_scores = []
        for scores in seed_scores:
            order = np.argsort(scores, kind="stable")
            ranks = np.empty_like(order, dtype=np.float64)
            ranks[order] = np.arange(len(order), dtype=np.float64)
            rank_scores.append(ranks / max(1, len(order) - 1))
        consensus = np.mean(rank_scores, axis=0)
        consensus[~np.isfinite(seed_scores[0])] = -np.inf
        choice = np.sort(np.argsort(consensus)[::-1][: config.topk]).astype(np.int64)
        selected[layer] = choice
        score_payload[f"layer_{layer}"] = consensus.astype(np.float32)
        seed_sets = [set(item["selected"]) for item in seed_diagnostics]
        pairwise_jaccard = []
        for left in range(len(seed_sets)):
            for right in range(left + 1, len(seed_sets)):
                pairwise_jaccard.append(
                    len(seed_sets[left] & seed_sets[right])
                    / len(seed_sets[left] | seed_sets[right])
                )
        diagnostics["layers"][str(layer)] = {
            "selected_consensus": choice.tolist(),
            "mean_pairwise_topk_jaccard": (
                float(np.mean(pairwise_jaccard)) if pairwise_jaccard else 1.0
            ),
            "fits": seed_diagnostics,
        }
        print(
            f"  [probe:{method}] layer {layer}: valF1="
            f"{np.mean([x['validation_macro_f1_full'] for x in seed_diagnostics]):.4f} "
            f"topk-stability={diagnostics['layers'][str(layer)]['mean_pairwise_topk_jaccard']:.3f}",
            flush=True,
        )

    np.savez(output_dir / "probe_scores.npz", **score_payload)
    (output_dir / "probe_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    return selected, diagnostics


@torch.no_grad()
def _gpu_moments(x_all: torch.Tensor, fit_idx: torch.Tensor, chunk: int = 8192):
    """Per-layer column mean/scale over fit rows of a stacked (L, N, D) fp16 GPU tensor."""
    n_layers, _, dim = x_all.shape
    device = x_all.device
    total = torch.zeros(n_layers, dim, device=device, dtype=torch.float64)
    total_sq = torch.zeros(n_layers, dim, device=device, dtype=torch.float64)
    for start in range(0, len(fit_idx), chunk):
        idx = fit_idx[start : start + chunk]
        xb = x_all.index_select(1, idx).double()  # (L, b, D)
        total += xb.sum(1)
        total_sq += (xb * xb).sum(1)
    count = max(1, int(len(fit_idx)))
    mean = total / count
    var = torch.clamp(total_sq / count - mean * mean, min=0.0)
    scale = torch.sqrt(var)
    usable = torch.isfinite(scale) & (scale > 1e-8)
    scale = torch.where(usable, scale, torch.ones_like(scale))
    return mean.float(), scale.float(), usable


@torch.no_grad()
def _batched_val_macro_f1(
    weight, bias, x_all, val_idx, mean_e, scale_e, labels, classes, batch_size
):
    """Macro-F1 per layer for the whole (L, C, D) probe stack over the validation rows."""
    n_layers = weight.shape[0]
    tp = torch.zeros(n_layers, classes, device=weight.device)
    fp = torch.zeros(n_layers, classes, device=weight.device)
    fn = torch.zeros(n_layers, classes, device=weight.device)
    for start in range(0, len(val_idx), batch_size):
        idx = val_idx[start : start + batch_size]
        zb = (x_all.index_select(1, idx).float() - mean_e) / scale_e  # (L, b, D)
        logits = torch.einsum("lbd,lcd->lbc", zb, weight) + bias.unsqueeze(1)
        pred = logits.argmax(-1)  # (L, b)
        truth = labels[idx].unsqueeze(0).expand_as(pred)  # (L, b)
        for c in range(classes):
            p, t = pred == c, truth == c
            tp[:, c] += (p & t).sum(1)
            fp[:, c] += (p & ~t).sum(1)
            fn[:, c] += (t & ~p).sum(1)
    denom = 2 * tp + fp + fn
    f1 = torch.where(denom > 0, 2 * tp / denom, torch.zeros_like(denom))
    return f1.mean(1)  # (L,)


def fit_group_sparse_layers(
    layer_features: dict[int, np.ndarray],
    sample_labels: np.ndarray,
    is_validation: np.ndarray,
    config: ProbeConfig,
    output_dir: Path,
    target: int = 0,
    search_rounds: int = 8,
    search_epochs: int = 10,
    lam_lo: float = 1e-3,
    lam_hi: float = 1e3,
) -> tuple[dict[int, np.ndarray], dict]:
    """Select features by *survival* under a strong group-lasso, not by ranking a dense probe.

    All layers are fitted simultaneously on the GPU as one stacked ``(L, C, D)`` linear probe
    (one batched einsum per step). Group-lasso proximal shrinkage zeros whole SAE feature columns
    (group = the ``C`` per-topic weights of one feature); a per-layer penalty ``lambda`` is
    binary-searched so that roughly ``target`` columns survive. The kept indices are the surviving
    columns -- the features the sparse model actually uses -- not the top-K of a dense L2 norm.
    """
    device = config.device
    layers = sorted(layer_features)
    n_layers = len(layers)
    dim = int(next(iter(layer_features.values())).shape[1])
    classes = config.classes
    topk = config.topk
    target = topk if target <= 0 else target
    seed = int(config.seeds[0])
    torch.manual_seed(seed)

    fit_idx = torch.as_tensor(np.flatnonzero(~is_validation), device=device)
    val_idx = torch.as_tensor(np.flatnonzero(is_validation), device=device)
    n_fit = int(len(fit_idx))
    labels_gpu = torch.as_tensor(sample_labels, dtype=torch.long, device=device)

    # Stack every layer's activations on the GPU once (fp16 storage, fp32 math per batch).
    n_rows = int(sample_labels.shape[0])
    x_all = torch.empty((n_layers, n_rows, dim), dtype=torch.float16, device=device)
    for li, layer in enumerate(layers):
        x_all[li] = torch.as_tensor(np.asarray(layer_features[layer]), device=device)
    print(
        f"  [probe:group_sparse] stacked {n_layers}x{n_rows}x{dim} fp16 on GPU "
        f"({x_all.element_size() * x_all.nelement() / 2**30:.1f} GiB)",
        flush=True,
    )
    mean, scale, usable = _gpu_moments(x_all, fit_idx)
    mean_e, scale_e = mean.unsqueeze(1), scale.unsqueeze(1)  # (L, 1, D)
    dead = ~usable  # never let a constant/dead column survive

    def run_fit(lam: torch.Tensor, epochs: int):
        """Train the stacked probe for ``epochs`` at per-layer penalty ``lam``; return column norms."""
        weight = torch.zeros(n_layers, classes, dim, device=device, requires_grad=True)
        bias = torch.zeros(n_layers, classes, device=device, requires_grad=True)
        optimizer = torch.optim.AdamW(
            [weight, bias], lr=config.learning_rate, weight_decay=config.weight_decay
        )
        loss_fn = torch.nn.CrossEntropyLoss()
        thresh = (config.learning_rate * lam).unsqueeze(1)  # (L, 1)
        with torch.enable_grad():  # build script runs under a global no-grad; re-enable here
            for _ in range(epochs):
                perm = fit_idx[torch.randperm(n_fit, device=device)]
                for start in range(0, n_fit, config.batch_size):
                    idx = perm[start : start + config.batch_size]
                    zb = (x_all.index_select(1, idx).float() - mean_e) / scale_e
                    logits = torch.einsum("lbd,lcd->lbc", zb, weight) + bias.unsqueeze(1)
                    target_b = labels_gpu[idx]
                    loss = loss_fn(
                        logits.reshape(n_layers * len(idx), classes),
                        target_b.repeat(n_layers),
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        norm = weight.norm(dim=1)  # (L, D) group = C class weights per feature
                        shrink = (1.0 - thresh / norm.clamp_min(1e-12)).clamp_min(0.0)
                        weight.mul_(shrink.unsqueeze(1))
                        weight[dead.unsqueeze(1).expand_as(weight)] = 0.0
        with torch.no_grad():
            return weight.detach(), bias.detach(), weight.norm(dim=1)  # norms (L, D)

    # ---- per-layer binary search on lambda (log space) to land near `target` survivors ----
    lo = torch.full((n_layers,), float(lam_lo), device=device)
    hi = torch.full((n_layers,), float(lam_hi), device=device)
    lam = torch.sqrt(lo * hi)
    for r in range(search_rounds):
        _, _, norms = run_fit(lam, search_epochs)
        survivors = (norms > 1e-8).sum(1)  # (L,)
        too_many = survivors > target
        lo = torch.where(too_many, lam, lo)  # more survivors -> need a stronger penalty
        hi = torch.where(too_many, hi, lam)
        lam = torch.sqrt(lo * hi)
        print(
            f"  [probe:group_sparse] search {r + 1}/{search_rounds} "
            f"survivors(min/med/max)={int(survivors.min())}/"
            f"{int(survivors.median())}/{int(survivors.max())} target={target}",
            flush=True,
        )

    # ---- final, longer fit at the chosen penalty; keep the surviving columns ----
    weight, bias, norms = run_fit(lam, config.epochs)
    val_f1 = _batched_val_macro_f1(
        weight, bias, x_all, val_idx, mean_e, scale_e, labels_gpu, classes, config.batch_size
    )
    selected: dict[int, np.ndarray] = {}
    diagnostics = {
        "method": "probe_group_sparse",
        "target_survivors_per_layer": int(target),
        "topk": int(topk),
        "seed": seed,
        "search_rounds": int(search_rounds),
        "search_epochs": int(search_epochs),
        "sample_tokens": int(n_rows),
        "fit_tokens": int(n_fit),
        "validation_tokens": int(len(val_idx)),
        "layers": {},
    }
    for li, layer in enumerate(layers):
        col_norm = norms[li]
        survivors = int((col_norm > 1e-8).sum())
        # Keep only columns the group-lasso actually kept: at most `topk`, fewer when the penalty
        # left fewer survivors. Padding up to `topk` would backfill zero-norm columns, and argsort
        # breaks those ties by index, so the layer would silently gain the lowest-numbered (often
        # dead) dictionary features. Layers may therefore differ in width, as in the dynamic caches.
        n_keep = min(topk, survivors)
        order = torch.argsort(col_norm, descending=True)[:n_keep]
        choice = np.sort(order.cpu().numpy()).astype(np.int64)
        selected[layer] = choice
        diagnostics["layers"][str(layer)] = {
            "lambda": float(lam[li]),
            "survivors": survivors,
            "kept": int(len(choice)),
            "survivors_ge_topk": bool(survivors >= topk),
            "validation_macro_f1_full": float(val_f1[li]),
            "selected": choice.tolist(),
        }
        print(
            f"  [probe:group_sparse] layer {layer}: lambda={float(lam[li]):.3g} "
            f"survivors={survivors} kept={len(choice)} valF1={float(val_f1[li]):.4f}",
            flush=True,
        )

    (output_dir / "probe_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    del x_all
    torch.cuda.empty_cache()
    return selected, diagnostics
