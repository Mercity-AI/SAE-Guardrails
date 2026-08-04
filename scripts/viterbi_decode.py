#!/usr/bin/env python3
"""Sticky-Viterbi decode over the frozen convnext posteriors.

No retraining. Take the existing best checkpoint, compute per-token class
posteriors, and replace independent argmax with a Viterbi path that pays a
switch penalty lambda per label change. lambda is swept on validation against
boundary-F1; the winner is reported once on the untouched test split.

Evaluation uses the pipeline's own boundary_report in ORIGINAL token
coordinates (identical to training.py), so numbers are directly comparable to
the stored results.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from utils import (
    PAD,
    TOPICS,
    boundary_positions,
    boundary_report,
    build_model,
    load_data,
    load_split,
)

ROOT = Path(__file__).resolve().parent
K = len(TOPICS)


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


@torch.inference_mode()
def posteriors(model, rows, device):
    """Per row: (full_labels[T], keep_mask[T], logp_kept[n_labeled, K])."""
    model.eval()
    out = []
    for features, labels in rows:
        x = torch.from_numpy(np.asarray(features, dtype=np.float32))[None].to(device)
        logp = model(x)[0].log_softmax(-1).cpu().numpy()  # [T, K]
        keep = labels != PAD
        out.append((labels.copy(), keep, logp[keep]))
    return out


def viterbi(logp: np.ndarray, lam: float) -> np.ndarray:
    """Max-score label path over labeled tokens. Stay costs 0, switch costs -lam."""
    T = logp.shape[0]
    if T == 0:
        return np.empty(0, dtype=np.int64)
    score = logp[0].copy()
    back = np.zeros((T, K), dtype=np.int64)
    for t in range(1, T):
        trans = np.full((K, K), -lam)      # trans[j_from, k_to]
        np.fill_diagonal(trans, 0.0)
        cand = score[:, None] + trans
        best_prev = cand.argmax(0)
        score = cand[best_prev, np.arange(K)] + logp[t]
        back[t] = best_prev
    path = np.empty(T, dtype=np.int64)
    path[-1] = int(score.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode_rows(post, lam):
    """Return pipeline-shaped rows in ORIGINAL coordinates: full labels + full preds."""
    rows = []
    for full_labels, keep, logp in post:
        path = viterbi(logp, lam)
        preds_full = np.full_like(full_labels, PAD)
        preds_full[keep] = path
        rows.append({"labels": full_labels, "predictions": preds_full})
    return rows


def evaluate(rows, tol=(5, 10)):
    truth = np.concatenate([r["labels"][r["labels"] != PAD] for r in rows])
    pred = np.concatenate([r["predictions"][r["labels"] != PAD] for r in rows])
    res = {
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
    }
    for t in tol:
        res[f"boundary_at_{t}"] = boundary_report(rows, t)
    # decomposition: how the model behaves on no-boundary vs one-boundary sequences
    no_b_falsealarms, no_b_count = 0, 0
    has_b_count, has_b_detected = 0, 0
    total_pred = 0
    for r in rows:
        tp = boundary_positions(r["labels"])
        pp = boundary_positions(r["predictions"])
        total_pred += len(pp)
        if len(tp) == 0:
            no_b_count += 1
            no_b_falsealarms += len(pp)
        else:
            has_b_count += 1
            # detected if any predicted boundary within tol=5 of the (single) true one
            if any(abs(int(p) - int(tp[0])) <= 5 for p in pp):
                has_b_detected += 1
    res["diag"] = {
        "n_seqs": len(rows),
        "no_boundary_seqs": no_b_count,
        "false_alarms_on_no_boundary_seqs": no_b_falsealarms,
        "false_alarms_per_no_boundary_seq": no_b_falsealarms / max(no_b_count, 1),
        "boundary_seqs": has_b_count,
        "boundary_seqs_detected_at5": has_b_detected,
        "seq_level_detection_rate": has_b_detected / max(has_b_count, 1),
        "pred_boundaries_per_seq": total_pred / max(len(rows), 1),
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="results/convnext_windows/window_505/checkpoint_best.pt")
    ap.add_argument("--cache", default="cache/sae500_2k_clean_prompt_response")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/viterbi_window_505/sweep.json")
    args = ap.parse_args()

    ckpt = torch.load(resolve(args.checkpoint), map_location="cpu", weights_only=False)
    model = build_model(ckpt["architecture"], ckpt["model"]).to(args.device)
    model.load_state_dict(ckpt["state_dict"])

    cache = resolve(args.cache)
    data = load_data(cache)
    parts = load_split(cache, data)
    val_post = posteriors(model, parts["validation"], args.device)
    test_post = posteriors(model, parts["test"], args.device)

    # sanity: true boundary counts are fixed regardless of lambda
    val_true = sum(len(boundary_positions(l)) for l, _, _ in [(p[0], None, None) for p in val_post])
    test_true = sum(len(boundary_positions(l)) for l, _, _ in [(p[0], None, None) for p in test_post])
    print(f"true boundaries: validation={val_true}  test={test_true}")

    lambdas = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0,
               40.0, 45.0, 50.0, 60.0, 75.0, 100.0, 150.0]
    print(f"\n{'lambda':>7} | {'tok_acc':>7} {'tok_mF1':>7} | {'bP@5':>6} {'bR@5':>6} {'bF1@5':>6} | "
          f"{'pred/sq':>7} {'FA/noB':>7} {'det@5':>6}")
    print("-" * 82)
    sweep, best = [], None
    for lam in lambdas:
        m = evaluate(decode_rows(val_post, lam))
        b, d = m["boundary_at_5"], m["diag"]
        sweep.append({"lambda": lam, "validation": m})
        print(f"{lam:7.1f} | {m['accuracy']:7.3f} {m['macro_f1']:7.3f} | "
              f"{b['precision']:6.3f} {b['recall']:6.3f} {b['f1']:6.3f} | "
              f"{d['pred_boundaries_per_seq']:7.2f} {d['false_alarms_per_no_boundary_seq']:7.2f} "
              f"{d['seq_level_detection_rate']:6.3f}")
        if best is None or b["f1"] > best[1]:
            best = (lam, b["f1"])

    best_lam = best[0]
    test_m = evaluate(decode_rows(test_post, best_lam))
    argmax_m = evaluate(decode_rows(test_post, 0.0))
    print(f"\n=== TEST (lambda={best_lam:.0f} selected by validation bF1@5) ===")
    for tag, mt in [("argmax (lambda=0)", argmax_m), (f"viterbi (lambda={best_lam:.0f})", test_m)]:
        b5, b10, d = mt["boundary_at_5"], mt["boundary_at_10"], mt["diag"]
        print(f"{tag:24s} acc={mt['accuracy']:.3f} mF1={mt['macro_f1']:.3f} | "
              f"bP@5={b5['precision']:.3f} bR@5={b5['recall']:.3f} bF1@5={b5['f1']:.3f} | "
              f"bF1@10={b10['f1']:.3f} | seq_det@5={d['seq_level_detection_rate']:.3f} "
              f"FA/noB={d['false_alarms_per_no_boundary_seq']:.2f}")

    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "checkpoint": str(resolve(args.checkpoint)),
        "selected_lambda": best_lam,
        "true_boundaries": {"validation": val_true, "test": test_true},
        "validation_sweep": sweep,
        "test_argmax": argmax_m,
        "test_viterbi": test_m,
    }, indent=2) + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
