"""
Post-experiment: per-step token-level classification statistics
==================================================================

Loads a checkpoint produced by main.py, re-generates hidden states on the
saved test prompts (best layer only for speed), and answers:
  1. At which generation step does topic classification confidence peak per topic?
  2. Is that step consistent across topics (generalizable)?

Two metrics are tracked jointly at each step t:
  - Cumulative confidence: correct-class MLP probability using OR'd features 0..t
    (matches how the MLP was trained; shows when enough signal has accumulated)
  - Single-step variance: cross-topic variance of SAE features fired only at step t
    (shows which step's raw activations are most intrinsically discriminative)

Usage:
    python training/post_experiments.py
    python training/post_experiments.py --run-prefix checkpoints/gemma-3-1b-it_20260618_112951
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Local imports here
from main import (
    configure_runtime_speed,
    generate_and_capture,
    load_all_saes,
    load_model,
)
from models import TopicClassifier
from utils import filter_dead_samples


# ==========================================================================
# CONFIG
# ==========================================================================

CHECKPOINT_DIR = "checkpoints_new"
MIN_COVERAGE = (
    0.5  # stop iterating steps when fewer than this fraction of samples remain active
)
LOCKIN_THRESHOLD = (
    0.9  # correct-class probability at which a sample is considered "locked in"
)
PLOT_DPI = 150


# ==========================================================================
# CHECKPOINT LOADING
# ==========================================================================


def find_latest_checkpoint(checkpoint_dir):
    """Return the run_prefix of the most recently created checkpoint."""
    meta_files = sorted(Path(checkpoint_dir).glob("*_meta.json"))
    if not meta_files:
        raise FileNotFoundError(f"no checkpoints found in {checkpoint_dir}")
    # filenames are {model_size}_{timestamp}_meta.json; lexicographic sort gives latest last
    return str(meta_files[-1]).replace("_meta.json", "")


def load_checkpoint_artifacts(run_prefix):
    """Load meta.json, MLP weights, and test parquet from a checkpoint run_prefix.

    Returns (meta dict, TopicClassifier on CPU, test DataFrame).
    """
    meta_path = Path(f"{run_prefix}_meta.json")
    mlp_path = Path(f"{run_prefix}_mlp.pt")
    test_path = Path(f"{run_prefix}_test.parquet")

    for p in (meta_path, mlp_path, test_path):
        if not p.exists():
            raise FileNotFoundError(f"checkpoint artifact missing: {p}")

    with meta_path.open() as f:
        meta = json.load(f)

    mlp = TopicClassifier(
        len(meta["feature_ids"]), len(meta["topics"]), meta["mlp_hidden"]
    )
    # weights_only=True prevents arbitrary code execution from the pickle payload
    mlp.load_state_dict(torch.load(mlp_path, map_location="cpu", weights_only=True))
    mlp.eval()

    return meta, mlp, pd.read_parquet(test_path)


# ==========================================================================
# STEP-LEVEL ANALYSIS
# ==========================================================================


def compute_step_stats(hidden_for_layer, labels, topics, sae, mlp, best_feats, device):
    """Compute per-step confidence and feature variance in a single pass over generation steps.

    At each step t (start-aligned, stopped when active samples < MIN_COVERAGE):
      - Updates a cumulative feature matrix (OR of best_feats fired 0..t) and passes
        it through the MLP to get per-sample correct-class probabilities.
      - Computes cross-topic variance of all SAE features fired only at step t.

    Returns a dict with:
      steps              : list of step indices included
      topic_confidence   : {topic -> list of mean correct-class prob per step}
      lockin_steps       : int array (n_samples,), step where prob first >= LOCKIN_THRESHOLD, -1 if never
      singlestep_variance: list of mean cross-topic variance per step
      step_coverage      : fraction of samples still active at each step
    """
    n_samples = len(hidden_for_layer)
    n_steps_per = [hs.shape[0] for hs in hidden_for_layer]
    max_steps = max(n_steps_per, default=0)
    d_sae = sae.d_sae
    n_classes = len(topics)
    best_feats_arr = np.asarray(best_feats)

    # cumulative fired matrix for best_feats only — avoids storing full d_sae across steps
    cum_feat = np.zeros((n_samples, len(best_feats_arr)), dtype=np.float32)
    lockin_steps = np.full(n_samples, -1, dtype=int)

    steps_out, topic_confidence, singlestep_variance, step_coverage = (
        [],
        {t: [] for t in topics},
        [],
        [],
    )

    for t in range(max_steps):
        active_mask = np.array([n > t for n in n_steps_per])
        coverage = float(active_mask.mean())
        if coverage < MIN_COVERAGE:
            break

        active_idx = np.where(active_mask)[0]
        n_active = len(active_idx)

        # stack hidden states at step t for active samples: (n_active, d_model)
        h_tensor = (
            torch.from_numpy(np.stack([hidden_for_layer[i][t] for i in active_idx]))
            .float()
            .to(device)
        )

        with torch.no_grad():
            topk_idx = sae.topk_indices(h_tensor)  # (n_active, k)

            # vectorized scatter: set True at each sample's top-k feature indices along dim=1
            step_fired = torch.zeros(n_active, d_sae, dtype=torch.bool, device=device)
            step_fired.scatter_(1, topk_idx, True)

        step_fired_np = step_fired.cpu().numpy()

        # OR best_feats columns into cumulative — inactive samples keep their previous cum value
        cum_feat[active_idx] = np.maximum(
            cum_feat[active_idx],
            step_fired_np[:, best_feats_arr].astype(np.float32),
        )

        # --- single-step cross-topic variance (all d_sae features) ---
        active_labels = labels[active_idx]
        topic_rates = [
            step_fired_np[active_labels == cls_i].mean(axis=0)
            for cls_i in range(n_classes)
            if (active_labels == cls_i).any()
        ]
        if len(topic_rates) == n_classes:
            singlestep_variance.append(float(np.stack(topic_rates).var(axis=0).mean()))
        else:
            singlestep_variance.append(0.0)

        # --- cumulative MLP confidence ---
        with torch.no_grad():
            X = torch.tensor(cum_feat, dtype=torch.float32).to(device)
            probs = F.softmax(mlp(X), dim=-1).cpu().numpy()  # (n_samples, n_classes)

        for cls_i, topic in enumerate(topics):
            cls_mask = labels == cls_i
            if cls_mask.any():
                topic_confidence[topic].append(float(probs[cls_mask, cls_i].mean()))

        # lock-in: first step where correct-class prob reaches threshold
        for i in range(n_samples):
            if lockin_steps[i] == -1 and probs[i, int(labels[i])] >= LOCKIN_THRESHOLD:
                lockin_steps[i] = t

        steps_out.append(t)
        step_coverage.append(coverage)

    return {
        "steps": steps_out,
        "topic_confidence": topic_confidence,
        "lockin_steps": lockin_steps,
        "singlestep_variance": singlestep_variance,
        "step_coverage": step_coverage,
    }


# ==========================================================================
# PLOTTING
# ==========================================================================


def plot_confidence_curves(stats, topics, out_path):
    """Plot mean correct-class cumulative MLP probability per topic vs generation step."""
    fig, ax1 = plt.subplots(figsize=(11, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(topics)))
    steps = stats["steps"]

    for i, topic in enumerate(topics):
        conf = stats["topic_confidence"].get(topic, [])
        if conf:
            ax1.plot(
                steps[: len(conf)], conf, label=topic, color=colors[i], linewidth=2
            )

    ax1.axhline(
        LOCKIN_THRESHOLD,
        color="gray",
        linestyle="--",
        alpha=0.6,
        label=f"lock-in threshold ({LOCKIN_THRESHOLD})",
    )
    ax1.set_xlabel("Generation step")
    ax1.set_ylabel("Mean correct-class probability (cumulative)")
    ax1.set_title("How fast does classification confidence build up per topic?")
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="lower right", fontsize=8)

    # shade coverage on a secondary axis so the reader knows when the curve gets sparse
    ax2 = ax1.twinx()
    ax2.fill_between(steps, stats["step_coverage"], alpha=0.08, color="gray")
    ax2.set_ylabel("Sample coverage", color="gray", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="gray")
    ax2.set_ylim(0, 1.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI)
    plt.close()
    print(f"  saved: {out_path}")


def plot_lockin_distribution(stats, labels, topics, out_path):
    """Box-plot the lock-in step distribution per topic — directly answers Q1 and Q2.

    Tight boxes = consistent lock-in within a topic (Q1 yes).
    Similar box positions across topics = generalizable step (Q2 yes).
    """
    fig, ax = plt.subplots(figsize=(max(6, len(topics) * 1.5), 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(topics)))

    box_data, tick_labels = [], []
    for cls_i, topic in enumerate(topics):
        cls_mask = labels == cls_i
        lockin = stats["lockin_steps"][cls_mask]
        reached = lockin[lockin >= 0].tolist()
        box_data.append(reached)
        pct = 100 * len(reached) / max(int(cls_mask.sum()), 1)
        tick_labels.append(f"{topic}\n({pct:.0f}% locked in)")

    bp = ax.boxplot(box_data, patch_artist=True, notch=False)
    ax.set_xticks(range(1, len(tick_labels) + 1))
    ax.set_xticklabels(tick_labels)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor((*color[:3], 0.45))

    ax.set_ylabel("Generation step at lock-in")
    ax.set_title(
        f"Lock-in step distribution per topic  (threshold={LOCKIN_THRESHOLD})\n"
        "Tight boxes → consistent within topic · Similar medians → generalizable across topics"
    )
    plt.xticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI)
    plt.close()
    print(f"  saved: {out_path}")


def plot_singlestep_variance(stats, out_path):
    """Plot cross-topic SAE feature variance at each individual step (no cumulation).

    The peak marks the single generation step whose raw activations are most
    discriminative — regardless of what happened in earlier steps.
    """
    steps = stats["steps"]
    variance = stats["singlestep_variance"]

    fig, ax1 = plt.subplots(figsize=(11, 4))
    ax1.plot(
        steps, variance, color="steelblue", linewidth=2, label="cross-topic variance"
    )

    if variance:
        peak_idx = int(np.argmax(variance))
        ax1.axvline(
            steps[peak_idx],
            color="crimson",
            linestyle="--",
            alpha=0.7,
            label=f"peak at step {steps[peak_idx]} (var={variance[peak_idx]:.5f})",
        )

    ax1.set_xlabel("Generation step")
    ax1.set_ylabel("Mean cross-topic variance of SAE features")
    ax1.set_title("Which single generation step's activations are most discriminative?")
    ax1.legend()

    ax2 = ax1.twinx()
    ax2.fill_between(steps, stats["step_coverage"], alpha=0.08, color="gray")
    ax2.set_ylabel("Sample coverage", color="gray", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="gray")
    ax2.set_ylim(0, 1.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI)
    plt.close()
    print(f"  saved: {out_path}")


# ==========================================================================
# MAIN ANALYSIS RUNNER
# ==========================================================================


def run_analysis(run_prefix):
    """Load checkpoint artifacts, re-generate hidden states, compute and save all step analyses."""
    print(f"Checkpoint: {run_prefix}")
    meta, mlp, test_df = load_checkpoint_artifacts(run_prefix)

    topics = meta["topics"]
    best_layer = meta["best_layer"]
    best_feats = meta["feature_ids"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    configure_runtime_speed()
    mlp = mlp.to(device)

    print(f"  model: {meta['model_name']} | topics: {topics}")
    print(f"  best layer: L{best_layer}  |  n features: {len(best_feats)}")

    # load model and only the SAE for the best layer — skip all other layers
    model, tok = load_model(meta["model_name"], device, dtype)
    saes = load_all_saes(
        meta["sae_repo"],
        [best_layer],
        model.config.hidden_size,
        meta["top_k_sae"],
        device,
    )
    sae = saes[best_layer]

    print(f"\nRe-generating {len(test_df)} test prompts (L{best_layer} only) ...")
    test_texts = test_df["prompt"].tolist()
    test_labels = test_df["label"].values

    hidden, _ = generate_and_capture(
        test_texts,
        model,
        tok,
        [best_layer],
        device,
        batch_size=meta.get("batch_size", 16),
    )

    # re-use existing filter to drop any samples that immediately hit EOS on re-generation
    dummy_gens = [""] * len(test_texts)
    hidden, test_labels, test_texts, _, n_dead = filter_dead_samples(
        hidden, test_labels, test_texts, dummy_gens, [best_layer], "test (re-gen)"
    )
    if n_dead:
        print(f"  note: {n_dead} samples dropped (hit EOS immediately on re-gen)")

    # free model memory before the step analysis loop, which is CPU/GPU memory intensive
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    print("\nComputing per-step statistics ...")
    stats = compute_step_stats(
        hidden[best_layer], test_labels, topics, sae, mlp, best_feats, device
    )
    n_steps_analyzed = len(stats["steps"])
    print(f"  analyzed {n_steps_analyzed} steps (coverage >= {MIN_COVERAGE:.0%})")

    # --- print summary ---
    print("\n--- Lock-in step summary (Q1: is there a consistent step per topic?) ---")
    all_medians = []
    for cls_i, topic in enumerate(topics):
        cls_mask = test_labels == cls_i
        lockin = stats["lockin_steps"][cls_mask]
        reached = lockin[lockin >= 0]
        if len(reached):
            med = float(np.median(reached))
            std = float(np.std(reached))
            all_medians.append(med)
            print(
                f"  {topic:30s}  median={med:.0f}  std={std:.1f}  "
                f"({len(reached)}/{int(cls_mask.sum())} samples reached threshold)"
            )
        else:
            print(f"  {topic:30s}  no samples reached threshold")

    if len(all_medians) > 1:
        spread = max(all_medians) - min(all_medians)
        print("\n--- Q2: generalizability across topics ---")
        print(f"  median lock-in range across topics: {spread:.0f} steps")
        print(
            f"  {'→ consistent (< 10 step spread)' if spread < 10 else '→ varies across topics'}"
        )

    if stats["singlestep_variance"]:
        peak_idx = int(np.argmax(stats["singlestep_variance"]))
        print(
            f"\n  peak single-step feature variance at step: {stats['steps'][peak_idx]}"
        )

    # --- save plots ---
    out_dir = Path(run_prefix).parent
    prefix = Path(run_prefix).name

    plot_confidence_curves(stats, topics, out_dir / f"{prefix}_step_confidence.png")
    plot_lockin_distribution(
        stats, test_labels, topics, out_dir / f"{prefix}_lockin_dist.png"
    )
    plot_singlestep_variance(stats, out_dir / f"{prefix}_singlestep_variance.png")

    print(f"\nAll plots saved to {out_dir}/")


def main():
    """Entry point: resolve checkpoint and run step-level analysis."""
    parser = argparse.ArgumentParser(
        description="Per-step SAE classification analysis."
    )
    parser.add_argument(
        "--run-prefix",
        default=None,
        help="Path prefix for a specific checkpoint (e.g. checkpoints_new/1.7B_20240617_120000). "
        "Defaults to the latest checkpoint in CHECKPOINT_DIR.",
    )
    args = parser.parse_args()

    run_prefix = args.run_prefix or find_latest_checkpoint(CHECKPOINT_DIR)
    run_analysis(run_prefix)


if __name__ == "__main__":
    main()
