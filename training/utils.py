"""
Data loading, splitting, filtering, and result-printing utilities.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset


def collect_text_values(value):
    """Recursively extract all non-empty strings from str/list/dict values.

    Handles nested structures because some HF datasets store messages as lists of dicts
    (e.g. ShareGPT conversation format) rather than flat strings.
    """
    if isinstance(value, str):
        text = " ".join(value.split())
        return [text] if text else []
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(collect_text_values(item))
        return out
    if isinstance(value, dict):
        out = []
        for item in value.values():
            out.extend(collect_text_values(item))
        return out
    return []


def extract_row_text(ex, spec):
    """Concatenate text from the spec's target columns, falling back to all columns."""
    parts = []
    for column in spec["columns"]:
        if column in ex:
            parts.extend(collect_text_values(ex[column]))
    # fallback: scan every column if the target columns yielded nothing
    if not parts:
        for value in ex.values():
            parts.extend(collect_text_values(value))
    return " ".join(" ".join(parts).split())


def load_topic_texts(topic, spec, seed, needed):
    """Stream `needed` rows from a HuggingFace dataset and extract text."""
    print(f"  loading {topic}: {spec['label']} ...")
    kwargs = {"path": spec["hf_id"], "split": spec["split"], "streaming": True}
    if spec["config"]:
        kwargs["name"] = spec["config"]

    try:
        ds = load_dataset(**kwargs)
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" in str(exc):
            raise RuntimeError(
                f"{topic} dataset {spec['hf_id']} is script-bound. "
                "Use a parquet/json/csv-backed Hub dataset instead."
            ) from exc
        raise

    # streaming shuffle only randomizes within the buffer window, not the full dataset
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    texts = []
    for ex in ds:
        text = extract_row_text(ex, spec)
        if text:
            texts.append(text)
        if len(texts) >= needed:
            break

    print(f"    {len(texts)} rows from {spec['columns']}")
    if len(texts) < needed:
        print(f"    warning: requested {needed}, found {len(texts)} usable rows")
    return texts


def load_topic_data(datasets_cfg, n_pos, seed, jsonl_path=None):
    """Load topic texts either from a JSONL file or from HuggingFace datasets.

    If jsonl_path is provided and exists, reads prompts from the JSONL; otherwise
    streams each dataset defined in datasets_cfg from HuggingFace.
    """
    if jsonl_path is not None and Path(jsonl_path).exists():
        data: dict[str, list[str]] = {}
        n_skipped = 0
        with Path(jsonl_path).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                # only use clean generations; failures have prompt=None
                if record.get("_status") != "ok":
                    n_skipped += 1
                    continue
                topic = record["topic"]
                prompt = record["prompt"]
                if topic not in data:
                    data[topic] = []
                if len(data[topic]) < n_pos:
                    data[topic].append(prompt)

        if n_skipped:
            print(f"  skipped {n_skipped} non-ok records")
        if not data:
            raise ValueError(f"no usable records found in {jsonl_path}")
    else:
        data = {}
        for topic, spec in datasets_cfg.items():
            texts = load_topic_texts(topic, spec, seed, n_pos)
            data[topic] = texts[:n_pos]

    for topic, texts in data.items():
        print(f"  {topic}: {len(texts)} samples")
    return data


def split_data(data, ratio, seed=42):
    """Shuffle then split each topic's list into train/test by ratio."""
    rng = np.random.default_rng(seed)
    train, test = {}, {}
    for topic, texts in data.items():
        shuffled = texts.copy()
        rng.shuffle(shuffled)
        split = int(len(shuffled) * ratio)
        train[topic] = shuffled[:split]
        test[topic] = shuffled[split:]
    return train, test


def build_labeled_set(data, topics):
    """Flatten topic→texts dict into (texts, labels) arrays, topics indexed by position."""
    texts, labels = [], []
    for i, topic in enumerate(topics):
        texts.extend(data[topic])
        labels.extend([i] * len(data[topic]))
    return texts, np.array(labels)


def filter_dead_samples(hidden_states, labels, texts, generations, layers, split_name):
    """Drop samples whose first generated token was EOS (zero hidden-state steps)."""
    # all layers share the same dead/alive mask, so checking one layer is sufficient
    ref_layer = layers[0]
    alive_mask = np.array([hs.shape[0] > 0 for hs in hidden_states[ref_layer]])
    n_dead = int((~alive_mask).sum())
    n_total = len(alive_mask)
    print(f"  {split_name}: {n_dead}/{n_total} dead samples (first token was EOS)")

    if n_dead == 0:
        return hidden_states, labels, texts, generations, n_dead

    alive_idx = np.where(alive_mask)[0]
    filtered = {layer: [hidden_states[layer][i] for i in alive_idx] for layer in layers}
    filtered_texts = [texts[i] for i in alive_idx]
    filtered_gens = [generations[i] for i in alive_idx]
    return filtered, labels[alive_idx], filtered_texts, filtered_gens, n_dead


def build_dataset_df(texts, gens, labels, topics, hidden_states_for_ref_layer):
    """Build a DataFrame with prompt metadata and generation stats for one split."""
    topic_names = [topics[l] for l in labels]
    # n_gen_steps is a rough proxy for how much the model generated before stopping
    n_gen_steps = [hs.shape[0] for hs in hidden_states_for_ref_layer]
    return pd.DataFrame(
        {
            "prompt": texts,
            "generation": gens,
            "label": labels.tolist(),
            "topic": topic_names,
            "char_len": [len(t) for t in texts],
            "word_count": [len(t.split()) for t in texts],
            "n_gen_steps": n_gen_steps,
        }
    )


def print_results(preds, labels, per_topic_metrics):
    """Print overall accuracy and per-topic precision/recall/F1."""
    acc = (preds == labels).mean()
    print(f"  overall accuracy: {acc:.4f}")
    for topic, metrics in per_topic_metrics.items():
        print(
            f"    {topic:12s}  "
            f"prec={metrics['precision']:.3f}  "
            f"rec={metrics['recall']:.3f}  "
            f"f1={metrics['f1']:.3f}  "
            f"({metrics['n_test']} samples)"
        )
