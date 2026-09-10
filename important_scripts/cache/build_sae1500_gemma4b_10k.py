#!/usr/bin/env python3
"""Build a 150-features-per-layer Gemma-3-4B SAE cache (sae1500) from the 10k run.

Mirrors the Gemma-1B sae1500 pipeline (build_1500_caches.py / build_sae1500_cache.py)
exactly in *convention*, adapted to the 4B backbone:
  * base google/gemma-3-4b-it (bf16, flash_attention_2), SAE gemma-scope-2-4b-it-res-all,
    layers 24-25..33 (10 layers), width_16k_l0_small
  * manual JumpReLU encode (same as 1B): pre = (act - b_dec) @ W_enc + b_enc ;
    feat = pre * (pre > threshold)     (SAE cfg apply_b_dec_to_input=False)
  * teacher-forced prefill over strip_tags(prompt)+strip_tags(response) via label_record
  * top-150 features/layer chosen by summed JumpReLU over ALL tokens of TRAIN records
  * response-only supervision comes straight from label_record (prompt tokens -> PAD)

Throughput: one large batched flash-attention prefill pass to SELECT (train only), a second
to ENCODE (all records). Records are length-sorted and packed to a token budget; a background
thread prefetches/pads the next batch while the GPU runs; OOM auto-splits a batch. Hidden
states are captured in the model's native bf16 (no fp16 overflow); the SAE matmul is done in
fp32 for fidelity; stored features are fp16 (clamped to finite range, clamp count reported).

Outputs (numpy cache + a single GPU-resident torch pack, like build_torch_cache_sae150.py):
  features.npy(fp16, N x1500) labels.npy lengths.npy role_ids.npy split_indices.npz
  selected_features.npz metadata.json gpu_cache.pt
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed
from sae_lens import SAE
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from transformers import AutoModelForCausalLM, AutoTokenizer

# local imports
from important_scripts.paths import PROJECT_ROOT
from important_scripts.cache.label_records import label_record, validate_topic_tags
from important_scripts.model.models import PAD, TOPICS
from important_scripts.cache.probe_selector import (
    ProbeConfig,
    fit_group_sparse_layers,
    fit_layer_probes,
    make_sample_plan,
)

ROOT = PROJECT_ROOT


# Supervised univariate feature scorers (SelectKBest family). Each maps a labeled
# (tokens x 16,384) activation matrix + topic ids to a per-feature score; we keep the
# top-K by that score, replacing the unsupervised summed-magnitude criterion.
#   f_classif    -- ANOVA F: does the feature's mean activation differ across the 7 topics?
#   chi2         -- chi-squared dependence (needs X >= 0; JumpReLU already is).
#   firing_rate_contrast -- max topic firing rate minus the mean topic firing rate.
#   mutual_info  -- mutual information (non-parametric; catches nonlinear/thresholded links).
SELECT_METHODS = (
    "summation",
    "firing_rate_contrast",
    "f_classif",
    "chi2",
    "mutual_info",
    "probe_l2",
    "probe_group",
)


def select_topk_supervised(
    method: str, features: np.ndarray, labels: np.ndarray, topk: int, seed: int
) -> np.ndarray:
    """Return the top-`topk` feature indices for one layer via sklearn ``SelectKBest``.

    ``features`` is a (n_labeled_tokens, D_SAE) float32 matrix of JumpReLU activations for a
    subsample of TRAIN tokens; ``labels`` are their topic ids. This is literally
    ``SelectKBest(f_classif|chi2|mutual_info, k=topk)`` -- SelectKBest ranks by the score and
    keeps the k highest, cleaning NaN scores (never-firing / constant columns) to the minimum
    internally, so dead features are never selected. Indices are returned sorted ascending,
    matching the summation path's convention.
    """
    if method == "chi2":
        features = np.clip(features, 0.0, None)  # chi2 requires non-negative inputs
        score_func = chi2
    elif method == "f_classif":
        score_func = f_classif
    elif method == "mutual_info":
        # pin the estimator RNG so the selection is reproducible run-to-run
        def score_func(x, y):
            return mutual_info_classif(x, y, random_state=seed)
    else:
        raise ValueError(f"unknown supervised selection method: {method}")
    # Many of the 16,384 columns never fire in a subsample -> constant -> NaN F/chi2 score;
    # SelectKBest cleans those to the minimum and drops them, so the warnings are expected noise.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selector = SelectKBest(score_func=score_func, k=topk).fit(features, labels)
    return np.sort(selector.get_support(indices=True)).astype(np.int64)


def scores_from_class_moments(
    method: str, count: np.ndarray, csum: np.ndarray, csumsq: np.ndarray
) -> np.ndarray:
    """Per-feature ANOVA-F or chi2 score from streamed per-class moments over ALL tokens.

    ``count`` is (K,) labeled-token count per topic; ``csum``/``csumsq`` are (K, D) per-class
    feature sum and sum-of-squares. These fully determine both statistics, so they are computed
    exactly over every token without ever storing the token x feature matrix:

      f_classif : F = (SSB/(K-1)) / (SSW/(N-K)), SSB/SSW the between/within-class sums of squares
                  (matches sklearn.feature_selection.f_classif).
      chi2      : sum_c (observed-expected)^2 / expected, observed = per-class feature sums,
                  expected = class_prob (x) feature_total (matches sklearn's chi2 on non-negative X).

    Degenerate features (constant -> 0/0) score -inf so SelectKBest-style ranking never keeps them.
    """
    count = count.astype(np.float64)
    csum = csum.astype(np.float64)
    csumsq = csumsq.astype(np.float64)
    n_total = count.sum()
    n_classes = count.shape[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        if method == "f_classif":
            class_mean = csum / count[:, None]
            grand_mean = csum.sum(0) / n_total
            ssb = (count[:, None] * (class_mean - grand_mean) ** 2).sum(0)
            ssw = (csumsq - count[:, None] * class_mean**2).sum(0)
            scores = (ssb / (n_classes - 1)) / (ssw / (n_total - n_classes))
        elif method == "chi2":
            observed = np.clip(csum, 0.0, None)
            expected = np.outer(count / n_total, observed.sum(0))
            scores = np.where(expected > 0, (observed - expected) ** 2 / expected, 0.0).sum(0)
        else:
            raise ValueError(f"streaming moments not defined for method {method!r}")
    return np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)


def scores_from_firing_counts(count: np.ndarray, fire_count: np.ndarray) -> np.ndarray:
    """Score features by their per-topic nonzero firing-rate contrast.

    ``count`` contains the labeled-token count for each topic and ``fire_count[c, f]``
    contains the number of those tokens on which feature ``f`` is nonzero. The score is
    ``max_c(rate[c, f]) - mean_c(rate[c, f])`` over topics represented in the training split.
    A feature firing at the same rate for every topic therefore scores zero, regardless of
    how frequently or strongly it fires.
    """
    count = np.asarray(count, dtype=np.float64)
    fire_count = np.asarray(fire_count, dtype=np.float64)
    if fire_count.ndim != 2 or fire_count.shape[0] != count.shape[0]:
        raise ValueError("fire_count must have shape (len(count), n_features)")
    present = count > 0
    if not present.any():
        return np.full(fire_count.shape[1], -np.inf, dtype=np.float64)
    rates = fire_count[present] / count[present, None]
    return rates.max(axis=0) - rates.mean(axis=0)


def _fit_layer(method, features, labels, topk, seed, layer):
    """Worker wrapper: run one layer's SelectKBest fit, return (layer, indices, seconds).

    Each layer is independent, so the 10 per-layer fits run in parallel processes. This
    matters for mutual_info, whose sklearn estimator is single-threaded -- 10-way parallelism
    turns a ~40 min serial fit into a few minutes.
    """
    t = time.time()
    idx = select_topk_supervised(method, features, labels, topk, seed)
    return layer, idx, time.time() - t
MODEL_ID = "google/gemma-3-4b-it"
SAE_RELEASE = "gemma-scope-2-4b-it-res-all"
LAYERS = list(range(24, 34))  # 10 layers
HIDDEN = 2560
TOPK = 150
WIDTH = TOPK * len(LAYERS)  # 1500
D_SAE = 16384
DEVICE = "cuda"
SEED = 42
FP16_MAX = 65504.0


# ---------------------------------------------------------------- backbone
def find_backbone(model):
    """Locate the Gemma text-transformer submodule (the one owning ``.layers``) across HF variants."""
    for cand in (
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(model, "language_model", None),
        getattr(model, "model", None),
    ):
        if cand is not None and hasattr(cand, "layers"):
            return cand
    raise TypeError("could not locate Gemma text backbone layers")


# ---------------------------------------------------------------- batching
def make_batches(indices, lengths, token_budget, max_seqs):
    """Greedy length-sorted packing to a max-token budget (indices -> list[list[idx]])."""
    order = sorted(indices, key=lambda i: lengths[i])
    batches, cur, cur_max = [], [], 0
    for i in order:
        seq_len = int(lengths[i])
        new_max = max(cur_max, seq_len)
        if cur and (new_max * (len(cur) + 1) > token_budget or len(cur) >= max_seqs):
            batches.append(cur)
            cur, cur_max = [], 0
            new_max = seq_len
        cur.append(i)
        cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


class Prefetcher:
    """Builds padded (input_ids, attention, meta) CPU tensors for the next batch in a thread."""

    def __init__(self, batches, ids_all, offsets, pad_id):
        self.batches, self.ids_all, self.offsets, self.pad_id = batches, ids_all, offsets, pad_id
        self.idx = 0
        self.lock = threading.Lock()
        self.nxt = None
        self._build_next()

    def _pack(self, batch):
        seqs = [self.ids_all[i] for i in batch]
        width = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), width), self.pad_id, dtype=torch.long)
        attention = torch.zeros((len(seqs), width), dtype=torch.long)
        rows = []
        for r, (gi, s) in enumerate(zip(batch, seqs, strict=False)):
            input_ids[r, : len(s)] = torch.tensor(s, dtype=torch.long)
            attention[r, : len(s)] = 1
            rows.append(np.arange(int(self.offsets[gi]), int(self.offsets[gi + 1])))
        rows_global = np.concatenate(rows) if rows else np.zeros(0, np.int64)
        return {
            "input_ids": input_ids.pin_memory(),
            "attention": attention.pin_memory(),
            "rows_global": rows_global,
            "batch": batch,
        }

    def _build_next(self):
        if self.idx >= len(self.batches):
            self.nxt = None
            return
        b = self.batches[self.idx]
        self.idx += 1
        self.nxt = self._pack(b)

    def __iter__(self):
        while self.nxt is not None:
            item = self.nxt
            t = threading.Thread(target=self._build_next)
            t.start()
            yield item
            t.join()


# ---------------------------------------------------------------- SAE math
def jumprelu(h_fp32, W, b_enc, b_dec, thr):
    """Manual Gemma-Scope JumpReLU encode: (h - b_dec) @ W_enc + b_enc, gated by pre > threshold."""
    pre = (h_fp32 - b_dec) @ W + b_enc
    return pre * (pre > thr)


@torch.inference_mode()
def run_backbone(backbone, captured, item):
    """Teacher-forced forward over one padded batch; layer hooks fill ``captured`` with hidden states."""
    captured.clear()
    backbone(
        input_ids=item["input_ids"].to(DEVICE, non_blocking=True),
        attention_mask=item["attention"].to(DEVICE, non_blocking=True),
        use_cache=False,
        return_dict=True,
    )
    if set(captured) != set(LAYERS):
        raise RuntimeError(f"missing hooked layers: {set(LAYERS) - set(captured)}")


def forward_with_oom_guard(fn, item, ids_all, offsets, pad_id):
    """Run fn(item); on CUDA OOM split the batch in half and recurse."""
    try:
        return fn(item)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        batch = item["batch"]
        if len(batch) == 1:
            raise
        mid = len(batch) // 2
        pf = Prefetcher([batch[:mid], batch[mid:]], ids_all, offsets, pad_id)
        for sub in pf:
            forward_with_oom_guard(fn, sub, ids_all, offsets, pad_id)


def main(a):
    """Two-pass build: select top-150/layer over TRAIN tokens, then encode all records to features.npy."""
    global MODEL_ID, SAE_RELEASE, LAYERS, TOPK, WIDTH, HIDDEN, DTYPE
    MODEL_ID = a.model
    SAE_RELEASE = a.sae_release
    TOPK = a.topk
    _lo, _hi = a.layers.split("-")
    LAYERS = list(range(int(_lo), int(_hi) + 1))
    WIDTH = TOPK * len(LAYERS)
    DTYPE = np.float16 if a.dtype == "fp16" else np.float32
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_grad_enabled(False)
    # TF32 tensor cores: ~5x faster fp32 matmuls; features are stored fp16 so the
    # tf32 mantissa (>fp16 precision) is not the limiting error. Selection ranking
    # and the 2560-contraction encode are both well within fp16's tolerance.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)

    # ---- label every record (ids, roles, labels) ----
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "right"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    t0 = time.time()
    recs = [json.loads(l)["record"] for l in Path(a.dataset).open()]
    if a.limit:
        recs = recs[: a.limit]
    invalid_tags = []
    for index, record in enumerate(recs):
        errors = validate_topic_tags(record)
        if errors:
            invalid_tags.append({"record": index, "errors": errors})
    if invalid_tags:
        message = f"{len(invalid_tags)} records contain invalid topic tags; first={invalid_tags[0]}"
        if a.tag_validation == "error":
            raise ValueError(message)
        if a.tag_validation == "warn":
            print(f"WARNING: {message}", flush=True)
        (out / "invalid_tags.json").write_text(json.dumps(invalid_tags, indent=2) + "\n")
    ids_all, roles_all, labels_all = [], [], []
    for k, r in enumerate(recs):
        ii, rr, ll = label_record(r, tok, supervise_prompt=a.supervise_prompt)
        ids_all.append(ii)
        roles_all.append(np.asarray(rr, np.int8))
        labels_all.append(np.asarray(ll, np.int64))
        if (k + 1) % 2000 == 0:
            print(f"  labeled {k + 1}/{len(recs)} ({time.time() - t0:.0f}s)", flush=True)
    lengths = np.asarray([len(x) for x in ids_all], np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    N = int(offsets[-1])
    labels = np.concatenate(labels_all).astype(np.int64)
    roles = np.concatenate(roles_all).astype(np.int8)
    print(f"records={len(recs)} tokens={N} ({time.time() - t0:.0f}s)", flush=True)

    # ---- record-level split 70/15/15 (seed 42) ----
    # canonical split: SAME RNG + ratios as stage1_label.py / models.split_indices, so the
    # 4B test set is IDENTICAL to the 1B 10k test set (decode manifest reuse + comparability).
    perm = np.random.RandomState(SEED).permutation(len(recs))
    n_tr = int(0.70 * len(recs))
    n_va = n_tr + int(0.15 * len(recs))
    train_rec = np.sort(perm[:n_tr])
    val_rec = np.sort(perm[n_tr:n_va])
    test_rec = np.sort(perm[n_va:])
    print(f"split train={len(train_rec)} val={len(val_rec)} test={len(test_rec)}", flush=True)

    # ---- model + hooks ----
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, attn_implementation=a.attn
        )
        .to(DEVICE)
        .eval()
    )
    backbone = find_backbone(model)
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for L in LAYERS:

        def hook(_m, _i, o, L=L):
            captured[L] = o[0] if isinstance(o, tuple) else o

        handles.append(backbone.layers[L].register_forward_hook(hook))

    # ---- SAE params (fp32, GPU) ----
    saep = {}
    for L in LAYERS:
        s = SAE.from_pretrained(
            release=SAE_RELEASE, sae_id=f"layer_{L}_width_16k_l0_small", device=DEVICE
        )
        s = s[0] if isinstance(s, tuple) else s
        s.eval()
        saep[L] = (s.W_enc.float(), s.b_enc.float(), s.b_dec.float(), s.threshold.float())
        del s
    HIDDEN = int(saep[LAYERS[0]][0].shape[0])  # 1152 (1B) or 2560 (4B)
    torch.cuda.empty_cache()
    print(
        f"loaded {len(saep)} SAEs; model+SAE VRAM={torch.cuda.memory_allocated() / 2**30:.1f} GiB",
        flush=True,
    )

    # ================= PASS 1: select top-150/layer over TRAIN tokens =================
    # Three selection regimes share this single prefill pass:
    #   summation             -- unsupervised: rank features by summed JumpReLU magnitude.
    #   firing_rate_contrast  -- supervised, STREAMED: accumulate per-class nonzero counts over
    #     ALL labeled train tokens, then rank by max topic rate minus mean topic rate.
    #   f_classif / chi2      -- supervised, STREAMED: accumulate per-class sum/sumsq per feature
    #     over ALL labeled train tokens, then score exactly -- no subsample, ~0 RAM.
    #   mutual_info           -- supervised, SUBSAMPLED: mutual_info_classif can't be streamed, so
    #     keep a Bernoulli subsample of labeled tokens and fit the 10 layers in parallel processes.
    method = a.select
    n_classes = len(TOPICS)
    tr_batches = make_batches(list(train_rec), lengths, a.token_budget, a.max_seqs)
    print(f"[select:{method}] {len(tr_batches)} batches over {len(train_rec)} train records", flush=True)
    t1 = time.time()
    done = 0

    if method == "summation":
        sums = {L: torch.zeros(D_SAE, device=DEVICE) for L in LAYERS}

        def sel_fn(item):
            run_backbone(backbone, captured, item)
            mask = item["attention"].to(DEVICE, non_blocking=True).bool()
            for L in LAYERS:
                h = captured[L][mask].float()  # (n_real, hidden)
                W, be, bd, thr = saep[L]
                sums[L] += jumprelu(h, W, be, bd, thr).sum(0)
    elif method in ("firing_rate_contrast", "f_classif", "chi2"):
        # Streamed per-class counts or moments over EVERY labeled train token (exact, tiny memory).
        cls_count = torch.zeros(n_classes, device=DEVICE, dtype=torch.float64)
        if method == "firing_rate_contrast":
            cls_fire = {
                L: torch.zeros(n_classes, D_SAE, device=DEVICE, dtype=torch.float64)
                for L in LAYERS
            }
        else:
            cls_sum = {
                L: torch.zeros(n_classes, D_SAE, device=DEVICE, dtype=torch.float64)
                for L in LAYERS
            }
            cls_sumsq = {
                L: torch.zeros(n_classes, D_SAE, device=DEVICE, dtype=torch.float64)
                for L in LAYERS
            }

        def sel_fn(item):
            run_backbone(backbone, captured, item)
            mask = item["attention"].to(DEVICE, non_blocking=True).bool()
            rows = item["rows_global"]
            y = labels[rows]
            keep = y != PAD
            if not keep.any():
                return
            pick = torch.as_tensor(np.where(keep)[0], device=DEVICE)
            yk = torch.as_tensor(y[keep], device=DEVICE)
            onehot = torch.zeros(len(yk), n_classes, device=DEVICE)
            onehot[torch.arange(len(yk), device=DEVICE), yk] = 1.0  # (n_lab, K)
            cls_count.add_(onehot.sum(0).double())
            for L in LAYERS:
                h = captured[L][mask][pick].float()  # (n_lab, hidden)
                feat = jumprelu(h, *saep[L])  # (n_lab, D_SAE)
                if method == "firing_rate_contrast":
                    cls_fire[L].add_((onehot.t() @ feat.ne(0).float()).double())
                else:
                    cls_sum[L].add_((onehot.t() @ feat).double())
                    cls_sumsq[L].add_((onehot.t() @ (feat * feat)).double())
    elif method == "mutual_info":  # subsample + parallel fit
        train_tok = np.zeros(N, dtype=bool)
        for r in train_rec:
            train_tok[offsets[r] : offsets[r + 1]] = True
        n_labeled = int(((labels != PAD) & train_tok).sum())
        p_keep = min(1.0, a.select_token_cap / max(1, n_labeled)) if a.select_token_cap else 1.0
        print(f"[select:{method}] labeled train tokens={n_labeled}  keep_prob={p_keep:.4f}", flush=True)
        sel_rng = np.random.default_rng(seed=SEED)
        buf_x: dict[int, list] = {L: [] for L in LAYERS}
        buf_y: list = []

        def sel_fn(item):
            run_backbone(backbone, captured, item)
            mask = item["attention"].to(DEVICE, non_blocking=True).bool()
            rows = item["rows_global"]
            y = labels[rows]
            take = (y != PAD) & (sel_rng.random(len(rows)) < p_keep)
            if not take.any():
                return
            pick = torch.as_tensor(np.where(take)[0], device=DEVICE)
            buf_y.append(y[take].astype(np.int64))
            for L in LAYERS:
                h = captured[L][mask][pick].float()
                buf_x[L].append(jumprelu(h, *saep[L]).half().cpu().numpy())

    else:  # learned per-token linear probe over the complete dictionary
        probe_plan = make_sample_plan(
            labels,
            roles,
            offsets,
            train_rec,
            a.probe_token_cap,
            SEED,
            PAD,
            balance=a.probe_balance,
            validation_fraction=a.probe_validation_fraction,
        )
        probe_rows = probe_plan["rows"]
        probe_lut = np.full(N, -1, dtype=np.int32)
        probe_lut[probe_rows] = np.arange(len(probe_rows), dtype=np.int32)
        probe_x = {
            L: np.empty((len(probe_rows), D_SAE), dtype=np.float16) for L in LAYERS
        }
        print(
            f"[select:{method}] buffering {len(probe_rows)} labeled TRAIN tokens x "
            f"{D_SAE} features x {len(LAYERS)} layers "
            f"({len(probe_rows) * D_SAE * len(LAYERS) * 2 / 2**30:.1f} GiB host)",
            flush=True,
        )

        def sel_fn(item):
            run_backbone(backbone, captured, item)
            mask = item["attention"].to(DEVICE, non_blocking=True).bool()
            rows = item["rows_global"]
            destination = probe_lut[rows]
            take = destination >= 0
            if not take.any():
                return
            pick = torch.as_tensor(np.flatnonzero(take), device=DEVICE)
            destination = destination[take]
            for L in LAYERS:
                h = captured[L][mask][pick].float()
                probe_x[L][destination] = jumprelu(h, *saep[L]).half().cpu().numpy()

    for bi, item in enumerate(Prefetcher(tr_batches, ids_all, offsets, pad_id)):
        forward_with_oom_guard(sel_fn, item, ids_all, offsets, pad_id)
        done += len(item["batch"])
        if (bi + 1) % 10 == 0 or bi + 1 == len(tr_batches):
            mem = torch.cuda.max_memory_allocated() / 2**30
            print(
                f"  [select] {done}/{len(train_rec)} recs  {done and (time.time() - t1) / (bi + 1):.2f}s/batch  peakVRAM={mem:.1f}GiB",
                flush=True,
            )

    if method == "summation":
        selected = {L: torch.topk(sums[L], TOPK).indices.sort().values for L in LAYERS}
        sel_np = {f"layer_{L}": selected[L].cpu().numpy().astype(np.int64) for L in LAYERS}
    elif method in ("firing_rate_contrast", "f_classif", "chi2"):
        # Exact per-feature score from the streamed moments -> top-150/layer. One instant numpy op
        # per layer over 16,384 features, so no parallelism is needed (chi2 is free here).
        count_np = cls_count.cpu().numpy()
        n_tok = int(count_np.sum())
        print(f"[select:{method}] scoring on ALL {n_tok} labeled train tokens (streamed, exact)", flush=True)
        sel_np, selected = {}, {}
        for L in LAYERS:
            if method == "firing_rate_contrast":
                scores = scores_from_firing_counts(count_np, cls_fire[L].cpu().numpy())
            else:
                scores = scores_from_class_moments(
                    method,
                    count_np,
                    cls_sum[L].cpu().numpy(),
                    cls_sumsq[L].cpu().numpy(),
                )
            idx = np.sort(np.argsort(scores)[::-1][:TOPK]).astype(np.int64)
            sel_np[f"layer_{L}"] = idx
            selected[L] = torch.as_tensor(idx, device=DEVICE)
    elif method == "mutual_info":
        y_all = np.concatenate(buf_y)
        xmats = {L: np.concatenate(buf_x[L]).astype(np.float32) for L in LAYERS}
        for L in LAYERS:
            buf_x[L].clear()
        njobs = min(len(LAYERS), a.select_jobs or (os.cpu_count() or 1))
        print(
            f"[select:{method}] fitting {len(LAYERS)} layers on {len(y_all)} tokens x {D_SAE} "
            f"feats across {njobs} procs",
            flush=True,
        )
        t_fit = time.time()
        sel_np, selected = {}, {}
        results = Parallel(n_jobs=njobs, backend="loky", return_as="generator_unordered")(
            delayed(_fit_layer)(method, xmats[L], y_all, TOPK, SEED, L) for L in LAYERS
        )
        for done_ct, (L, idx, secs) in enumerate(results, 1):
            sel_np[f"layer_{L}"] = idx
            selected[L] = torch.as_tensor(idx, device=DEVICE)
            print(
                f"  [select:{method}] layer {L} done ({done_ct}/{len(LAYERS)}) "
                f"fit={secs:.0f}s  elapsed={time.time() - t_fit:.0f}s",
                flush=True,
            )
        del xmats
    else:  # learned probe
        probe_config = ProbeConfig(
            classes=n_classes,
            topk=TOPK,
            epochs=a.probe_epochs,
            batch_size=a.probe_batch_size,
            learning_rate=a.probe_lr,
            weight_decay=a.probe_weight_decay,
            group_lambda=a.probe_group_lambda,
            patience=a.probe_patience,
            seeds=tuple(int(value) for value in a.probe_seeds.split(",") if value),
            device=DEVICE,
        )
        if method == "probe_group":
            # Sparse survivor selection: strong group-lasso, keep the columns that live.
            selected_np, probe_diagnostics = fit_group_sparse_layers(
                probe_x,
                labels[probe_rows].astype(np.int64),
                probe_plan["is_validation"],
                probe_config,
                out,
                target=a.probe_sparse_target,
                search_rounds=a.probe_search_rounds,
                search_epochs=a.probe_search_epochs,
                lam_lo=a.probe_lam_lo,
                lam_hi=a.probe_lam_hi,
            )
        else:  # probe_l2: dense probe ranked by class-centered L2 norm
            selected_np, probe_diagnostics = fit_layer_probes(
                probe_x,
                labels[probe_rows].astype(np.int64),
                probe_plan["is_validation"],
                probe_config,
                method,
                out,
            )
        sel_np = {f"layer_{L}": selected_np[L] for L in LAYERS}
        selected = {L: torch.as_tensor(selected_np[L], device=DEVICE) for L in LAYERS}
        del probe_x, probe_lut
    print(f"[select] done ({time.time() - t1:.0f}s)", flush=True)

    if a.selection_only:
        np.savez(out / "selected_features.npz", **sel_np)
        for h in handles:
            h.remove()
        del model, backbone, saep
        gc.collect()
        torch.cuda.empty_cache()
        print(f"wrote selection diagnostics to {out}", flush=True)
        return

    # Per-layer widths: selectors may keep fewer than TOPK (probe_group under a strong penalty),
    # so the layout is cumulative rather than uniform TOPK stripes.
    widths = [int(len(selected_np[L])) for L in LAYERS]
    layer_off = np.concatenate(([0], np.cumsum(widths))).astype(int)
    WIDTH = int(layer_off[-1])
    layer_cols = {int(L): (int(layer_off[li]), int(layer_off[li + 1])) for li, L in enumerate(LAYERS)}
    if any(w != TOPK for w in widths):
        print(f"[select] per-layer widths={widths} TOTAL_WIDTH={WIDTH} (TOPK={TOPK})", flush=True)

    # precompute selected SAE param slices
    saesel = {
        L: (
            saep[L][0][:, selected[L]],
            saep[L][1][selected[L]],
            saep[L][2],
            saep[L][3][selected[L]],
        )
        for L in LAYERS
    }

    # ================= PASS 2: encode all records -> features.npy =================
    feats = np.lib.format.open_memmap(
        out / "features.npy", mode="w+", dtype=DTYPE, shape=(N, WIDTH)
    )
    all_batches = make_batches(list(range(len(recs))), lengths, a.token_budget, a.max_seqs)
    print(f"[encode] {len(all_batches)} batches over {len(recs)} records", flush=True)
    t2 = time.time()
    done = 0
    clamp_ct = 0
    fmax = 0.0

    def enc_fn(item):
        nonlocal clamp_ct, fmax
        run_backbone(backbone, captured, item)
        mask = item["attention"].to(DEVICE, non_blocking=True).bool()
        rows = item["rows_global"]
        packed = np.empty((rows.shape[0], WIDTH), DTYPE)
        for li, L in enumerate(LAYERS):
            h = captured[L][mask].float()  # (n_real, 2560) row-major (rec,token)
            W, be, bd, thr = saesel[L]
            col = jumprelu(h, W, be, bd, thr)  # (n_real, widths[li]) fp32
            fmax = max(fmax, float(col.max()) if col.numel() else 0.0)
            if np.float16 == DTYPE:
                col = col.clamp(max=FP16_MAX)
            packed[:, layer_off[li] : layer_off[li + 1]] = col.cpu().numpy().astype(DTYPE, copy=False)
        order = np.argsort(rows, kind="stable")  # sequential writes on networked FS
        feats[rows[order]] = packed[order]

    for bi, item in enumerate(Prefetcher(all_batches, ids_all, offsets, pad_id)):
        forward_with_oom_guard(enc_fn, item, ids_all, offsets, pad_id)
        done += len(item["batch"])
        if (bi + 1) % 10 == 0 or bi + 1 == len(all_batches):
            mem = torch.cuda.max_memory_allocated() / 2**30
            rate = N * (done / len(recs)) / max(time.time() - t2, 1e-9)
            print(
                f"  [encode] {done}/{len(recs)} recs  {(time.time() - t2) / (bi + 1):.2f}s/batch  "
                f"~{rate / 1e3:.0f}k tok/s  peakVRAM={mem:.1f}GiB",
                flush=True,
            )
    feats.flush()
    print(
        f"[encode] done ({time.time() - t2:.0f}s)  max_feat={fmax:.1f}  fp16_clamped={clamp_ct}",
        flush=True,
    )

    for h in handles:
        h.remove()
    del model, backbone, saep, saesel
    gc.collect()
    torch.cuda.empty_cache()

    # ---- save numpy sidecars ----
    np.save(out / "labels.npy", labels.astype(np.int16))
    np.save(out / "lengths.npy", lengths.astype(np.int32))
    np.save(out / "role_ids.npy", roles.astype(np.int8))
    np.savez(out / "split_indices.npz", train=train_rec, validation=val_rec, test=test_rec)
    np.savez(out / "selected_features.npz", **sel_np)
    meta = {
        "dataset": str(Path(a.dataset).resolve()),
        "base_model": MODEL_ID,
        "sae_release": SAE_RELEASE,
        "layers": LAYERS,
        "hidden_per_layer": HIDDEN,
        "selected_features_per_layer": TOPK,
        "features_per_layer": {int(L): int(len(selected_np[L])) for L in LAYERS},
        "layer_cols": layer_cols,
        "uniform_layer_width": all(len(selected_np[L]) == TOPK for L in LAYERS),
        "total_features": WIDTH,
        "dictionary_width": D_SAE,
        "sparsity_profile": "small",
        "encode_convention": "manual JumpReLU (act-b_dec)@W_enc+b_enc, gate>threshold (matches Gemma-1B sae1500)",
        "feature_selection": (
            "summed JumpReLU over ALL tokens of TRAIN records"
            if a.select == "summation"
            else (
                f"max-minus-mean per-topic nonzero firing rate over labeled TRAIN tokens (top-{TOPK}/layer)"
                if a.select == "firing_rate_contrast"
                else (
                f"per-layer multinomial logistic probe over labeled TRAIN tokens (top-{TOPK}/layer)"
                if a.select.startswith("probe_")
                else f"SelectKBest({a.select}) over labeled TRAIN tokens (top-{TOPK}/layer)"
                )
            )
        ),
        "selection_method": a.select,
        # All closed-form selectors stream every train token; only mutual_info uses a subsample cap.
        "selection_token_cap": (
            a.select_token_cap
            if a.select == "mutual_info"
            else (a.probe_token_cap if a.select.startswith("probe_") else None)
        ),
        "supervise_prompt": bool(a.supervise_prompt),
        "probe_balance": (a.probe_balance if a.select.startswith("probe_") else None),
        "probe_seeds": (
            [int(value) for value in a.probe_seeds.split(",") if value]
            if a.select.startswith("probe_")
            else None
        ),
        "invalid_tag_records": len(invalid_tags),
        "tag_validation": a.tag_validation,
        "execution": "teacher_forced_prefill_only",
        "attn_implementation": a.attn,
        "feature_dtype": str(np.dtype(DTYPE)),
        "fp16_clamped_values": 0,
        "max_feature_value": fmax,
        "split_ratio": "70/15/15",
        "split_rng": "RandomState(42) canonical (matches 1B 10k)",
        "records": len(recs),
        "tokens": N,
        "seed": SEED,
        "train": len(train_rec),
        "validation": len(val_rec),
        "test": len(test_rec),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    # NOTE: gpu_cache.pt (torch GPU-resident pack) intentionally NOT written -- the baseline
    # run path (run_4b_baselines_sae150.py -> training.run) reads features.npy via mmap, so the
    # extra fp32 duplicate (~53 GiB) is pure overhead. selected_features.npz above carries the
    # per-layer SAE indices the decode arm needs.
    print(f"\nwrote {out}", flush=True)
    print(
        f"  features.npy {(N, WIDTH)} {np.dtype(DTYPE).name} = "
        f"{N * WIDTH * np.dtype(DTYPE).itemsize / 2**30:.1f} GiB",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=str(ROOT / "runs/GPT_5.6_10k_2108/dataset.final.jsonl"))
    p.add_argument("--output", default=str(ROOT / "cache/sae1500_10k_gemma4b_prompt_response"))
    p.add_argument("--model", default=MODEL_ID, help="HF model id (google/gemma-3-1b-it or -4b-it)")
    p.add_argument("--sae-release", default=SAE_RELEASE, help="sae_lens release for this model")
    p.add_argument(
        "--layers", default="24-33", help="inclusive layer range 'lo-hi' (1B: 16-25, 4B: 24-33)"
    )
    p.add_argument("--topk", type=int, default=150, help="features/layer (50 or 150)")
    p.add_argument(
        "--dtype",
        default="fp32",
        choices=["fp16", "fp32"],
        help="fp16 (1B, small acts) or fp32 (4B, large acts)",
    )
    p.add_argument("--token-budget", type=int, default=110000)
    p.add_argument("--max-seqs", type=int, default=320)
    p.add_argument("--attn", default="flash_attention_2")
    p.add_argument("--limit", type=int, default=0, help="cap #records (0 = all 10k)")
    p.add_argument(
        "--select",
        default="summation",
        choices=SELECT_METHODS,
        help="feature selection: summation, firing-rate contrast, or another supervised scorer",
    )
    p.add_argument(
        "--select-token-cap",
        type=int,
        default=50000,
        help="max labeled TRAIN tokens (Bernoulli-subsampled) used to fit the supervised scorer; "
        "0 = use all. Bounds RAM and keeps mutual_info tractable. Ignored for summation.",
    )
    p.add_argument(
        "--supervise-prompt",
        action="store_true",
        help="prompt+response supervision: also label prompt <topic> spans (default response-only)",
    )
    p.add_argument(
        "--select-jobs",
        type=int,
        default=0,
        help="parallel processes for the per-layer supervised fit (0 = min(#layers, #cpus))",
    )
    p.add_argument(
        "--probe-token-cap",
        type=int,
        default=100000,
        help="labeled TRAIN tokens buffered for learned probes (0 = all; usually impractical)",
    )
    p.add_argument("--probe-validation-fraction", type=float, default=0.10)
    p.add_argument(
        "--probe-balance",
        choices=["natural", "topic_role"],
        default="natural",
        help="natural matches F-statistic weighting; topic_role is the secondary balanced ablation",
    )
    p.add_argument("--probe-epochs", type=int, default=20)
    p.add_argument("--probe-batch-size", type=int, default=1024)
    p.add_argument("--probe-lr", type=float, default=0.01)
    p.add_argument("--probe-weight-decay", type=float, default=1e-4)
    p.add_argument("--probe-group-lambda", type=float, default=1e-4)
    p.add_argument("--probe-patience", type=int, default=4)
    # probe_group survivor selection: binary-search a per-layer group-lasso penalty so ~target
    # feature columns survive, then keep those columns (final width is still --topk).
    p.add_argument(
        "--probe-sparse-target",
        type=int,
        default=0,
        help="target surviving feature columns/layer for probe_group (0 = use --topk)",
    )
    p.add_argument("--probe-search-rounds", type=int, default=8)
    p.add_argument("--probe-search-epochs", type=int, default=10)
    p.add_argument("--probe-lam-lo", type=float, default=1e-3)
    p.add_argument("--probe-lam-hi", type=float, default=1e3)
    p.add_argument(
        "--selection-only",
        action="store_true",
        help="fit and save feature selection diagnostics without building features.npy",
    )
    p.add_argument(
        "--probe-seeds",
        default="42,43,44",
        help="comma-separated probe seeds; rankings are aggregated across seeds",
    )
    p.add_argument(
        "--tag-validation",
        choices=["error", "warn", "off"],
        default="error",
        help="handling for malformed or unrecognized topic tags",
    )
    main(p.parse_args())
