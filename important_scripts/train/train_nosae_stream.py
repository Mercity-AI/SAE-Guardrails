#!/usr/bin/env python3
"""Streaming no-SAE ablation: SAE-150 vs raw-hidden, co-training GRU/Transformer/ConvNeXt.

Derived from build_and_train_nosae.py, but *streaming* instead of RAM-resident. The raw arm at
10k is 204 GB (1B) / 453 GB (4B); rather than materialise it, we recompute hidden states one
batch at a time, every epoch, so peak memory is a single batch. ONE teacher-forced flash-attn
prefill per batch feeds BOTH arms and ALL THREE architectures:

  * raw    = concat of the 10 layers' residual hidden states (float32 into the classifier)
  * SAE-150 = manual JumpReLU on the 150 selected indices/layer (from the sae1500 cache's
              selected_features.npz), matching the cache's encode convention exactly

Two arms x three archs = six heads, each with its own AdamW; every head trains off the same
per-batch prefill. Response-only labels and the canonical 10k 70/15/15 split come straight from
the sae1500 cache, so every number is directly comparable to the static SAE-150 baselines.

  python train_nosae_stream.py --model google/gemma-3-1b-it --layers 16-25 \
      --sae-release gemma-scope-2-1b-it-res-all \
      --sel150 cache/sae1500_10k_1b_prompt_response \
      --dataset runs/GPT_5.6_10k_2108/dataset.final.jsonl \
      --split  cache/sae1500_10k_1b_prompt_response \
      --outdir ablation/nosae_stream_1b_10k --pack-seqs 64
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sae_lens import SAE
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPTS_DIR = Path(__file__).resolve().parent

# local imports
from important_scripts.paths import PROJECT_ROOT
from important_scripts.cache.build_sae1500_gemma4b_10k import find_backbone, make_batches
from important_scripts.model.decoder_utils import decode_logits
from important_scripts.cache.label_records import label_record
from important_scripts.model.models import (
    PAD,
    TOPICS,
    build_model,
    collate,
    flatten_labeled,
    metrics,
    seed_everything,
)

ROOT = PROJECT_ROOT
from sklearn.metrics import f1_score

DEVICE = "cuda"
SEED = 42
NCLASS = len(TOPICS)
RAW_DECODER = {"method": "raw"}

# Same per-architecture hyperparameters as the established baselines (train_from_pt.ARCH).
ARCH = {
    "gru": {
        "architecture": "gru",
        "learning_rate": 3e-4,
        "batch_size": 16,
        "model": {"hidden_width": 256, "depth": 3, "dropout": 0.1},
    },
    "transformer": {
        "architecture": "transformer",
        "learning_rate": 3e-4,
        "batch_size": 16,
        "model": {
            "hidden_width": 256,
            "heads": 8,
            "depth": 2,
            "feedforward_width": 512,
            "dropout": 0.1,
        },
    },
    "convnext": {
        "architecture": "convnext",
        "learning_rate": 1e-3,
        "batch_size": 16,
        "model": {
            "channels": 256,
            "depth": 6,
            "kernel": 7,
            "expansion": 4,
            "dropout": 0.1,
            "layer_scale": 0.01,
        },
    },
}


# ------------------------------------------------------------------ prefill + featurise
def epoch_groups(indices, lengths, token_budget, max_seqs, rng, bucket=1024):
    """Shuffle records, then length-sort within buckets so packing wastes little padding
    while epoch-to-epoch order stays stochastic. Returns a shuffled list of index groups."""
    idx = list(indices)
    rng.shuffle(idx)
    groups = []
    for c in range(0, len(idx), bucket):
        chunk = sorted(idx[c : c + bucket], key=lambda i: lengths[i])
        groups.extend(make_batches(chunk, lengths, token_budget, max_seqs))
    rng.shuffle(groups)
    return groups


@torch.no_grad()  # NOT inference_mode: these tensors feed the classifiers' backward()
def featurise(backbone, captured, saep, sel, layers, ids_all, labels_all, group, pad_id, arms):
    """One prefill over a group of records -> {arm: [(features, labels), ...]} GPU tensors.

    Only the requested arms are built (raw = concat of the 10 layers; sae150 = manual JumpReLU
    on the cached selected indices), so a raw-only run does no SAE work."""
    seqs = [ids_all[i] for i in group]
    width = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(seqs), width), dtype=torch.long)
    for r, s in enumerate(seqs):
        input_ids[r, : len(s)] = torch.tensor(s, dtype=torch.long)
        attention[r, : len(s)] = 1
    captured.clear()
    backbone(
        input_ids=input_ids.to(DEVICE, non_blocking=True),
        attention_mask=attention.to(DEVICE, non_blocking=True),
        use_cache=False,
        return_dict=True,
    )
    if set(captured) != set(layers):
        raise RuntimeError(f"missing hooked layers: {set(layers) - set(captured)}")
    out = {arm: [] for arm in arms}
    for r, i in enumerate(group):
        L = len(seqs[r])
        hs = [captured[ly][r, :L, :].float() for ly in layers]  # 10 x (L, HIDDEN)
        lab = torch.as_tensor(labels_all[i], device=DEVICE, dtype=torch.long)
        if "raw" in arms:
            out["raw"].append((torch.cat(hs, dim=-1), lab))  # (L, HIDDEN*10)
        if "sae150" in arms:
            sae_parts = []
            for ly, h in zip(layers, hs, strict=False):
                W, be, bd, thr = saep[ly]
                s = sel[ly]
                pre = (h - bd) @ W[:, s] + be[s]  # manual JumpReLU
                sae_parts.append(pre * (pre > thr[s]))
            out["sae150"].append((torch.cat(sae_parts, dim=-1), lab))  # (L, 1500)
    captured.clear()
    return out


# ------------------------------------------------------------------ train / eval
def train_group(feats, heads, loss_fn, clip, micro, rng):
    """One optimiser step per model over every micro-batch (16) inside the prefill group."""
    n = len(next(iter(feats.values())))
    order = list(range(n))
    rng.shuffle(order)
    for s in range(0, len(order), micro):
        chunk = order[s : s + micro]
        batches = {
            arm: collate([rows[j] for j in chunk], input_features=rows[chunk[0]][0].shape[-1])
            for arm, rows in feats.items()
        }
        for (arm, arch), (model, opt) in heads.items():
            model.train()
            x, y = batches[arm]
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, NCLASS), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()


@torch.inference_mode()
def stream_eval(
    backbone,
    captured,
    saep,
    sel,
    layers,
    ids_all,
    labels_all,
    indices,
    lengths,
    heads,
    token_budget,
    max_seqs,
    pad_id,
    arms,
):
    """Stream the given records once; return per-model [{labels, predictions}] rows."""
    rows = {k: [] for k in heads}
    for group in make_batches(list(indices), lengths, token_budget, max_seqs):
        feats = featurise(
            backbone, captured, saep, sel, layers, ids_all, labels_all, group, pad_id, arms
        )
        for (arm, arch), (model, _) in heads.items():
            model.eval()
            data = feats[arm]
            x, _ = collate(data, input_features=data[0][0].shape[-1])
            logits = model(x).float().cpu().numpy()
            for r, (feat, lab) in enumerate(data):
                L = feat.shape[0]
                rows[(arm, arch)].append(
                    {
                        "labels": lab.cpu().numpy(),
                        "predictions": decode_logits(logits[r, :L], RAW_DECODER),
                    }
                )
        del feats
    return rows


def save_head(
    outdir,
    arm,
    arch,
    spec,
    cfg,
    best_state,
    best_epoch,
    final_state,
    epochs,
    val_metrics,
    test_metrics,
    fval_metrics,
    ftest_metrics,
    nparams,
):
    """Save one trained classifier head together with its metrics."""
    run_dir = outdir / arm / arch
    run_dir.mkdir(parents=True, exist_ok=True)
    model_config = {"input_features": cfg["input_features"], "classes": NCLASS, **spec["model"]}
    best_ckpt = {
        "state_dict": best_state,
        "architecture": arch,
        "model": model_config,
        "classes": TOPICS,
        "config": cfg,
        "epoch": best_epoch,
        "selection": "best_validation_macro_f1",
    }
    torch.save(best_ckpt, run_dir / "checkpoint_best.pt")
    torch.save(
        {**best_ckpt, "state_dict": final_state, "epoch": epochs, "selection": "final_epoch"},
        run_dir / "checkpoint_final.pt",
    )
    (run_dir / "checkpoint_best.metrics.json").write_text(
        json.dumps(
            {
                "checkpoint": "checkpoint_best.pt",
                "epoch": best_epoch,
                "validation": val_metrics,
                "test": test_metrics,
            },
            indent=2,
        )
        + "\n"
    )
    (run_dir / "checkpoint_final.metrics.json").write_text(
        json.dumps(
            {
                "checkpoint": "checkpoint_final.pt",
                "epoch": epochs,
                "validation": fval_metrics,
                "test": ftest_metrics,
            },
            indent=2,
        )
        + "\n"
    )
    result = {
        "arm": arm,
        "architecture": arch,
        "input_features": cfg["input_features"],
        "learning_rate": spec["learning_rate"],
        "batch_size": spec["batch_size"],
        "seed": SEED,
        "parameters": nparams,
        "best_epoch": best_epoch,
        "validation": val_metrics,
        "test": test_metrics,
        "final_epoch": epochs,
        "final_validation": fval_metrics,
        "final_test": ftest_metrics,
    }
    (run_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    ov = test_metrics["topic_overlap"]
    print(
        f"  [{arm}/{arch}] test F1={test_metrics['macro_f1']:.4f} "
        f"IoU={ov['macro_iou']:.4f} (best epoch {best_epoch})",
        flush=True,
    )
    return result


def main(a):
    """Streaming no-SAE trainer: co-train GRU/Transformer/ConvNeXt on SAE-150 vs raw-hidden arms."""
    lo, hi = a.layers.split("-")
    layers = list(range(int(lo), int(hi) + 1))
    arms = [x for x in a.arms.split(",") if x]
    if not set(arms) <= {"raw", "sae150"}:
        raise SystemExit(f"--arms must be a subset of raw,sae150 (got {arms})")
    outdir = ROOT / a.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    micro = a.micro_batch
    token_budget = a.pack_seqs * a.token_per_seq
    seed_everything(SEED, deterministic=False)  # wide raw GRU + cuDNN determinism can hang
    rng = np.random.RandomState(SEED)

    # ---- records: tokenise + response-only labels via the validated labeler ----
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "right"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ds_path = ROOT / a.dataset
    recs = [json.loads(l)["record"] for l in ds_path.open()]
    ids_all, labels_all, lengths = [], [], []
    for r in recs:
        ii, rr, ll = label_record(r, tok)
        ids_all.append(ii)
        labels_all.append(np.asarray(ll, np.int64))
        lengths.append(len(ii))
    lengths = np.asarray(lengths, np.int64)

    # ---- canonical split + length check against the sae1500 cache ----
    split_dir = ROOT / a.split
    z = np.load(split_dir / "split_indices.npz")
    split = {k: z[k] for k in ("train", "validation", "test")}
    cache_lengths = np.load(split_dir / "lengths.npy")
    if not np.array_equal(cache_lengths, lengths):
        n_bad = int((cache_lengths != lengths).sum())
        raise SystemExit(
            f"tokenisation drift: {n_bad} records differ in length from the cache "
            f"({split_dir / 'lengths.npy'}); split/labels would misalign."
        )
    print(
        f"records={len(recs)} tokens={int(lengths.sum()):,} "
        f"split={len(split['train'])}/{len(split['validation'])}/{len(split['test'])} "
        f"(lengths match cache)",
        flush=True,
    )

    # ---- supervision: default is RESPONSE-ONLY (label_record). For the prompt+response redo,
    #      override with the CANONICAL PR labels.npy from the cache (same labeler that built every
    #      other PR experiment), aligned by the just-verified per-record lengths. ----
    sup_default = int(sum((l != PAD).sum() for l in labels_all))
    if a.pr_labels:
        pr = np.load(ROOT / a.pr_labels / "labels.npy").astype(np.int64)
        if len(pr) != int(lengths.sum()):
            raise SystemExit(f"--pr-labels length {len(pr)} != total tokens {int(lengths.sum())}")
        offs = np.concatenate([[0], np.cumsum(lengths)])
        labels_all = [pr[offs[k] : offs[k + 1]] for k in range(len(recs))]
        sup_pr = int((pr != PAD).sum())
        print(
            f"SUPERVISION = prompt+response (canonical PR labels from {a.pr_labels}); "
            f"supervised tokens {sup_pr:,} vs response-only {sup_default:,} "
            f"(+{sup_pr - sup_default:,} prompt-topic tokens)",
            flush=True,
        )
    else:
        print(
            f"SUPERVISION = response-only (in-code labeler); supervised tokens {sup_default:,}",
            flush=True,
        )

    # ---- Gemma backbone (bf16, flash-attn) + layer hooks ----
    model = (
        AutoModelForCausalLM.from_pretrained(
            a.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
        .to(DEVICE)
        .eval()
    )
    backbone = find_backbone(model)
    HIDDEN = int(getattr(model.config, "text_config", model.config).hidden_size)
    captured: dict[int, torch.Tensor] = {}
    for L in layers:
        backbone.layers[L].register_forward_hook(
            lambda _m, _i, o, L=L: captured.__setitem__(L, o[0] if isinstance(o, tuple) else o)
        )

    # ---- SAE params + selected indices: only when the sae150 arm is requested ----
    sel, saep = {}, {}
    if "sae150" in arms:
        sel_npz = np.load(ROOT / a.sel150 / "selected_features.npz")
        sel = {L: torch.as_tensor(sel_npz[f"layer_{L}"], device=DEVICE) for L in layers}
        for L in layers:
            s = SAE.from_pretrained(
                release=a.sae_release, sae_id=f"layer_{L}_width_16k_l0_small", device=DEVICE
            )
            s = s[0] if isinstance(s, tuple) else s
            s.eval()
            saep[L] = (s.W_enc.float(), s.b_enc.float(), s.b_dec.float(), s.threshold.float())

    F = {"raw": HIDDEN * len(layers), "sae150": 150 * len(layers)}
    print(
        f"HIDDEN={HIDDEN} layers={layers}  arms={arms} "
        + " ".join(f"{arm} F={F[arm]}" for arm in arms),
        flush=True,
    )

    # ---- models: len(arms) x 3 archs, each its own model + optimiser ----
    heads, cfgs = {}, {}
    for arm in arms:
        for arch, spec in ARCH.items():
            mc = {"input_features": F[arm], "classes": NCLASS, **spec["model"]}
            m = build_model(spec["architecture"], mc).to(DEVICE)
            opt = torch.optim.AdamW(m.parameters(), lr=spec["learning_rate"], weight_decay=0.01)
            heads[(arm, arch)] = (m, opt)
            cfgs[(arm, arch)] = {"input_features": F[arm]}
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)

    best = dict.fromkeys(heads, (-1.0, 0, None))  # (score, epoch, state)

    # ---- training: one prefill per batch, every model steps off it ----
    for epoch in range(1, a.epochs + 1):
        t0 = time.time()
        groups = epoch_groups(split["train"], lengths, token_budget, a.pack_seqs, rng)
        for gi, group in enumerate(groups):
            feats = featurise(
                backbone, captured, saep, sel, layers, ids_all, labels_all, group, pad_id, arms
            )
            train_group(feats, heads, loss_fn, a.clip, micro, rng)
            del feats
            if (gi + 1) % 20 == 0:
                print(
                    f"  epoch {epoch} {gi + 1}/{len(groups)} groups "
                    f"{(time.time() - t0) / (gi + 1):.2f}s/grp "
                    f"VRAM={torch.cuda.max_memory_allocated() / 2**30:.1f}GiB",
                    flush=True,
                )
        val_rows = stream_eval(
            backbone,
            captured,
            saep,
            sel,
            layers,
            ids_all,
            labels_all,
            split["validation"],
            lengths,
            heads,
            token_budget,
            a.pack_seqs,
            pad_id,
            arms,
        )
        line = []
        for k, rows in val_rows.items():
            truth, pred = flatten_labeled(rows)
            score = f1_score(truth, pred, average="macro", zero_division=0)
            line.append(f"{k[0][:3]}/{k[1][:4]}={score:.3f}")
            if score > best[k][0]:
                best[k] = (
                    score,
                    epoch,
                    {n: v.detach().cpu().clone() for n, v in heads[k][0].state_dict().items()},
                )
                # crash-safe: persist the best-so-far checkpoint each time it improves
                arm, arch = k
                rd = outdir / arm / arch
                rd.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "state_dict": best[k][2],
                        "architecture": arch,
                        "model": {
                            "input_features": cfgs[k]["input_features"],
                            "classes": NCLASS,
                            **ARCH[arch]["model"],
                        },
                        "classes": TOPICS,
                        "epoch": epoch,
                        "selection": "best_validation_macro_f1",
                    },
                    rd / "checkpoint_best.pt",
                )
        print(
            f"epoch {epoch:2d}/{a.epochs} [{(time.time() - t0) / 60:.1f}m] valF1 " + " ".join(line),
            flush=True,
        )

    # ---- final metrics: best-checkpoint and final-epoch, on val + test ----
    tolerances = [5, 10]
    final_state = {
        k: {n: v.detach().cpu().clone() for n, v in heads[k][0].state_dict().items()} for k in heads
    }
    fval = stream_eval(
        backbone,
        captured,
        saep,
        sel,
        layers,
        ids_all,
        labels_all,
        split["validation"],
        lengths,
        heads,
        token_budget,
        a.pack_seqs,
        pad_id,
        arms,
    )
    ftest = stream_eval(
        backbone,
        captured,
        saep,
        sel,
        layers,
        ids_all,
        labels_all,
        split["test"],
        lengths,
        heads,
        token_budget,
        a.pack_seqs,
        pad_id,
        arms,
    )
    fval_m = {k: metrics(v, tolerances) for k, v in fval.items()}
    ftest_m = {k: metrics(v, tolerances) for k, v in ftest.items()}
    for k in heads:
        heads[k][0].load_state_dict(best[k][2])
    bval = stream_eval(
        backbone,
        captured,
        saep,
        sel,
        layers,
        ids_all,
        labels_all,
        split["validation"],
        lengths,
        heads,
        token_budget,
        a.pack_seqs,
        pad_id,
        arms,
    )
    btest = stream_eval(
        backbone,
        captured,
        saep,
        sel,
        layers,
        ids_all,
        labels_all,
        split["test"],
        lengths,
        heads,
        token_budget,
        a.pack_seqs,
        pad_id,
        arms,
    )

    summary = {
        "model": a.model,
        "layers": layers,
        "records": len(recs),
        "tokens": int(lengths.sum()),
        "epochs": a.epochs,
        "arms": {arm: F[arm] for arm in arms},
        "runs": {},
    }
    for arm, arch in heads:
        k = (arm, arch)
        spec = ARCH[arch]
        res = save_head(
            outdir,
            arm,
            arch,
            spec,
            cfgs[k],
            best[k][2],
            best[k][1],
            final_state[k],
            a.epochs,
            metrics(bval[k], tolerances),
            metrics(btest[k], tolerances),
            fval_m[k],
            ftest_m[k],
            sum(p.numel() for p in heads[k][0].parameters()),
        )
        summary["runs"][f"{arm}/{arch}"] = res
    (outdir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nWrote {outdir / 'metrics.json'}\nSTREAMING NO-SAE ABLATION DONE", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--layers", required=True)
    p.add_argument("--sae-release", default=None, help="required only for the sae150 arm")
    p.add_argument("--sel150", default=None, help="required only for the sae150 arm")
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--arms", default="raw", help="comma-separated subset of raw,sae150")
    p.add_argument(
        "--pr-labels",
        default=None,
        help="cache dir whose labels.npy holds canonical PROMPT+RESPONSE labels; "
        "when set, overrides the in-code response-only labeler (aligned by lengths)",
    )
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--pack-seqs", type=int, default=64)
    p.add_argument("--token-per-seq", type=int, default=1200)
    p.add_argument("--micro-batch", type=int, default=16)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--attn", default="flash_attention_2")
    main(p.parse_args())
