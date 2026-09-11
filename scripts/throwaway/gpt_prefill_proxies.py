#!/usr/bin/env python3
"""Compute the decode proxy measures on the TEACHER (GPT) response, from a cached prefill arm.

The decode proxies read the base model's own generation; these read the teacher response that was
teacher-forced into the same detectors, so the two are directly comparable. Also reports generator
coverage: whether the teacher's response itself (gold labels) covers each prompt topic, which
separates 'the detector missed it' from 'the writer never wrote it'. Static (dense) caches only.

  python gpt_prefill_proxies.py --cache cache/sae1500_10k_1b_pr \
      --ckpt results/gemma1b_baselines_sae150_10k_pr --out results/gpt_proxies_static_1b.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/workspace/scope")
sys.path.insert(0, str(ROOT / "modeling"))
from models import PAD, TOPICS, build_model  # noqa: E402

TI = {t: i for i, t in enumerate(TOPICS)}
SUBS = {"GRU": "gru_lr3e4", "Transformer": "transformer_wide", "ConvNeXt": "convnext_window505"}
DATASET = ROOT / "runs/GPT_5.6_10k_2108/dataset.final.jsonl"


def main(a):
    """Compute the decode proxy metrics on the teacher (GPT) response from a cached prefill arm."""
    cache = ROOT / a.cache
    feats = np.load(cache / "features.npy", mmap_mode="r")
    labels = np.load(cache / "labels.npy").astype(np.int64)
    roles = np.load(cache / "role_ids.npy").astype(np.int8)
    lengths = np.load(cache / "lengths.npy")
    offs = np.concatenate(([0], np.cumsum(lengths)))
    test = np.load(cache / "split_indices.npz")["test"]
    ds = [json.loads(l)["record"] for l in DATASET.open()]

    dets = {}
    for name, sub in SUBS.items():
        p = torch.load(
            ROOT / a.ckpt / sub / "checkpoint_best.pt", map_location="cpu", weights_only=False
        )
        m = build_model(p["architecture"], p["model"])
        m.load_state_dict(p["state_dict"])
        dets[name] = m.to("cuda").eval()

    res = {}
    for name, m in dets.items():
        pur = []
        cov = []
        full = []
        sacc = []
        trec = []
        tprec = []
        gencov = []
        with torch.inference_mode():
            for r in test:
                r = int(r)
                lo, hi = int(offs[r]), int(offs[r + 1])
                x = torch.from_numpy(np.asarray(feats[lo:hi]).astype(np.float32))[None].to("cuda")
                pred = m(x)[0].argmax(-1).cpu().numpy()
                role = roles[lo:hi]
                lab = labels[lo:hi]
                resp = role == 2
                p_resp = pred[resp]
                g_resp = lab[resp]
                topics = ds[r]["topics"]
                truth = {TI[t] for t in topics if t in TI}
                if not truth or len(p_resp) == 0:
                    continue
                pur.append(float(np.isin(p_resp, list(truth)).mean()))
                reach = [t for t in truth if (p_resp == t).mean() >= 0.10]
                cov.append(len(reach) / len(truth))
                full.append(1.0 if len(reach) == len(truth) else 0.0)
                # generator coverage from gold response labels (ignoring PAD)
                gr = g_resp[g_resp != PAD]
                greach = [t for t in truth if len(gr) and (gr == t).mean() >= 0.10]
                gencov.append(len(greach) / len(truth))
                if len(topics) == 1:
                    sacc.append(float((p_resp == TI[topics[0]]).mean()))
                else:
                    pset = {int(v) for v in p_resp}
                    trec.append(len(pset & truth) / len(truth))
                    tprec.append(len(pset & truth) / max(1, len(pset)))
        res[name] = {
            "purity": float(np.mean(pur)),
            "coverage": float(np.mean(cov)),
            "full_coverage": float(np.mean(full)),
            "single_acc": float(np.mean(sacc)) if sacc else None,
            "set_recall": float(np.mean(trec)) if trec else None,
            "set_precision": float(np.mean(tprec)) if tprec else None,
            "generator_coverage": float(np.mean(gencov)),
        }
        print(
            f"{name:12s} purity={res[name]['purity']:.3f} cover={res[name]['coverage']:.3f} "
            f"full={res[name]['full_coverage']:.3f} sAcc={res[name]['single_acc']} "
            f"gen_cov={res[name]['generator_coverage']:.3f}",
            flush=True,
        )
    json.dump(res, open(ROOT / a.out, "w"), indent=2)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    main(p.parse_args())
