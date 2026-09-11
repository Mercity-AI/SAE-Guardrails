#!/usr/bin/env python3
"""Deep look at the decode generations themselves (no model runs): Gemma vs GPT.

Joins, per test record:
  * Gemma greedy generation text + SAE-GRU per-token topic preds  (gemma_arm/generations_greedy)
  * dynamic detectors' per-token topic preds on the SAME Gemma tokens  (decode_dynamic_preds.pt)
  * GPT reference response with its topic TAGS (test_manifest gpt_response_raw) + GPT scoring (gpt_arm)

Reports: length distribution, prompt-faithfulness (single-topic), topic switch points
(GPT true switch from tags vs Gemma detector switch), two-topic on-topic coverage, and
Gemma repetition/collapse; then prints concrete sample sets with the text and the run-length
encoded predicted-topic trajectories so the switches/fragmentation are visible.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace/scope/modeling")
from models import TOPICS

ROOT = Path("/workspace/scope")
GEN = ROOT / "ablation/gemma_arm/generations_greedy/per_record.jsonl"
GPT = ROOT / "ablation/gpt_arm/gpt_arm_per_record.jsonl"
MAN = ROOT / "ablation/test_manifest.jsonl"
DYN = ROOT / "results/dyn150_2k/decode_dynamic_preds.pt"
SH = {0: "Enterprise", 1: "GenNews", 2: "CustSvc", 3: "Legal", 4: "Financial", 5: "HR", 6: "Health"}
TAG = re.compile(r"<(/?)(" + "|".join(re.escape(t) for t in TOPICS) + r")>")


def rle(ids, min_run=5):
    """Compact run-length encode: show runs >= min_run tokens; count the brief flips."""
    out = []
    for v in ids:
        if out and out[-1][0] == v:
            out[-1][1] += 1
        else:
            out.append([v, 1])
    big = " ".join(f"{SH[v]}×{n}" for v, n in out if n >= min_run)
    brief = sum(1 for v, n in out if n < min_run)
    switches = sum(1 for a, b in zip(ids, ids[1:], strict=False) if a != b)
    return f"{big}   [{switches} switches, {brief} brief flips]"


def gpt_true_switch_frac(raw):
    """From tagged GPT text, the char fraction at which the topic changes (None if one topic)."""
    segs = []  # (topic, text)
    pos = 0
    cur = None
    buf = []
    for m in TAG.finditer(raw):
        buf.append(raw[pos : m.start()])
        pos = m.end()
        closing, topic = m.group(1), m.group(2)
        if not closing:
            cur = topic
            buf = []
        else:
            segs.append((cur, "".join(buf)))
            buf = []
    lens = [len(t) for _, t in segs if t.strip()]
    topics_in_order = [tp for tp, t in segs if t.strip()]
    if len(lens) < 2:
        return None, topics_in_order
    total = sum(lens)
    return lens[0] / total, topics_in_order


def best_split_frac(preds, t0, t1):
    """Boundary fraction that best splits preds into t0-dominant then t1-dominant (or reverse)."""
    p = np.asarray(preds)
    n = len(p)
    if n < 2:
        return None
    best_b, best_score, best_orient = 0, -1, None
    left0 = np.cumsum(p == t0)
    left1 = np.cumsum(p == t1)
    tot0, tot1 = left0[-1], left1[-1]
    for b in range(1, n):
        s_ab = left0[b - 1] + (tot1 - left1[b - 1])  # t0 left, t1 right
        s_ba = left1[b - 1] + (tot0 - left0[b - 1])  # t1 left, t0 right
        if s_ab >= s_ba and s_ab > best_score:
            best_score, best_b, best_orient = s_ab, b, (t0, t1)
        elif s_ba > best_score:
            best_score, best_b, best_orient = s_ba, b, (t1, t0)
    return best_b / n


def repetition(text):
    """distinct-4gram ratio (low=repetitive) and longest immediate word run."""
    w = text.split()
    if len(w) < 4:
        return 1.0, 1
    grams = list(zip(w, w[1:], w[2:], w[3:], strict=False))
    distinct = len(set(grams)) / len(grams)
    run = maxrun = 1
    for a, b in zip(w, w[1:], strict=False):
        run = run + 1 if a == b else 1
        maxrun = max(maxrun, run)
    return distinct, maxrun


def main():
    """Join and summarize the decode generations (Gemma vs GPT) with their per-token topic predictions."""
    gen = {r["index"]: r for r in (json.loads(l) for l in GEN.open())}
    gpt = {r["index"]: r for r in (json.loads(l) for l in GPT.open())}
    man = {r["index"]: r for r in (json.loads(l) for l in MAN.open())}
    dyn = {r["index"]: r for r in torch.load(DYN, weights_only=False)["records"] if "pred" in r}

    idxs = [i for i in gen if i in gpt and i in man]
    single = [i for i in idxs if gen[i]["single_topic"]]
    two = [i for i in idxs if not gen[i]["single_topic"]]

    # ---------- 1. LENGTH ----------
    g_len = np.array([gen[i]["generated_tokens"] for i in idxs])
    p_len = np.array([gpt[i]["response_tokens"] for i in idxs])
    ratio = g_len / np.maximum(p_len, 1)
    print("=" * 84)
    print("1. RESPONSE LENGTH (tokens):  Gemma decode  vs  GPT reference")
    print(
        f"   Gemma : mean {g_len.mean():6.1f}  median {np.median(g_len):6.1f}  max {g_len.max()}  "
        f"cap-hit {np.mean([gen[i]['hit_cap'] for i in idxs]):.1%}"
    )
    print(f"   GPT   : mean {p_len.mean():6.1f}  median {np.median(p_len):6.1f}  max {p_len.max()}")
    print(
        f"   Gemma/GPT length ratio: mean {ratio.mean():.2f}x  median {np.median(ratio):.2f}x  "
        f"(Gemma longer in {np.mean(ratio > 1):.0%} of records)"
    )

    # ---------- 2. FAITHFULNESS (single-topic) ----------
    print("\n" + "=" * 84)
    print(
        f"2. PROMPT FAITHFULNESS (single-topic, n={len(single)}): frac of response tokens on the prompt topic"
    )
    gpt_f = np.array([gpt[i]["response_assumed_label_acc"] for i in single])
    sae_f = np.array([gen[i]["assumed_label_acc"] for i in single])
    dynf = {a: [] for a in ["GRU", "Transformer", "ConvNeXt"]}
    for i in single:
        for a in dynf:
            pr = dyn[i]["pred"][a].numpy()
            dynf[a].append(float((pr == gen[i]["assumed_label_id"]).mean()))
    print(
        f"   GPT text (SAE-GRU)          : mean {gpt_f.mean():.4f}  median {np.median(gpt_f):.4f}  "
        f"<0.8 in {np.mean(gpt_f < 0.8):.0%}"
    )
    print(
        f"   Gemma text (SAE-GRU)        : mean {sae_f.mean():.4f}  median {np.median(sae_f):.4f}  "
        f"<0.8 in {np.mean(sae_f < 0.8):.0%}"
    )
    for a in dynf:
        v = np.array(dynf[a])
        print(
            f"   Gemma text (dynamic {a:11s}): mean {v.mean():.4f}  median {np.median(v):.4f}  <0.8 in {np.mean(v < 0.8):.0%}"
        )

    # ---------- 3. SWITCH POINTS (two-topic) ----------
    print("\n" + "=" * 84)
    print(
        f"3. TWO-TOPIC SWITCH POINT (n={len(two)}):  GPT true switch (tags)  vs  Gemma detector switch"
    )
    gpt_sw, gem_sw, gpt_ontopic, gem_ontopic, gem_frag = [], [], [], [], []
    for i in two:
        frac, order = gpt_true_switch_frac(man[i]["gpt_response_raw"])
        if frac is not None:
            gpt_sw.append(frac)
        tset = {TOPICS.index(t) for t in gen[i]["topics"]}
        pr = dyn[i]["pred"]["GRU"].numpy()
        if len(tset) == 2:
            t0, t1 = sorted(tset)
            bf = best_split_frac(pr, t0, t1)
            if bf is not None:
                gem_sw.append(bf)
        gem_ontopic.append(float(np.mean([p in tset for p in pr])))
        gem_frag.append(int((pr[1:] != pr[:-1]).sum()))
        gp = np.asarray(gpt[i]["predicted_topic_ids_on_response"])
        gpt_ontopic.append(float(np.mean([p in tset for p in gp])))
    gpt_sw = np.array(gpt_sw)
    gem_sw = np.array(gem_sw)
    print(
        f"   GPT true switch fraction   : mean {gpt_sw.mean():.2f}  median {np.median(gpt_sw):.2f}  "
        f"(0.50 = symmetric split)   [{np.mean((gpt_sw > 0.4) & (gpt_sw < 0.6)):.0%} within 0.4-0.6]"
    )
    print(
        f"   Gemma detector switch frac : mean {gem_sw.mean():.2f}  median {np.median(gem_sw):.2f}"
    )
    print("   On-topic coverage (frac of response tokens predicted within the 2 prompt topics):")
    print(f"      GPT   mean {np.mean(gpt_ontopic):.4f}   Gemma mean {np.mean(gem_ontopic):.4f}")
    print(
        f"   Gemma predicted-topic switches per record (fragmentation): mean {np.mean(gem_frag):.1f}  max {max(gem_frag)}"
    )

    # ---------- 4. REPETITION / COLLAPSE (Gemma) ----------
    print("\n" + "=" * 84)
    print("4. GEMMA REPETITION / COLLAPSE")
    reps = [(i, *repetition(gen[i]["gemma_response"])) for i in idxs]
    d4 = np.array([r[1] for r in reps])
    runs = np.array([r[2] for r in reps])
    print(
        f"   distinct-4gram ratio: mean {d4.mean():.3f}  median {np.median(d4):.3f}  "
        f"(<0.5 = repetitive in {np.mean(d4 < 0.5):.0%} of records)"
    )
    print(f"   longest immediate word run: mean {runs.mean():.1f}  max {runs.max()}")
    print(
        f"   cap-hit (generation ran to {gen[idxs[0]].get('response_content_tokens') and 1536} tokens): "
        f"{np.mean([gen[i]['hit_cap'] for i in idxs]):.1%}"
    )
    worst = sorted(reps, key=lambda r: r[1])[:5]
    print("   most repetitive records (index, distinct-4gram, maxrun, cap, gen_tokens):")
    for i, dd, rr in worst:
        print(
            f"      idx {i}: d4={dd:.3f} run={rr} cap={gen[i]['hit_cap']} tok={gen[i]['generated_tokens']}"
        )

    # ---------- 5. SAMPLE SETS ----------
    def show(i, label, nchar=600):
        g = gen[i]
        tset = [TOPICS.index(t) for t in g["topics"]]
        gacc = gpt[i]["response_assumed_label_acc"]
        gacc = f"{gacc:.2f}" if gacc is not None else "n/a(2-topic)"
        print("\n" + "-" * 84)
        print(
            f"[{label}]  idx {i}  prompt topics: {[SH[t] for t in tset]}  single={g['single_topic']}"
        )
        print(
            f"  GPT   ({gpt[i]['response_tokens']} tok, assumed-acc {gacc}): "
            f"{TAG.sub('', man[i]['gpt_response_raw'])[:nchar].strip()}"
        )
        print(
            f"  GEMMA ({g['generated_tokens']} tok, SAE-acc {g.get('assumed_label_acc', '-')}): "
            f"{g['gemma_response'][:nchar].strip()}"
        )
        for a in ["GRU", "Transformer", "ConvNeXt"]:
            print(f"    dynamic {a:11s} traj: {rle(dyn[i]['pred'][a].tolist())}")
        print(f"    SAE-GRU        traj: {rle(gen[i]['pred_topic_ids'])}")

    print("\n" + "=" * 84 + "\n5. SAMPLE SETS")
    # single-topic faithful, single-topic drift, two-topic clean, most-repetitive
    sfaith = max(
        single, key=lambda i: np.mean(dyn[i]["pred"]["GRU"].numpy() == gen[i]["assumed_label_id"])
    )
    sdrift = min(
        single, key=lambda i: np.mean(dyn[i]["pred"]["GRU"].numpy() == gen[i]["assumed_label_id"])
    )

    # two-topic: pick a balanced record — GPT switch near 0.5 and both topics present in Gemma's dyn-GRU preds
    def twoscore(i):
        frac, _ = gpt_true_switch_frac(man[i]["gpt_response_raw"])
        if frac is None:
            return -1
        pr = dyn[i]["pred"]["GRU"].numpy()
        tset = [TOPICS.index(t) for t in gen[i]["topics"]]
        both = min(np.mean(pr == tset[0]), np.mean(pr == tset[1]))  # min share of the two topics
        return both - abs(frac - 0.5)  # balanced GPT split AND both topics detected

    twoc = max(two, key=twoscore) if two else None
    show(sfaith, "SINGLE-TOPIC, Gemma faithful")
    show(sdrift, "SINGLE-TOPIC, Gemma drifts most")
    if twoc is not None:
        show(twoc, "TWO-TOPIC")
    show(worst[0][0], "MOST REPETITIVE Gemma generation")


if __name__ == "__main__":
    main()
