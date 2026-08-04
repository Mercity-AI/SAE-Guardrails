# Sticky-Viterbi decoding for topic-drift boundaries — experiment report (2026-08-04)

Scope: diagnosis of the over-segmentation failure in the prefill-MVP topic-drift detector, a
zero-retraining fix (sticky-Viterbi decoding of the frozen convnext posteriors), and a full audit
that corrects an evaluation-scope inflation in the headline number. All numbers are on the persisted
70/15/15 response-level split of `runs/2k_run_5.6` (train 1387 / validation 297 / test 298),
using the frozen checkpoint `results/convnext_windows/window_505/checkpoint_best.pt`
(convnext, channels 256, depth 6, kernel 9, expansion 4) over the 500-dim train-selected SAE feature
cache `cache/sae500_2k_clean_prompt_response`. Reproduce with `viterbi_decode.py`; raw sweep in
`results/viterbi_window_505/sweep.json`.

## 1. Problem recap

The detector labels every token with one of seven topics (Gemma-1B prefill → SAE top-K sparse
features → causal convnext-TCN). Token accuracy and macro-F1 are strong (~0.94), but **boundary**
detection — the position where the topic changes — was near-useless: high recall, ~0.10 precision.

Root cause (established in prior analysis, confirmed here): boundaries were never a trained target.
They are read off post-hoc as any position where the per-token **argmax** changes, and the loss is
plain per-token cross-entropy with no temporal-persistence term. Independent argmax has no memory, so
inside a single true segment it flickers between topics on ambiguous tokens, and every flicker becomes
a spurious boundary. On the test split this produced ~11 predicted boundaries per sequence against a
true rate of ~1.3.

## 2. Method: sticky-Viterbi decoding

The fix changes the **decoder, not the model** — no gradient steps, same frozen checkpoint.

Instead of taking the argmax topic at each token independently, we choose the single label *path* that
maximizes

```
score(path) = Σ_t  logP(topic_t | token_t)   −   λ · (number of label switches in path)
```

- The first term rewards agreeing with the model's per-token posterior (emission score).
- The second term charges a fixed penalty **λ** for every position where the label changes.

The maximizing path is found exactly by the Viterbi dynamic program (O(T·K²), K=7 topics), using a
Potts transition (stay = 0, any switch = −λ). **λ is a single interpretable knob = "how much evidence
is required before declaring a topic change."** λ = 0 reproduces argmax exactly; λ → ∞ forces one topic
for the whole sequence. λ is swept on validation and selected by boundary-F1; the test split is scored
once at the selected λ.

## 3. Headline result (full metric, all boundaries)

Validation selection picked **λ = 100** (a genuine interior peak; validation bF1@5 turns over by
λ = 150). Test, in original token coordinates, tolerance ±5 tokens:

| metric | argmax (λ=0) | sticky-Viterbi (λ=100) |
|---|---|---|
| boundary F1 @5 | 0.186 | **0.692** |
| boundary F1 @10 | 0.205 | **0.841** |
| boundary precision @5 | 0.104 | 0.714 |
| boundary recall @5 | 0.870 | 0.672 |
| predicted boundaries / seq | 11.04 | 1.24 |
| false alarms / no-boundary seq | 6.20 | 0.04 |
| token accuracy | 0.943 | 0.967 |
| token macro-F1 | 0.943 | 0.967 |

Token accuracy *rises* under Viterbi — direct evidence that the removed "errors" were transient
flickers inside true segments. Condensed validation sweep (original coordinates):

| λ | tok acc | bP@5 | bR@5 | bF1@5 | pred/seq | FA/no-boundary seq |
|----|------|------|------|------|------|------|
| 0   | 0.938 | 0.099 | 0.876 | 0.178 | 12.24 | 7.62 |
| 10  | 0.951 | 0.400 | 0.759 | 0.524 | 2.63  | 0.96 |
| 40  | 0.961 | 0.646 | 0.727 | 0.684 | 1.56  | 0.17 |
| 100 | 0.964 | 0.741 | 0.696 | **0.718** | 1.30 | 0.04 |
| 150 | 0.962 | 0.768 | 0.659 | 0.709 | 1.19  | 0.02 |

## 4. Audit — verification and a corrected interpretation

The headline looked suspiciously high, so the implementation and evaluation were audited.

**Verified correct (no bug):** the evaluation was rewritten to call the pipeline's own
`boundary_report` in original token coordinates. It reproduces the stored `results.json` exactly
(argmax bF1@5 = 0.186; test true-boundary count = 393). The convnext is causal (left-padded dilated
convolutions), so the real-time framing is intact for the *model* (the Viterbi backtrace itself is
offline — see §6).

**No tag leakage:** the data builder (`strip_and_label`) discards the `<Topic>…</Topic>` tags and
keeps only the inner prose; any record with a surviving tag is dropped. The model sees clean text, so
boundary detection is genuinely semantic, not tag-reading.

**Scope inflation found — the headline counted the wrong boundaries.** The 393 test "true boundaries"
decompose as **167 single-topic sequences (0 boundaries) + 131 two-topic sequences with 3 each**. A
two-topic record tokenizes to segments `[A_prompt, B_prompt, A_response, B_response]`, yielding three
switches:

1. `A→B` inside the **prompt** — a real change, but in the *user's* text, not the model's response.
2. `B→A` at the **prompt→response junction** — an **artifact**: `boundary_positions` glues the last
   prompt token to the first response token across the masked template gap. It is the turn boundary,
   not a drift.
3. `A→B` inside the **response** — the **only** boundary that is "drift in the model's response," i.e.
   the deployment objective.

Two of the three counted boundaries are not the target and are easier than the real one, inflating the
score.

## 5. Corrected result — response-internal drift only (the true objective)

Re-scoring restricted to response tokens (`role_id == 2`) gives true count = **131** (exactly one per
two-topic sequence, matching the taxonomy's "at most one boundary" design). λ re-selected on the
response-scoped validation objective (λ = 200; the F1 peak is broad/flat from ~100 upward):

| | boundary P@5 | R@5 | **F1@5** | F1@10 | predicted (true = 131) |
|---|---|---|---|---|---|
| argmax (current) | 0.088 | 0.908 | **0.161** | 0.172 | 1347 |
| sticky-Viterbi | 0.573 | 0.779 | **0.660** | 0.751 | 178 |

## 6. Conclusions

- **The diagnosis is fully confirmed.** Over-segmentation is a decoding problem. On the correctly
  scoped objective the argmax decoder predicts **1347** response-boundaries against **131** real ones
  (≈10× hallucination); sticky-Viterbi cuts this to 178 with zero retraining.
- **Sticky-Viterbi is a large, genuine win**: response-internal drift F1 **0.16 → 0.66** (@5) and
  **0.75** (@10), plus a free token-accuracy bump (0.943 → 0.967).
- **The first "0.69" headline was inflated** by easy prompt boundaries and the junction artifact. The
  honest number for drift in the model's response is **~0.66**, with **precision 0.57** (~4 in 10
  flagged drifts still false) and **recall 0.78** (~1 in 5 real drifts missed at ±5 tokens). Good, not
  magic; the residual weakness is precise localization (the @5→@10 gap).
- **λ is the precision/recall dial.** Lower λ (~40) trades precision for recall if missing a drift is
  the costlier error.

## 7. Recommended next steps

1. **Re-scope training and evaluation to response-internal boundaries** regardless of decoder — the
   current metric partly optimizes a turn-boundary artifact. This is the highest-value change.
2. **Fixed-lag Viterbi for deployment.** The exact Viterbi backtrace is offline (needs the full
   sequence). For streaming, commit a token's label once it is D tokens in the past (D ≈ tolerance);
   same math, bounded latency.
3. **Optional: learn the transition prior with a linear-chain CRF head** instead of a hand-tuned scalar
   λ, allowing the switch cost to depend on the topic pair. Justified only if it buys measurable recall
   over a single global λ, given how well one λ already works.

## Appendix: token overlap metrics (Dice / IoU)

Per topic (one-vs-rest over token positions), the **Dice coefficient equals F1 exactly**:
`Dice = 2·TP / (|A|+|B|) = 2·TP / (2·TP+FP+FN) = F1`. Hence macro-F1 already *is* mean per-topic Dice;
reporting "macro-Dice" reproduces it. **IoU/Jaccard** is a monotonic transform (`Dice = 2J/(1+J)`),
so per class it ranks identically to F1; only macro-averaging can mildly reorder models. All three are
**token-overlap** metrics and remain **boundary-insensitive** — they do not measure the drift-boundary
problem and add no new signal about it. Only the tolerance-based boundary P/R/F1 (§5, response-scoped)
sees it.
