# Topic-axis config review, edits, and pilot runs — session report (2026-08-04)

Scope: review of the `prefill_mvp_topic_axis` dataset config against simula's own playbook
(SKILL.md), evidence-driven edits, two real 15-row pilots, and cost/time projections for the full
run. The full 2,000-row run was launched and then deliberately stopped during its meta-prompt phase
on request; the config is left ready to fire.

## 1. Purpose recap

Generate prompt+response pairs with topic-labelled spans (seven domains, one or two topics per
record, at most one clean boundary) to prefill into Gemma 1B, extract SAE top-K features, and train
token-level topic classifiers (TCN et al.) for topic-drift detection. This config is the corrective
successor to `runs/1500`, whose persona-driven strategies collapsed the label distribution
(Customer service 29.7% of spans) and shipped logically false lineage.

## 2. Initial review — what the setup already did right

Measured against SKILL.md's strategy-phase checklist:

- The topic label is the controlled axis (uniform Primary/Second factors, hand-authored
  `taxonomy.json`/`strategies.json` reused verbatim); persona/tone/task stay free variation.
- One unpinned strategy + `never_combine` fixes the v2-review defects (four correlated bundles,
  single-topic rows with boundary lineage, A->A pairs).
- Free 40k-draw audit with hard asserts before any paid call.
- Custom prompt module makes the sampled topics mandatory and the critic rules target the exact
  historical failure modes (foreign tags, interleaving, boundary-naming).

## 3. Issues found, and what was done about each

| # | Finding | Evidence | Resolution |
|---|---|---|---|
| 1 | cwd-relativity foot-gun: `output_dir` and `.env` resolve from the launch directory; wrong cwd silently rebuilds a model taxonomy and writes to a stray tree | stray `/workspace/Simula/prefill-mvp/runs/taxonomy_v2/`; later bit this session's own first full-run launch (instant path error, $0) | Documented; always launch from `/workspace/local`. `.env` created there from `vars.txt` |
| 2 | Fake-model smoke can never satisfy the custom schema (canned `{input,output}` record), so the old smoke's 0/10 accepted proved plumbing only | `runs/topic_axis_smoke` raw rows | Understood; smoke path discarded by owner |
| 3 | Redraw conditioning: `never_combine` redraws mean the realized single/two split is the *surviving* fraction (~53% single), not the naive weight product | audits: 53.5% / 52.5% single | Documented as benign; tune the `None` weight against the audit if an exact ratio is ever needed |
| 4 | Over-specified "laundry-list" prompts (the perceived complexity of the 1500 run) | 1500-run samples; median prompt 69w, max 622w | PROMPT REGISTER paragraph added to the meta-prompt override (chat-message register, no enumerated sub-asks, no fixed sentence counts) |
| 5 | Placement/difficulty taxonomy too fine-grained; two placement leaves undefined; "Abrupt shift" mislabeled as hard | 1500-run adherence stats: 17 v1 placement nodes collapse to 3 realized behaviors | Difficulty cut to 3 leaves (Explicit 0.40 / Light 0.35 / Unmarked-or-shared-entity 0.25); short-block leaves removed; then the whole placement factor removed (owner call) |
| 6 | Response length factor unnecessary | v1 had no length factor and produced a healthy natural spread (median 207w, p90 381w) | Factor removed |
| 7 | "Table or FAQ-like" surface leaf contradicts the clean-prose rule — empirically ungenerable | pilot A: item-10 rejected 3/3 for exactly this | Leaf deleted (Surface structure now 5 leaves) |
| 8 | Critic over-enforced soft ingredients, inconsistently (burned ~10% of pilot-A attempts on placement %, accepted others violating the same band) | pilot A items 6, 10 vs 16 | Mostly mooted by removing placement + the FAQ leaf; critique-prompt softening noted but not applied |
| 9 | Tag-vs-topics structural leak the LLM critic misses (~1/15 per pilot): untagged second span (pilot A), truncated `</Enterprise>` closing tag (pilot B) | both pilots' final rows | Owner decision: accept for now; handle with future judge passes. Expect ~100–150 defective rows per 2k for the downstream aligner to drop |
| 10 | Description verbosity: 96 lines embedded in every meta/generate/critic call | llm_calls sizes | Compressed to ~47 lines; Includes/Excludes reduced to 2–3 vague lines per topic; GENERAL INSTRUCTIONS removed (each rule already lives in the schema, meta-prompt TOPIC RULE, or critic) |
| 11 | Raw-JSON ingredient dump in the meta-prompt (level/path noise) | prompt inspection | Mix now rendered as plain `- Factor: value — description` lines (`_mix_lines`) |

## 4. Config/knob changes (final state)

- Taxonomy factors (4): Primary topic, Second topic (None/Present), Boundary difficulty
  (No boundary | Explicit 0.40 / Light 0.35 / Unmarked-or-shared-entity 0.25), Surface structure
  (prose 0.30 / multi-paragraph 0.30 / bullets 0.15 / numbered 0.15 / mixed 0.07).
- `never_combine`: single<->boundary consistency pair + 7 A->A prohibitions (placement rules removed
  with the factor). Builder audit rerun clean: 42/42 ordered pairs, uniform marginals (14.2–14.4%),
  zero illegal combos.
- Models: bulk/critic/strategic considerations — critic switched Terra -> Luna (10x cheaper; see
  trade-off in §6). Strategic (Terra) makes zero calls while the design artifacts exist; it only
  fires — silently rebuilding the taxonomy — if `taxonomy.json` goes missing. Do not delete that file.
- Generation: `target_size: 2000`, `overgenerate_ratio: 1.2`, `complexity_ratio: 0.0`,
  `max_refine_attempts: 2`, `concurrency: 32`.

## 5. Pilot A — 15 rows, Terra critic (superseded)

21 attempts -> 19 accepted (90.5%) -> 15 final, 46 s, ~$0.09 (Terra critic ≈ 80% of cost).
Register fix confirmed: prompt median 51w / max 106w (vs 69w / 622w in the 1500 run). Refine loop
repaired real defects (visible headings, split spans, unclosed tag). 1/15 structural leak
(untagged second span). Two attempts fully burned on soft-ingredient rejections (see §3.8).

## 6. Pilot B — 15 rows, Luna critic (current baseline)

20 attempts -> 20 accepted (100%) -> 15 final, 33 s, $0.023 (89.9k in / 22.8k out tokens).
3 refines succeeded, 2 JSON auto-repairs. Prompt median 83w (one 233w outlier), response median
151w, 7 single / 8 two-topic. 1/15 structural leak (truncated closing tag, accepted by Luna).

Trade-off observed: Terra critic rejects more (some legitimately) at 10x the cost; Luna accepts
everything. Final-dataset defect rate was ~1/15 under both — the leak class is critic-independent
and belongs to a deterministic post-check or judge pass, not a stronger critic.

## 7. Projections for the full run (from pilot B)

| Target | Cost | Wall-clock (concurrency 32) |
|---|---|---|
| 2,000 | ~$3.0 | ~75 min extrapolated; 1.5–2.5 h realistic envelope |
| 3,000 | ~$4.5 | ~110 min; 2–3.5 h envelope |

Pricing basis: OpenRouter GPT-5.6 Luna $0.10/M input, $0.60/M output. At 100% accept,
`overgenerate_ratio: 1.2` trims ~400 surplus rows on a 2k run (~$0.6 — immaterial).

## 8. Full-run launch and deliberate stop

The 2,000-row run was fired (after one $0 false start from the wrong cwd — §3.1) and then stopped
on request during the meta-prompt phase. State at stop:

- `meta_prompts.jsonl`: 635 rows (20 pilot + 615 new) — these persist and are reused on the next
  launch, so the spend is not wasted.
- `dataset.raw.jsonl`: 20 rows (pilot only; no new generation attempts had started).
- Total session spend across everything: 683 calls, 1.00M in / 0.23M out tokens ≈ **$0.24**.

## 9. How to fire the full run (nothing else needed)

```bash
cd /workspace/local        # cwd matters: .env and output_dir resolve from here
python -m simula.cli run prefill-mvp/prefill_mvp_topic_axis.yaml
```

Resume is on: the 15 pilot rows count toward the 2,000 and the 635 meta prompts are reused.
Checkpoints every 25 rows make interruptions cheap.

**Caveats before firing:**

1. If *any* further edit touches the description, schema, prompt module, taxonomy, or strategies,
   resume will refuse (fingerprint) — rerun with `--no-resume` **and also delete
   `meta_prompts.jsonl`**, because meta prompts are deliberately outside the fingerprint and stale
   ones (e.g. still referencing removed ingredients) would otherwise be reused.
2. After the run, check the console accept count and `eval_report.json`; if final < 2,000 the accept
   rate dropped — investigate `dataset.raw.jsonl` rejection reasons before raising overgenerate.
3. Budget ~5–7% of final rows failing strict tag alignment downstream (§3.9) until a judge pass
   exists.

## 10. Deferred / open items

- Subdomain leaves under each topic (diversity for ~1,000+ single-topic rows) — owner: later.
- Deterministic tag/structure post-check or model judge pass over the final dataset — owner: later.
- Critique-prompt softening ("ingredients are soft preferences") — unnecessary while placement and
  the FAQ leaf are gone; revisit if new soft-ingredient rejections appear.
- Stale artifacts not cleaned: `prefill_mvp_topic_axis_smoke.yaml`, `runs/topic_axis_smoke/`,
  `/workspace/Simula/prefill-mvp/` (wrong-cwd stray).
