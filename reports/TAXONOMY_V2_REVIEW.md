# Topic-drift taxonomy v2 review

## Current 1,500-row distribution

- Strategy counts: everyday single-topic 679; workplace 426; public-information 242; personal-case 153.
- Topic count: 666 single-topic and 834 two-topic records.
- Topic appearances: Customer service 694; HR 336; Healthcare 321; Enterprise 267; General 267; Financial 230; Legal 219.
- Ordered-pair coverage: 41/42. Financial -> Customer service is absent. HR -> Enterprise has 94 rows, while several directions have only 1-4.
- Boundary lineage: midpoint 701; early 399; late 315; unmarked 43; near-end 42.

## Principal defect

The strategies pin almost every factor, collapsing a large taxonomy into four correlated bundles. In realized data:

- all 679 everyday-single rows are action-planning and mostly casual consumer scenarios;
- all 426 workplace rows are formal drafting scenarios with midpoint boundaries;
- all 242 public-information rows are casual information requests with early boundaries;
- all 153 personal-case rows are urgent escalation scenarios with late boundaries.

This permits shortcut learning and leaves many taxonomy branches with zero realized coverage. Empty `never_combine` rules also allow logically false lineage: single-topic rows receive two-topic boundary-placement leaves.

## V2 factors

1. Topic composition and order.
2. Topic-pair relationship, with exact directed pairs and single-topic N/A.
3. Boundary difficulty, separated from position and with a single exact semantic boundary.
4. Boundary placement and segment balance, using measurable proportions and single-topic N/A.
5. Response length, using explicit word bands plus post-generation Gemma-token audits.
6. Surface structure, limited to mechanical layout and prohibited from aligning systematically with boundaries.

Persona, tone, task, and specificity remain free variation rather than quota-bearing axes.

## V2 strategies

- routine_single_topic: weight 0.70
- hard_single_topic_negative: weight 0.30
- confusable_two_topic: weight 0.40
- moderately_related_two_topic: weight 0.30
- distant_two_topic: weight 0.30

Strategies pin only the learning objective and applicability branches. Other factors remain broadly sampled. `never_combine` rules enforce single/two consistency.

## Required review gate after real taxonomy generation

1. Stop after strategies; do not generate records yet.
2. Manually edit the generated topic-pair tree until all 42 ordered directions exist.
3. Check N/A/no-boundary applicability and `never_combine` paths.
4. Sample at least 10,000 mixes for free.
5. Audit marginals and cross-tabs: single/two, seven topic appearances, 42 ordered pairs, relationship, difficulty, position, length, and layout.
6. Reject strategies that pin style/task/domain or stack multiple rare branches.
7. Run a 30-50-row paid pilot only after the sampled plan looks correct.
8. Audit actual response words, Gemma tokens, tag validity, exact topic order, boundary cue leakage, and whether visible structural breaks correlate with boundaries.
