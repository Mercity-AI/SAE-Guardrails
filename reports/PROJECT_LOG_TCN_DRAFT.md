# Real-Time Topic Classification with TCNs — Experiment Log

## Hypothesis

The hypothesis was that a small causal sequence model could read Gemma's
internal features token by token and identify the current topic while Gemma
generates. Unlike an offline classifier, this model must not look at future
tokens. It should recognize the topic at each point in the sequence and remain
stable until there is enough evidence that the topic has actually changed.

The experiments tested several related hypotheses:

- Scaling the dataset from roughly 1,500 records to roughly 2,000 would improve
  topic classification and expose the model to a better-balanced set of topic
  transitions.
- A temporal convolutional network could provide causal history without the
  recurrent hidden state used by a GRU.
- Different TCN blocks and receptive-field sizes would change both topic
  accuracy and temporal stability.
- The main remaining error might not be topic recognition, but
  **over-segmentation**: predicting many short, false topic changes inside an
  otherwise stable segment.
- Stable decoding, a temporal smoothing objective, or a dedicated boundary head
  might reduce those false transitions.

## Data and methodology

### The 2K topic-transition dataset

We generated 2,000 prompt-and-response pairs with Simula. Every prompt and
response contains one or two contiguous spans drawn from seven topic areas:
enterprise documents, general news and content, customer service, legal,
financial, HR and people operations, and healthcare.

The records were checked for valid topic tags and clean span structure.
Eighteen malformed records were removed, leaving **1,982 clean records**. Of
the original records, 1,070 were single-topic examples and 930 contained a
topic transition. All 42 possible ordered topic pairs appeared. The clean
dataset was divided into 1,387 training records, 297 validation records, and
298 test records. The test set remained untouched while models and settings
were selected on validation.

### How each token becomes an input

The topic tags provide the training labels, but the tags themselves are removed
before the text is processed. The prompt and response are rendered with Gemma's
real chat template so that the model sees the same kind of sequence it will see
in deployment.

For every token, we collect Gemma's hidden state from ten layers. Each hidden
state is passed through its matching sparse autoencoder, and the 50 most useful
SAE features from each layer are kept. Concatenating those layers gives **500
SAE features per token**. Feature selection is fitted on the training records
only.

The time dimension is preserved throughout. A conversation with `N` tokens
becomes a sequence of `N × 500` features. Prompt and response topic tokens
both receive supervision. Chat-template and other non-topic tokens remain
available as context but do not contribute to the topic loss.

### Common training configuration

Unless an ablation explicitly changed something, the models used the same
setup:

| Setting | Configuration |
|---|---|
| Topic classes | 7 |
| Input per token | 500 SAE features from 10 layers |
| Epochs | 15 |
| Batch size | 16 sequences |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.01 |
| Dropout | 0.1 |
| Model selection | Best validation macro-F1 |
| Final report | Untouched test split |

All architectures are causal: the prediction at a token can use that token and
earlier tokens, but never later ones. Full-sequence GPU evaluation produces the
same result as feeding growing prefixes one at a time. Processing the sequence
in parallel therefore does not give the model access to the future.

### Architectures compared

The **baseline TCN** uses six residual causal-convolution blocks with a
127-token receptive field. The **gated TCN** uses WaveNet-style gates over the
same history. The **MultiScale TCN** combines several convolution widths inside
each block so it can detect both short and longer patterns. The **ConvNeXt
TCN** uses causal depthwise convolution, per-token normalization, channel
expansion, and residual connections. Its original configuration sees 379 tokens
of causal history.

These models receive exactly the same token features and labels. The comparison
therefore asks whether the temporal architecture itself changes the result.

## Evaluation

### The two tiers of metrics

We evaluate the system at two levels because a model can label most tokens
correctly while still switching topics too often.

**Topic labeling** measures whether each supervised token receives the correct
topic. We report token accuracy, macro-F1 or macro-Dice, and macro-IoU. For
hard categorical predictions, per-topic F1 and Dice are mathematically
identical. Each topic is treated as a binary token mask, its overlap is
calculated, and the seven topic scores are averaged equally.

**Boundary detection** measures whether predicted topic changes occur near the
annotated topic changes. A prediction is counted as correct when it can be
matched one-to-one to a true boundary within either ±5 or ±10 tokens. We
report boundary precision, recall, F1, the number of predicted and true
boundaries, and the timing error for matched boundaries.

Precision answers: *when the model says the topic changed, how often was there
a real change?* Recall answers: *of all real changes, how many did the model
find?* Boundary F1 balances the two.

### What the overlap score actually measures

The overlap calculation pools all supervised test tokens for each topic. It
compares the set of token positions predicted as that topic with the set
labeled as that topic, then averages across the seven topics. It is therefore a
**dataset-level token-overlap score**. It is not the distance between predicted
and true boundaries, and it does not say that a boundary was off by a certain
number of tokens.

For example, if a transition is recognized ten tokens late, those ten tokens
count as topic-label errors and reduce Dice and IoU. The overlap score reflects
how much of the two topic regions is still correct, but it does not directly
report “ten tokens late.” The tolerant boundary metric does report whether
the transition was close enough, while the signed timing error is the right way
to quantify how early or late it was.

The overlap and boundary scores can diverge sharply. A one-token wobble affects
very little of the overall topic region, but it creates two false boundaries:
one switch into the wrong topic and one switch back. This is why high Dice does
not automatically imply clean segmentation.

We report prompt, response, and combined-stream views separately. The combined
stream includes the prompt, the assistant response, and the topic reset that
can occur at the turn transition. The test set contains 131 prompt boundaries,
131 response boundaries, and 131 turn resets, for 393 combined chronological
boundaries.

## Results

### Architecture comparison

The corrected architecture comparison used the shared training configuration
and supervision on both prompt and response tokens.

| Architecture | Best epoch | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|
| Baseline TCN | 8 | 88.82% | 88.59% |
| Gated TCN | 12 | 87.44% | 87.39% |
| MultiScale TCN | 14 | 89.90% | 89.77% |
| **ConvNeXt TCN** | **10** | **91.94%** | **91.67%** |

ConvNeXt was the strongest architecture. The result supported the central
modeling hypothesis: causal TCNs can recover the active topic from SAE features
with strong token-level accuracy.

### Best ordinary TCN

The strongest topic-only model was a wider ConvNeXt variant with a 505-token
receptive field.

| Partition | Accuracy | Macro-F1 / Dice | Macro-IoU |
|---|---:|---:|---:|
| Prompt | 88.71% | 89.09% | 80.46% |
| Response | 96.77% | 96.59% | 93.48% |
| Combined | **94.30%** | **94.35%** | **89.36%** |

These figures mean that the predicted topic masks overlap strongly with the
true topic masks on the synthetic held-out set. They do not mean that 94.35% of
complete segments or boundaries are correct.

The raw boundary results tell a different story:

| Tolerance | Precision | Recall | Boundary F1 | Predicted | True |
|---|---:|---:|---:|---:|---:|
| ±5 tokens | 10.39% | 87.02% | 18.57% | 3,291 | 393 |
| ±10 tokens | 11.46% | 95.93% | 20.47% | 3,291 | 393 |

The model finds the neighborhood of most real transitions, but it predicts far
too many extra ones. Its topic labeling is strong; its raw temporal
segmentation is not deployable.

## Ablations

### Starting from the boundary problem

The original 379-token ConvNeXt produced 5,392 combined transitions for only
393 true transitions. At ±5 tokens it achieved 88.55% recall but only 6.45%
precision. This established over-segmentation—not broad topic
recognition—as the central failure to address.

### Causal decoding interventions

We first kept the trained ConvNeXt fixed and changed only how its left-to-right
predictions were turned into topic states. Minimum-duration filtering required
a new topic to persist briefly. Hysteresis required the challenger to beat the
current topic by a confidence margin. A transition penalty required accumulated
evidence before allowing a change.

| Decoder | Boundary precision | Boundary recall | Boundary F1 | Predicted boundaries |
|---|---:|---:|---:|---:|
| Raw predictions | 6.45% | 88.55% | 12.03% | 5,392 |
| Hysteresis | 6.84% | 88.30% | 12.69% | 5,074 |
| Transition penalty | 14.71% | 77.61% | 24.73% | 2,074 |
| Minimum duration | **19.91%** | 56.74% | **29.48%** | **1,120** |

Minimum duration produced the highest F1 but discarded many real transitions.
The transition penalty was better balanced because it retained more recall.
Hysteresis barely helped: many false switches were confident rather than
marginal decisions. Post-processing clearly reduced noise, but every method
traded delayed or missed transitions for stability.

### Temporal-consistency training

We next encouraged the model's topic distribution to remain similar between
adjacent tokens that belonged to the same true segment. This follows the
temporal smoothing idea used in prior multi-stage TCN work. The regularized
model improved topic macro-F1 from 91.67% to 92.44% and reduced predicted
boundaries from 5,392 to 3,986. Boundary F1 rose only from 12.03% to 13.38%.

The hypothesis was directionally correct, but the gain was too small.
Encouraging smooth topic probabilities is not the same as explicitly learning
where a boundary exists.

### Causal-history width

We compared ConvNeXt models with 43-, 379-, and 505-token receptive fields.

| Causal history | Topic F1 | Prompt boundary F1 | Response boundary F1 | Combined boundary F1 |
|---:|---:|---:|---:|---:|
| 43 tokens | 93.54% | 7.64% | 12.37% | 15.02% |
| 379 tokens | 91.67% | 8.26% | 7.62% | 12.03% |
| **505 tokens** | **94.35%** | **9.51%** | **16.10%** | **18.57%** |

The 505-token model was strongest overall. The good short-context result shows
that much of the topic evidence is local, but the longer history helps on
harder transitions and produces the best combined topic and boundary result
among ordinary TCNs.

### Learning rate

The shared architecture comparison used the same learning rate for fairness,
but that rate was not necessarily optimal for ConvNeXt. Lowering it from 0.001
to 0.0003 improved test topic F1 from 91.67% to 93.53%. This shows that part of
the architecture comparison was influenced by optimization, although the
505-token ConvNeXt remained the strongest final topic-only model.

## Joint topic and boundary model

### Hypothesis and design

The ordinary TCN learns only one thing: which topic is active at every token.
Its boundaries are then inferred indirectly whenever two adjacent topic
predictions differ. That makes every brief classification mistake look like two
topic transitions.

The joint model tests a cleaner hypothesis: topic identity and topic transition
should be learned as related but distinct decisions. A shared causal ConvNeXt
reads the same 500-dimensional token sequence and develops one representation
of the recent context. One output head names the active topic. A second output
head decides whether the current token begins a new topic segment.

Both decisions use the same causal evidence, but the boundary head is trained
directly on the annotated transition points. Because boundaries are rare,
boundary examples receive additional weight during training. The purpose is
simple: the model should not have to express every change through unstable
topic argmax predictions; it gets a dedicated mechanism for learning what a
real transition looks like.

### Evaluation

The joint head's reported boundary score uses exact-token decisions rather than
the ±5 or ±10 tolerance used for the ordinary topic-derived boundaries. That
makes its boundary result stricter, but it also means the two tables are not
perfectly like-for-like. Future reporting should apply both exact and tolerant
evaluation to every model.

Topic performance and boundary performance favored different stages of
training. The model state with the best topic labeling achieved 87.73% combined
topic F1 and 42.66% exact boundary F1. The state with the best boundary
detection produced the following result:

| Partition | Topic F1 | Boundary precision | Boundary recall | Exact boundary F1 |
|---|---:|---:|---:|---:|
| Prompt | 73.92% | 49.76% | 79.39% | 61.18% |
| Response | 88.05% | 21.78% | 78.63% | 34.11% |
| Combined | **83.78%** | **48.97%** | **84.99%** | **62.14%** |

It predicted 682 combined boundaries for 393 true boundaries. This is still too
many, but it is the first model whose transition count is in the same order of
magnitude as the ground truth. Compared with the topic-only ConvNeXt, it gives
up substantial token-level topic accuracy in exchange for a much cleaner and
more explicit boundary signal.

The prompt-response difference is important. Prompt boundary F1 reached 61.18%,
while response boundary F1 was only 34.11%. Prompts tend to state their
separate requests more explicitly; responses use connective language and longer
explanations, making the exact transition harder to locate. A combined score
alone hides this deployment-relevant weakness.

## Interpretation

The TCN experiments support several conclusions. The SAE features contain
enough information for strong live topic classification, and ConvNeXt is the
best temporal block tested so far. More causal history helps, but context width
is not the main obstacle. The best ordinary TCN labels tokens extremely well on
this dataset while still fragmenting its output into thousands of false
segments.

Simple decoding rules reduce fragmentation but impose an unavoidable
precision-recall tradeoff. Temporal consistency helps modestly, confirming that
smoothness matters, but it does not teach the model the difference between a
momentary topic ambiguity and a genuine transition.

The joint-head result is the strongest evidence about the task itself. Explicit
boundary supervision improves segmentation much more than smoothing topic
predictions after the fact. At the same time, its loss in topic F1 shows that
the two objectives compete under the current setup. The next model should
preserve the strong 505-token topic representation while using the boundary
head to control transitions, rather than forcing one checkpoint to optimize
both outcomes equally.

The overall result is therefore mixed but useful: **topic recognition is
strong; boundary detection has improved substantially with explicit
supervision; response-side transition timing remains the largest unresolved
modeling problem.**

## Next evaluation and modeling steps

The next report should apply the same evaluation protocol to every
architecture: topic accuracy, macro-Dice and macro-IoU; exact and tolerant
boundary precision, recall, and F1; prompt, response, and combined partitions;
signed early-or-late boundary delay; and the ratio of predicted to true
transitions. Per-conversation overlap and WindowDiff should be added so that
pooled token accuracy cannot hide fragmented individual sequences.

For modeling, the most direct next experiment is structured decoding—Viterbi
or a CRF—using the topic and boundary signals together. This would score a
coherent topic sequence rather than making every token decision independently.
Combinations of the current interventions have not yet been tested; they were
kept separate here so that each result could be interpreted cleanly. 
