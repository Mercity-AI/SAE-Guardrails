# Teacher-Forced Prefill for Real-Time Topic-Drift Detection

## 1. The hypothesis

Our earlier approaches asked the base model to generate complete answers and then tried to recover
topic membership or boundaries from those generations. That was expensive, slow to scale, and
difficult to supervise precisely. Natural answers also tend to blend adjacent subjects, which makes
the location of a supposedly exact boundary debatable.

The new hypothesis was simpler:

> What if a stronger teacher writes both the prompt and the response, including clean hidden topic
> boundaries, while the smaller model only reads that exchange in a single prefill pass? Can a small
> sequence classifier learn token-level topic changes from the smaller model's resulting internal
> features?

This separates data creation from representation extraction. The teacher provides controlled text
and exact span labels. The smaller model supplies the internal activity that will also be available
at deployment. Prefill replaces slow autoregressive generation during training-data extraction.

## 2. How the dataset was generated

The generation run targeted 1,500 prompt-and-response pairs across seven areas: enterprise
documents, general news and content, customer service, legal, financial, human resources and people
operations, and healthcare.

Each record contained either one topic or two topics:

- A single-topic response remained within one area from beginning to end. These records served as
  negative examples for drift: the detector should not invent a change.
- A two-topic response completed its first topic, crossed one clean sentence-level boundary, and
  then completed the second topic without returning to the first.

The dataset contained 666 single-topic records and 834 two-topic records. Topic selection, order,
persona, tone, structure, and boundary placement were deliberately varied. The prompt and response
contained inline span tags for auditing, but the tags were removed before the smaller model saw the
text.

A stronger reasoning model planned the diversity taxonomy and judged quality. A faster teacher
model produced the bulk records. Both were run through the flexible service tier. The generation
pipeline overproduced candidates, deduplicated them, checked taxonomy coverage, and retained 1,500
final records.

The final topic counts were not perfectly balanced:

| Topic | Records containing it |
|---|---:|
| Customer service | 694 |
| Human resources and people operations | 336 |
| Healthcare | 321 |
| Enterprise documents | 267 |
| General news and content | 267 |
| Financial | 230 |
| Legal | 219 |

Of the 1,500 records, 1,458 aligned cleanly after chat templating and tokenization. Forty-two were
excluded because the cleaned prompt or response could not be located reliably after formatting.
No partially aligned record was silently repaired.

## 3. From teacher text to token features

The teacher-generated prompt and response were formatted as an ordinary user-assistant exchange.
The smaller model then read the complete exchange in one causal prefill pass. It generated no new
tokens.

Although the response was available in full, the model's causal attention preserved the live-use
condition: the internal state at a response token could use the prompt and earlier response tokens,
but not later response tokens.

We collected the residual representation from each of the final ten internal layers. Each layer's
representation was passed through its corresponding sparse autoencoder. Fifty consistently active
sparse features were retained from each layer, producing a 500-value feature vector for every token.

Only tagged response-topic tokens contributed to training and evaluation. Prompt tokens, chat
formatting, and untagged connective text were ignored. This removed the earlier neutral class and
prevented easy formatting tokens from inflating the score.

The resulting shared cache contained 576,002 token vectors. Every classifier used the same cache,
the same seeded response-level split, and the same labels:

- 1,167 responses for training
- 291 responses for held-out evaluation
- 15 training epochs
- Best checkpoint selected by held-out macro F1

## 4. What the classifiers were asked to learn

At every response token, a classifier received the current 500-value sparse feature vector and the
history allowed by its architecture. It returned one of the seven topic labels.

This is deliberately a token-level classification task. A drift boundary is inferred when the
stable predicted topic changes from one class to another. The experiment therefore has two levels
of success:

1. Can the classifier name the topic of individual tokens?
2. Are those predictions stable enough to identify one clean boundary without producing many false
   changes?

The first question is measured by token accuracy and macro F1. Accuracy is the fraction of all held-
out topic tokens classified correctly. Macro F1 calculates an F1 score for every topic and gives all
seven equal weight, preventing the largest classes from hiding failures on smaller ones.

## 5. The recurrent baseline

The baseline was a three-layer causal gated recurrent network with a 256-value hidden state. It read
the sparse token vectors in order and maintained a recurrent memory of the preceding response.

On the 1,500-record dataset, its held-out macro F1 peaked at 51.2% but later regressed, finishing at
36.4%. The network tended to collapse toward the most common customer-service class. This was an
important baseline rather than a successful endpoint: more data alone did not make this particular
recurrent setup stable.

## 6. Temporal convolutional networks

The temporal convolutional model replaced recurrent memory with causal one-dimensional
convolutions. Each layer looked backward over the token sequence, and progressively wider spacing
allowed later blocks to cover a longer history. Residual connections preserved the current token's
signal while the network combined it with earlier context.

The starting model used six residual blocks and 128 channels. We then changed one factor at a time:
depth, width, and learning rate.

| Temporal convolutional configuration | Held-out accuracy | Macro F1 |
|---|---:|---:|
| Six blocks, 128 channels, standard learning rate | 70.8% | 64.4% |
| Eight blocks, 128 channels | 71.1% | 65.8% |
| Six blocks, 256 channels | **74.2%** | **70.1%** |
| Six blocks, lower learning rate | 72.4% | 66.7% |
| Six blocks, higher learning rate | 70.6% | 63.9% |

Making the convolutional model wider helped much more than making it deeper. The wider model gained
almost six macro-F1 points over the starting configuration. The higher learning rate was harmful;
the lower rate was safer but still did not match the wider network.

## 7. Causal Transformers

The Transformer projected each 500-value token vector into a smaller internal representation and
used masked self-attention to compare it with preceding response tokens. The mask prevented access
to future tokens, preserving the live-deployment condition. A topic-classification head produced a
label at every position.

The starting model used two Transformer layers, four attention heads, and an internal width of 128.
We tested greater depth, greater width, and three learning rates.

| Transformer configuration | Held-out accuracy | Macro F1 |
|---|---:|---:|
| Two layers, width 128, standard learning rate | 77.5% | 72.5% |
| Four layers, width 128 | **79.2%** | **75.0%** |
| Two layers, width 256 | 78.3% | 73.7% |
| Two layers, lower learning rate | 74.7% | 68.8% |
| Two layers, higher learning rate | 78.5% | 74.1% |

The four-layer Transformer was the strongest classifier in the experiment. Depth helped more than
doubling width, while using fewer parameters. The higher learning rate also improved the two-layer
baseline, but did not overtake the deeper model. The lower rate learned too slowly within the fixed
15-epoch budget.

## 8. Overall comparison

| Best version of each family | Held-out accuracy | Macro F1 |
|---|---:|---:|
| Three-layer recurrent baseline, best observed epoch | — | 51.2% |
| Wider temporal convolutional network | 74.2% | 70.1% |
| Four-layer causal Transformer | **79.2%** | **75.0%** |

The result supports the central hypothesis: teacher-forced prefill produces internal sparse
features that can train a useful token-topic classifier. The strongest model correctly classified
nearly four out of five held-out topic tokens and maintained a 75% F1 average when every topic was
weighted equally.

The architecture mattered substantially. Both the temporal convolutional network and Transformer
outperformed the recurrent baseline. The Transformer benefited from directly comparing the current
token with earlier positions instead of compressing the entire prefix into one recurrent state.

## 9. Per-topic results

| Topic | Wider temporal convolutional F1 | Four-layer Transformer F1 |
|---|---:|---:|
| Enterprise documents | 59.4% | **63.6%** |
| General news and content | **76.5%** | 76.1% |
| Customer service | 87.2% | **92.7%** |
| Legal | 63.6% | **67.8%** |
| Financial | 68.2% | **73.0%** |
| Human resources and people operations | 50.7% | **62.2%** |
| Healthcare | 85.3% | **89.4%** |

Customer service and healthcare were the cleanest classes. Their language often contains concrete,
distinctive cues: accounts, orders, refunds, symptoms, treatments, and clinical actions. The
Transformer classified more than nine out of ten customer-service and healthcare tokens correctly
by F1.

Enterprise documents and human resources remained the hardest pair. The confusion matrix showed
that internal policies, procedures, team operations, and employee management frequently activated
overlapping patterns. The Transformer improved both classes, but human resources remained the
weakest at 62.2% F1.

Legal also had an asymmetric problem. When the Transformer predicted Legal it was usually correct,
but it recovered only about 57% of the true Legal tokens. Legal passages were often absorbed into
financial, human-resources, or enterprise-document predictions when a contract or regulation
appeared inside a broader workplace or money-related scenario.

General news performed well despite having fewer training records. The wider convolutional model
was marginally better there, but the difference from the Transformer was negligible.

## 10. What the result means for drift detection

The experiment succeeded at token-level topic recognition, but raw token predictions are not yet a
finished drift alarm. Even the stronger models sometimes alternate briefly between adjacent topics
inside a single true section. Treating every one-token label change as a boundary therefore creates
many false changes.

The correct claim is:

> Pure prefill over teacher-generated responses can produce a strong causal token-topic classifier,
> and a causal Transformer currently reads those sparse feature trajectories best.

The experiment does not yet establish that every raw predicted transition is a reliable stopping
point. Topic recognition and stable change detection are related, but they are not the same metric.
The synthetic data provides exact boundaries; the remaining work is to make the predicted
trajectory respect them without hiding genuine changes.

## 11. Mistakes and practical findings

| What we found | Why it mattered | Resolution |
|---|---|---|
| Untagged and formatting tokens formed an easy neutral class | It inflated aggregate scores without improving topic recognition | Removed them from both training and evaluation |
| Only 100 examples produced unstable and missing-class validation results | Architecture comparisons changed with a handful of examples | Scaled to 1,500 records and used one shared response-level split |
| Repeating prefill for every model wasted most of the runtime | Classifier ablations do not require new base-model features | Stored one shared 500-feature token cache |
| The recurrent model collapsed toward common classes | A single recurrent state was not using these noisy features reliably | Tested causal convolutions and attention |
| More depth did not always help | The deeper convolutional model improved only slightly and became unstable after its best epoch | Saved the best held-out checkpoint instead of the final epoch |
| Wider and deeper Transformers behaved differently | More parameters alone were not the explanation | The four-layer narrow model beat the larger two-layer wide model |
| Raw label changes greatly overcount boundaries | Good token accuracy did not automatically produce a usable halt signal | Kept boundary quality separate from token-classification quality |

## 12. What we can and cannot claim

**What holds up:** teacher-written responses can be processed entirely through causal prefill, their
internal states can be reduced to sparse features across ten layers, and a small causal Transformer
can recover token topics with 79.2% held-out accuracy and 75.0% macro F1. This is substantially
better than the recurrent baseline and does not require the base model to decode the training
responses.

**What remains unproven:** the current raw trajectory is not stable enough to treat every predicted
label change as a real drift boundary. The data is synthetic and deliberately clean, so performance
on naturally blended or ambiguous topic transitions may be lower. Forty-two generated records also
failed strict alignment and were excluded, showing that the data pipeline still needs repair.

## 13. Next steps

- **Fix the data.** Repair the 42 alignment failures, balance topic and topic-pair coverage, reduce
  ambiguity between enterprise documents and human resources, and audit boundary placement and
  teacher-style shortcuts.
- **Try two or three additional architectures.** Compare a convolution-plus-recurrent hybrid, a
  lightweight state-space sequence model, and a structured transition model using the same cached
  features and held-out split.

---

*Internal experiment report. All figures come from held-out response-level splits over teacher-
generated exchanges processed by a one-billion-parameter assistant in causal prefill mode. Figures
are rounded; exact run settings and logs are archived with the experiment.*
