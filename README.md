# SAE Guardrails

Accompanying repo for the post: <placeholder>

Online topic guardrails for LLMs. A small causal detector reads the model's activations and labels the topic of every token as it is generated. Because the label is per token, it can switch partway through a response, at the point where the subject changes. We train these detectors on Gemma 3 1B and 4B and compare three kinds of input: static and dynamic [Gemma Scope 2](https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior) SAE features, and raw hidden states with no SAE.

| Detector input | Input width | Best macro-F1, 1B | Best macro-F1, 4B |
|---|---|---|---|
| Static SAE, 150 features per layer, chosen by summed activation | 1,500 | 96.2 | 92.4 |
| Static SAE, features chosen with the labels (F-statistic, group lasso) | ~1,500 | 96.8 | 96.4 |
| Dynamic SAE, each token's own top 150 features per layer | ~54,000 | 97.3 | 97.1 |
| Raw hidden states, no SAE | 11,520 / 25,600 | 96.7 | 97.5 |

These scores are on the 1,500-record test split: seven business topics, with prompt and response tokens scored together.

**Released** (collection: [Online Guardrails using Sparse Autoencoders](https://huggingface.co/collections/Mercity/online-guardrails-using-sparse-autoencoders-6ac602940468c62fefb2b9e8)):

- the dataset [`Mercity/topic-shift-10k`](https://huggingface.co/datasets/Mercity/topic-shift-10k);
- the best 1B and 4B detector for each input: [`sae-summation`](https://huggingface.co/Mercity/online-guardrails-sae-summation), [`sae-fstatistic`](https://huggingface.co/Mercity/online-guardrails-sae-fstatistic), [`sae-grouplasso`](https://huggingface.co/Mercity/online-guardrails-sae-grouplasso), [`sae-dynamic`](https://huggingface.co/Mercity/online-guardrails-sae-dynamic) and [`raw-hidden`](https://huggingface.co/Mercity/online-guardrails-raw-hidden).

## Requirements

- **Python** 3.10 or newer. The original runs used Python 3.12, PyTorch 2.8 and CUDA 12.8.
- **A CUDA GPU** for anything that runs Gemma: the quickstart, the cache builders, raw training, decode and the perturbation scoring. The 4B caches and the raw-hidden-state trainer ran on 80 GB cards.
- **Gemma 3 access** on the Hugging Face Hub. The model is gated, so accept its license first.
- **Disk**, only for a full reproduction: the feature caches come to about 200 GB, 25–50 GB each.
- **An API key** for an OpenAI-compatible endpoint, only to generate new perturbations or new data.

## Setup

```bash
pip install -r requirements.txt  # optional: flash-attn (FlashAttention-2), wandb (training logs)
hf auth login                    # Gemma 3 is gated; accept its license on the Hub first
```

The Gemma Scope SAEs download through `sae_lens` the first time they're needed.

## Quickstart: score the published detectors

If you only want to evaluate the released checkpoints, this one script is all you need:

```bash
python run.py --model-size 1b
python run.py --model-size 4b --arms dynamic,raw --limit 50
```

It downloads the test split and the detectors from the Hub and rebuilds each detector's input features from one teacher-forced pass. It then prints test macro-F1 and overlap next to the numbers recorded at training time. It needs no local cache.

## Layout

```
run.py                   quick eval of the Hub checkpoints
models.py                the three detectors (GRU, causal Transformer, causal ConvNeXt)
common/                  data loading and split, labels, metrics, decoding, paths, logging
dataset/                 download the dataset, or regenerate it with Simula (see its README)
preprocessing/           feature caches: static SAE (3 selection rules), dynamic SAE, prompt+response labels
training/                static, dynamic and raw trainers; configs/ holds the experiment YAMLs
evals/
  decode/                decode transfer: Gemma generates, the detectors label its tokens
  knockout/              layer knockout and feature attribution
  perturbations/         off-topic insertion, removal, reordering, substitution
```

Every script runs from the repository root as `python <folder>/<script>.py`, and `--help` lists its arguments. Data folders (`runs/`, `cache/`, `results/`, `ablation/`) aren't in the repo. They're resolved against `SCOPE_ROOT` if that's set, otherwise against the repo root.

## Reproducing the experiments

Run the stages **in order**. Each one reads what the previous one wrote, and nothing is downloaded for you between them:

1. **Data** writes `runs/GPT_5.6_10k_2108/dataset.final.jsonl`.
2. **Caches** need the data. They write the base feature caches, and then the `_pr` caches that every reported run trains on. Each cache is 25–50 GB.
3. **Training** needs the `_pr` caches. The trainers stop if a cache is missing; they don't build it. The raw-hidden-state trainer is the one exception: it streams from the model and only needs the data.
4. **Evals** (decode, knockout, perturbations) need the trained checkpoints, and knockout also needs the caches.

The commands below show 1B. For 4B:
- give the static builder `--model google/gemma-3-4b-it --layers 24-33 --sae-release gemma-scope-2-4b-it-res-all`;
- use `build_dynamic_cache_4b` for the dynamic cache;
- use the `gemma4b_*` configs for static training;
- pass `--model-size 4b` to the evals.

### 1. Data

```bash
python dataset/download.py
```

To generate the data from scratch instead, see [`dataset/README.md`](dataset/README.md).

### 2. Feature caches

Build the base caches, then derive the `_pr` caches from them. The two labelling commands have to run after the builders, static before dynamic. The `_pr` folders hold only new labels and links to the base arrays, so they take almost no space.

```bash
python preprocessing/build_static_cache.py --model google/gemma-3-1b-it --layers 16-25 \
    --sae-release gemma-scope-2-1b-it-res-all --output cache/sae1500_10k_1b_prompt_response
python preprocessing/build_dynamic_cache_1b.py
python preprocessing/add_prompt_labels.py --model-size 1b          # -> cache/sae1500_10k_1b_pr
python preprocessing/add_dynamic_prompt_labels.py --model-size 1b  # -> cache/dyn150_sparse_10k_1b_pr
```

For the label-based feature sets, add `--select f_classif` (F-statistic) or `--select probe_group` (group lasso) to the static builder.

### 3. Training

Start only once step 2 has finished.

```bash
python training/train_static.py training/configs/gemma1b_baselines_sae150_10k_pr.yaml
python training/train_dynamic.py --cache cache/dyn150_sparse_10k_1b_pr --out results/dyn150_10k_1b_pr
python training/train_raw.py --model google/gemma-3-1b-it --layers 16-25 \
    --dataset runs/GPT_5.6_10k_2108/dataset.final.jsonl --split cache/sae1500_10k_1b_pr \
    --pr-labels cache/sae1500_10k_1b_pr --sel150 cache/sae1500_10k_1b_pr \
    --outdir ablation/nosae_stream_1b_10k_pr --arms raw --pack-seqs 64
```

`train_static.py` trains every detector its config names, and the other configs cover the F-statistic, group-lasso and 4B runs. `train_dynamic.py` trains the three detectors one after another, or one with `--arch gru`. Set `SCOPE_WANDB_PROJECT` to log training to W&B.

### 4. Evals

**Decode transfer.** Gemma writes its own responses to 1,000 test prompts, and the detectors label them. `--teacher` runs the same detectors on GPT's reference responses instead, which is the comparison point.

```bash
python evals/decode/build_test_manifest.py
python evals/decode/decode.py --model-size 1b --features static            # or dynamic, raw
python evals/decode/decode.py --model-size 1b --features static --teacher
python evals/decode/score_decode.py --out results/decode_1b.json \
    --run static=ablation/decode/1b_static_decode/per_record.jsonl \
    --run static_gpt=ablation/decode/1b_static_teacher/per_record.jsonl
```

For the F-statistic or group-lasso detectors, pass their cache and checkpoints, for example `--cache cache/sae1500_10k_1b_pr_probe_gsparse --ckpt-dir results/gemma1b_sae150_10k_pr_probe_gsparse`.

**Knockout and attribution.** These also need the step-2 caches. `--supervise pr` scores prompt and response tokens, as training did; without it, the scripts refuse to run on a prompt+response cache.

```bash
python evals/knockout/knockout_static.py --supervise pr --cache cache/sae1500_10k_1b_pr \
    --ckpt-dir results/gemma1b_baselines_sae150_10k_pr --out results/analysis_sae150_1b10k_pr
python evals/knockout/knockout_dynamic.py --supervise pr --cache cache/dyn150_sparse_10k_1b_pr \
    --ckpt-dir results/dyn150_10k_1b_pr --out results/analysis_dyn150_1b10k_pr
```

For 4B, use `cache/sae1500_10k_gemma4b_pr` and `results/gemma4b_baselines_sae150_pr`. For group lasso, use the `_probe_gsparse` cache and results folders. `--families ConvNeXt,Transformer` runs only some detectors, and later runs merge into the same output file.

**Perturbations.**

```bash
python evals/perturbations/insert_offtopic.py                                    # LLM inserts off-topic asides
python evals/perturbations/build_perturbations.py --kind removal                 # also reordering, substitution
python evals/perturbations/score_perturbations.py --kind insertion --model-size 1b
python evals/perturbations/score_perturbations.py --kind removal --model-size 1b
```

`score_perturbations.py` scores all four feature inputs by default; `--arm dynamic` picks one. `--kind confidence` scores the inserted asides for abstention: how confident the detectors are on them, compared with real topic tokens.

> [!NOTE]
> The perturbations start from the 1,500 test records in `evals/perturbations/manifest_test_full1500.jsonl`. `insert_offtopic` has an LLM add off-topic asides to each record, and the removal, reordering and substitution sets are built from its output.
>
> To try your own perturbations:
> - choose the model with `--model`, `--base-url` and `--api-key-env` (any OpenAI-compatible endpoint);
> - edit the instructions in `injection_prompts.py`;
> - check the prompts with `--dry-run`, then try a few records with `--limit 5`.

## Conventions

- **Split:** record-level 70/15/15 from `numpy.random.RandomState(42)`. Every feature-selection rule is fitted on the training split only.
- **Supervision:** topic tokens in both the prompt and the response. Template tokens and the text between spans are unlabelled.
- **Checkpoints:** we keep the epoch with the best validation macro-F1 and report every number on the test split.
- **Scoring:** each token gets its argmax label, with no smoothing. Macro-F1 and overlap (per-topic IoU) are pooled over all labelled test tokens.
