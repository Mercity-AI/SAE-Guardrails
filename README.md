# TopicScope

Code, data and detectors for [**Train on prefill, monitor at decode: Using sparse autoencoders for topic monitoring**](https://arxiv.org/abs/2611.12935).

<p align="center">
  <a href="https://arxiv.org/abs/2611.12935">📄 Paper</a> -
  <a href="https://www.mercity.ai/research/online-topic-guardrails-with-saes/">📝 Blog post</a> -
  <a href="https://huggingface.co/collections/Mercity/online-guardrails-using-sparse-autoencoders-6ac602940468c62fefb2b9e8">🤗 Hugging Face collection</a>
</p>

Online topic guardrails for LLMs. A small causal detector reads the model's activations and labels the topic of every token as it is generated. Because the label is per token, it can switch partway through a response, at the point where the subject changes. We train these detectors on Gemma 3 1B and 4B and compare three kinds of input: static and dynamic [Gemma Scope 2](https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior) SAE features, and raw hidden states with no SAE.

| Detector input | Input width | Best macro-F1, 1B | Best macro-F1, 4B |
|---|---|---|---|
| Static SAE, 150 features per layer, chosen by summed activation | 1,500 | 96.2 | 92.4 |
| Static SAE, features chosen with the labels (F-statistic, group lasso) | ~1,500 | 96.8 | 96.4 |
| Dynamic SAE, each token's own top 150 features per layer | ~54,000 | 97.3 | 97.1 |
| Raw hidden states, no SAE | 11,520 / 25,600 | 96.7 | 97.5 |

These scores are on the 1,500-record test split: seven business topics, with prompt and response tokens scored together.

## Requirements

| | Needed for |
|---|---|
| **Python 3.10+** (we used 3.12, PyTorch 2.8, CUDA 12.8) | everything |
| **A CUDA GPU**, an 80 GB A100 recommended to avoid running out of memory | anything that runs Gemma: the quickstart, cache building, raw-state training, decode and perturbation scoring |
| **Access to [Gemma 3](https://huggingface.co/google/gemma-3-1b-it)** on the Hugging Face Hub | everything that runs Gemma. The model is gated: accept its license, then log in with `hf auth login`. |
| **~200 GB of disk** | a full reproduction only (the feature caches are 25–50 GB each) |
| **An OpenAI-compatible API key** | generating new data or new perturbations only |

## Setup

```bash
git clone https://github.com/Mercity-AI/SAE-Guardrails.git
cd SAE-Guardrails
pip install -r requirements.txt
```

Optional: `flash-attn` (FlashAttention-2 for the cache builders and raw-state trainer) and `wandb` (training logs).

## Quickstart: score the published detectors

```bash
python run.py --model-size 1b                                # every detector on Gemma 3 1B
python run.py --model-size 4b --arms dynamic,raw --limit 50  # a quick check on 50 records
```

You don't need any local data or caches. `run.py` downloads everything from the Hugging Face Hub:

- the 1,500 test records from [`Mercity/topic-shift-10k`](https://huggingface.co/datasets/Mercity/topic-shift-10k);
- each detector's checkpoint and its selected SAE features from the [collection](https://huggingface.co/collections/Mercity/online-guardrails-using-sparse-autoencoders-6ac602940468c62fefb2b9e8);
- Gemma 3 and its Gemma Scope 2 SAEs.

It runs each record through Gemma once, rebuilds the features every detector reads, and prints test macro-F1 and overlap next to the scores recorded at training time. The results are also saved to `results/hf_detector_eval.json`. `--arms` takes any of `summation,fstatistic,grouplasso,dynamic,raw`.

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

## Citation

```bibtex
@misc{mallagundla2026topicscope,
  title         = {Train on prefill, monitor at decode: Using sparse autoencoders for topic monitoring},
  author        = {Mallagundla, Rishikesh and Patel, Pranav},
  year          = {2026},
  eprint        = {2611.12935},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2611.12935}
}
```
