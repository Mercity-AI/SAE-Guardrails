# SAE Guardrails

Topic classification guardrails built on sparse autoencoders (SAEs). Instead of inspecting raw text, this system classifies what topic an LLM is *responding about* by reading the SAE feature activations from the model's hidden states during generation.

A lightweight MLP trained on binary SAE firing patterns achieves **95%+ accuracy** across 7 domains (28 subtopics) using only 100 selected features from a single transformer layer — with zero access to the generated text itself.

## How It Works

1. **Data generation** — An LLM generates diverse user prompts across 28 subtopics, including adversarial prompts designed to camouflage the true topic.
2. **Hidden state capture** — A target model (e.g. Gemma 3 1B) generates responses to these prompts while we capture residual stream hidden states at every layer.
3. **SAE encoding** — Pre-trained SAEs (Gemma Scope JumpReLU or Qwen TopK) encode hidden states into sparse binary feature vectors.
4. **Feature discovery** — Cross-topic variance selects the single best layer and the top-100 most discriminative SAE features.
5. **Classifier training** — A two-layer MLP maps the 100-dimensional binary feature vector to topic logits.

## Repository Structure

```
SAE-Guardrails/
├── data/                    # Prompt generation pipeline
│   ├── generate.py          # Main generation script (OpenAI Responses API)
│   ├── prompts.py           # System/user prompt templates and adversarial levels
│   ├── utils.py             # Subtopic taxonomy (28 labels), sampling helpers
│   └── config.yaml          # Generation config (model, batch size, etc.)
├── training/                # SAE feature extraction and classifier training
│   ├── main.py              # End-to-end training pipeline
│   ├── models.py            # TopKSAE, JumpReLUSAE, TopicClassifier modules
│   ├── utils.py             # Data loading, splitting, filtering utilities
│   ├── post_experiments.py  # Per-step classification confidence analysis
│   ├── temporal_analysis.py # Early stopping simulation and confidence curves
│   ├── temporal_dynamics.py # Per-step calibrated classifiers with full SAE variance
│   ├── ablations.py         # Classifier and MLP hyperparameter ablations
│   └── mlp_ablations.py     # MLP architecture depth/width sweep
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

## Setup

**Requirements:** Python 3.10+, CUDA GPU recommended.

```bash
# Install PyTorch for your CUDA version first (see https://pytorch.org)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt

# Optional: flash attention for faster generation
pip install flash-attn --no-build-isolation
```

## Usage

### 1. Generate Topic Prompts

Generate the `prompts.jsonl` file that the training pipeline consumes. Requires an OpenAI API key.

```bash
# Set your API key
export OPENAI_API_KEY=sk-...

# Edit data/config.yaml to configure the run, then:
python data/generate.py

# Preview prompts without API calls:
# Set dry_run: true in data/config.yaml, then run the same command
```

The generator produces a `prompts.jsonl` file with one record per prompt, tagged with topic, subtopic, and adversarial level.

### 2. Train the SAE Classifier

Run the full pipeline: load model + SAEs, generate responses, extract features, train MLP.

```bash
# Default: Gemma 3 1B with Gemma Scope SAEs
python training/main.py

# Specify model and settings
python training/main.py --model-sizes gemma-3-1b-it --capture-mode decode

# Use custom data and checkpoint directory
python training/main.py --data-path path/to/prompts.jsonl --checkpoint-dir my_checkpoints

# Adjust training hyperparameters
python training/main.py --n-pos 1000 --mlp-epochs 50 --mlp-hidden 128
```

Run `python training/main.py --help` for all available options.

**Outputs** (saved to `--checkpoint-dir`):
- `*_mlp.pt` — trained classifier weights
- `*_meta.json` — full config, metrics, feature IDs, layer scores
- `*_train.parquet` / `*_test.parquet` — prompt metadata and generation stats
- `*_train_features.parquet` / `*_test_features.parquet` — binary feature vectors

### 3. Post-Training Analysis

```bash
# Per-step classification confidence analysis
python training/post_experiments.py
python training/post_experiments.py --run-prefix checkpoints/gemma-3-1b-it_20260618_112951

# Temporal dynamics with per-step calibrated classifiers
python training/temporal_dynamics.py --checkpoint-dir my_checkpoints

# Temporal analysis with early stopping simulation
python training/temporal_analysis.py --checkpoint checkpoints/gemma-3-4b-it_20260619_122931

# MLP architecture and hyperparameter ablations (reads pre-computed features)
python training/mlp_ablations.py
python training/ablations.py
```

## Supported Models

| Key | Model | SAE Source | SAE Type |
|-----|-------|-----------|----------|
| `gemma-3-1b-it` | Gemma 3 1B IT | Gemma Scope v2 | JumpReLU |
| `gemma-3-4b-it` | Gemma 3 4B IT | Gemma Scope v2 | JumpReLU |
| `gemma-2-9b-it` | Gemma 2 9B IT | Gemma Scope v1 | JumpReLU |
| `0.6B` | Qwen3 0.6B Base | Qwen SAE | TopK |
| `1.7B` | Qwen3 1.7B Base | Qwen SAE | TopK |
| `1.7B-Instruct` | Qwen3 1.7B IT | Qwen SAE | TopK |
| `8B` | Qwen3 8B Base | Qwen SAE | TopK |

## Topic Taxonomy

7 domains, 28 subtopics:

- **Enterprise documents** — internal records, operational procedures, project planning, commercial transactions
- **General news & content** — current affairs, science & tech, culture & sports, environment & society
- **Customer service** — billing disputes, account access, technical support, shipping & returns
- **Legal** — contract clauses, regulatory compliance, employment law, IP & licensing
- **Financial** — investor communications, personal banking, insurance & risk, tax & accounting
- **HR & people operations** — recruiting, onboarding, performance management, compensation & benefits
- **Healthcare** — clinical documentation, public health, medical research, mental health
