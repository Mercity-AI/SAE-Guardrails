# Dataset

The 10,000-record topic-shift dataset ([`Mercity/topic-shift-10k`](https://huggingface.co/datasets/Mercity/topic-shift-10k)), generated with [Simula](https://github.com/Mercity-AI/Simula).

Every script downstream reads `runs/GPT_5.6_10k_2108/dataset.final.jsonl`. You can either download it or generate it again.

## Use the published data

```bash
python dataset/download.py
```

This joins the three Hub splits, restores the original record order and writes `dataset.final.jsonl`. Record order matters, because the canonical split is a permutation of line numbers. The script checks that the published splits match `split_indices` before it writes anything.

## Generate from scratch

| File | Role |
|---|---|
| `build_taxonomy.py` | Writes the hand-authored `taxonomy.json` and `strategies.json` (1–4 topics in a fixed order, every ordered pair reachable), then audits 40k sampled mixes. The audit is free and makes no API calls. |
| `topic_shift_10k.yaml` | The Simula run config: 10k records, GPT-5.6 Luna as generator and critic, OpenRouter. |
| `simula_prompts.py` | The generation and critique prompts. The YAML loads this file by its name, so keep the two files together. |

**1. Install Simula** (Python ≥ 3.11), in the same environment as this repo:

```bash
git clone https://github.com/Mercity-AI/Simula.git
pip install -e ./Simula
```

**2. Add your API key.** Simula reads keys only from a `.env` file in the working directory. A shell variable is ignored. `.env` is gitignored here.

```bash
echo 'OPENROUTER_API_KEY=sk-or-...' > .env
```

**3. Run from the repository root**, since `output_dir` in the YAML is relative:

```bash
python dataset/build_taxonomy.py                                # writes the taxonomy into runs/GPT_5.6_10k_2108/ and audits it
python -m simula.cli validate dataset/topic_shift_10k.yaml      # checks the config, makes no model calls
python -m simula.cli run dataset/topic_shift_10k.yaml           # paid; resumable, checkpoints every 25 rows
```

Run the taxonomy builder before Simula. If `taxonomy.json` or `strategies.json` is missing, Simula builds its own taxonomy with a model, and the topic balance is lost. The builder reproduces the published files exactly.

A new run writes different text, so its `dataset.final.jsonl` won't match the published one record for record. The split, labels and every later step work on it unchanged.
