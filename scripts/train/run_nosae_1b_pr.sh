#!/bin/bash
# Resolve the checkout from this script's own location, so the driver works from any working
# directory and on any machine. SCOPE_ROOT overrides it.
ROOT="${SCOPE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT"
: "${HF_TOKEN:=$(cat "${HF_HOME:-$HOME/.cache/huggingface}/token" 2>/dev/null || true)}"
export HF_TOKEN
python -m important_scripts.train.train_nosae_stream --model google/gemma-3-1b-it --layers 16-25 \
    --sel150 cache/sae1500_10k_1b_pr \
    --dataset runs/GPT_5.6_10k_2108/dataset.final.jsonl \
    --split  cache/sae1500_10k_1b_pr \
    --pr-labels cache/sae1500_10k_1b_pr \
    --outdir ablation/nosae_stream_1b_10k_pr --arms raw --pack-seqs 64
echo "NOSAE 1B PR DONE"
