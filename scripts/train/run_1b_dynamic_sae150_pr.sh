#!/bin/bash
# 1B dynamic-150 prompt+response baselines (GRU/Transformer/ConvNeXt), test-split eval.
# Resolve the checkout from this script's own location, so the driver works from any working
# directory and on any machine. SCOPE_ROOT overrides it.
ROOT="${SCOPE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT"
: "${HF_TOKEN:=$(cat "${HF_HOME:-$HOME/.cache/huggingface}/token" 2>/dev/null || true)}"
export HF_TOKEN
export DYN_CACHE=cache/dyn150_sparse_10k_1b_pr
export DYN_OUT=results/dyn150_10k_1b_pr
python -m important_scripts.train.train_dynamic_gpu_lowmem   # DYN_ARCH unset -> all three sequentially
