#!/bin/bash
# Resolve the checkout from this script's own location, so the driver works from any working
# directory and on any machine. SCOPE_ROOT overrides it.
ROOT="${SCOPE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT"
: "${HF_TOKEN:=$(cat "${HF_HOME:-$HOME/.cache/huggingface}/token" 2>/dev/null || true)}"
export HF_TOKEN
KNOCKOUT_SUPERVISE=pr \
DYN_CACHE=cache/dyn150_sparse_10k_1b_pr \
DYN_OUT=results/dyn150_10k_1b_pr \
DYN_ANALYSIS_OUT=results/analysis_dyn150_1b10k_pr \
DYN_ANALYSIS_FAMILIES=Transformer,ConvNeXt \
python -m important_scripts.analysis.exp1_layer_feature_attrib_dynamic
echo "DYN KNOCKOUT (Tf,Cx) DONE"
