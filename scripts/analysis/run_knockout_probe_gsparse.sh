#!/bin/bash
# Knockout + attribution for the GROUP-SPARSE (probe_group) selection arm, both backbones.
#
# Everything is named explicitly, as in run_knockout_pr.sh. The layer blocks come from the
# cache's layer_cols map because these caches keep only the features that survived the penalty,
# so layers differ in width (1B 1438 total, 4B 1489) and col//per_layer arithmetic would address
# the wrong block.
set -e
# Resolve the checkout from this script's own location, so the driver works from any working
# directory and on any machine. SCOPE_ROOT overrides it.
ROOT="${SCOPE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT"
: "${HF_TOKEN:=$(cat "${HF_HOME:-$HOME/.cache/huggingface}/token" 2>/dev/null || true)}"
export HF_TOKEN
FAMILIES="${FAMILIES:-GRU,Transformer,ConvNeXt}"

echo "############ 1B STATIC group-sparse PR knockout+attribution ############"
KNOCKOUT_SUPERVISE=pr \
KNOCKOUT_CACHE=cache/sae1500_10k_1b_pr_probe_gsparse \
KNOCKOUT_CKPT_DIR=results/gemma1b_sae150_10k_pr_probe_gsparse \
KNOCKOUT_OUT=results/analysis_sae150_1b10k_pr_probe_gsparse \
python -m important_scripts.analysis.exp1_layer_feature_attrib --families "$FAMILIES"

echo "############ 4B STATIC group-sparse PR knockout+attribution ############"
KNOCKOUT_SUPERVISE=pr \
KNOCKOUT_CACHE=cache/sae1500_10k_4b_pr_probe_gsparse \
KNOCKOUT_CKPT_DIR=results/gemma4b_sae150_10k_pr_probe_gsparse \
KNOCKOUT_OUT=results/analysis_sae150_4b10k_pr_probe_gsparse \
python -m important_scripts.analysis.exp1_layer_feature_attrib --families "$FAMILIES"

echo "ALL GROUP-SPARSE KNOCKOUT PR DONE"
