#!/bin/bash
# Stage 4: knockout + attribution on the PROMPT+RESPONSE checkpoints, both backbones.
#
# One canonical script per arm. Everything it needs is named explicitly, because nothing has a
# default any more:
#   KNOCKOUT_CACHE      feature cache; its metadata.json fixes the backbone, layers and width
#   KNOCKOUT_CKPT_DIR   detector checkpoints (KNOCKOUT_CKPT_<FAMILY> overrides one of them)
#   KNOCKOUT_OUT        output directory
#   KNOCKOUT_SUPERVISE  pr, or prompt spans are masked and the scores stop matching the baselines
# Pass --families to knock out a subset, e.g. --families ConvNeXt,Transformer. Results merge
# into any existing json, so a later GRU-only run adds to an earlier Transformer+ConvNeXt run.
set -e
# Resolve the checkout from this script's own location, so the driver works from any working
# directory and on any machine. SCOPE_ROOT overrides it.
ROOT="${SCOPE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT"
: "${HF_TOKEN:=$(cat "${HF_HOME:-$HOME/.cache/huggingface}/token" 2>/dev/null || true)}"
export HF_TOKEN
FAMILIES="${FAMILIES:-GRU,Transformer,ConvNeXt}"

echo "############ 1B STATIC PR knockout+attribution ############"
KNOCKOUT_SUPERVISE=pr \
KNOCKOUT_CACHE=cache/sae1500_10k_1b_pr \
KNOCKOUT_CKPT_DIR=results/gemma1b_baselines_sae150_10k_pr \
KNOCKOUT_OUT=results/analysis_sae150_1b10k_pr \
python -m important_scripts.analysis.exp1_layer_feature_attrib --families "$FAMILIES"

echo "############ 4B STATIC PR knockout+attribution ############"
KNOCKOUT_SUPERVISE=pr \
KNOCKOUT_CACHE=cache/sae1500_10k_gemma4b_pr \
KNOCKOUT_CKPT_DIR=results/gemma4b_baselines_sae150_pr \
KNOCKOUT_OUT=results/analysis_sae150_4b10k_pr \
python -m important_scripts.analysis.exp1_layer_feature_attrib --families "$FAMILIES"

echo "############ 1B DYNAMIC PR knockout+attribution ############"
KNOCKOUT_SUPERVISE=pr \
DYN_CACHE=cache/dyn150_sparse_10k_1b_pr \
DYN_OUT=results/dyn150_10k_1b_pr \
DYN_ANALYSIS_OUT=results/analysis_dyn150_1b10k_pr \
DYN_ANALYSIS_FAMILIES="$FAMILIES" \
python -m important_scripts.analysis.exp1_layer_feature_attrib_dynamic

echo "ALL KNOCKOUT PR DONE"
