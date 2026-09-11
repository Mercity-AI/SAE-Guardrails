#!/bin/bash
# Static decode transfer for the GROUP-SPARSE (probe_group) selection arm, both backbones.
#
# Same footing as the F-statistic and summation decode runs: leakage-free test manifest, PR
# detectors, 8000-token EOS-natural cap. Only the cache and the checkpoint dir differ, and the
# cache is what fixes the backbone, SAE release, layer range and the per-layer feature indices,
# so the encode convention cannot drift from the one these detectors were trained on. These
# caches have VARIABLE layer widths (1B 1438, 4B 1489), which the decode path handles because it
# concatenates per-layer index arrays rather than assuming a fixed block size.
set -e
# Resolve the checkout from this script's own location, so the driver works from any working
# directory and on any machine. SCOPE_ROOT overrides it.
ROOT="${SCOPE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT"
: "${HF_TOKEN:=$(cat "${HF_HOME:-$HOME/.cache/huggingface}/token" 2>/dev/null || true)}"
export HF_TOKEN

echo "############ 1B STATIC PR group-sparse decode ############"
python -m important_scripts.decode.run_decode_static_pr --model-size 1b \
  --cache cache/sae1500_10k_1b_pr_probe_gsparse \
  --baselines results/gemma1b_sae150_10k_pr_probe_gsparse \
  --outdir ablation/gemma_arm/decode_1b_static_pr_probe_gsparse \
  --max-new-tokens 8000 "$@"

echo "############ 4B STATIC PR group-sparse decode ############"
python -m important_scripts.decode.run_decode_static_pr --model-size 4b \
  --cache cache/sae1500_10k_4b_pr_probe_gsparse \
  --baselines results/gemma4b_sae150_10k_pr_probe_gsparse \
  --outdir ablation/gemma_arm/decode_4b_static_pr_probe_gsparse \
  --max-new-tokens 8000 "$@"

echo "ALL GROUP-SPARSE DECODE PR DONE"
