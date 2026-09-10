#!/bin/bash
# Static SAE-150 decode ablation on the leakage-free test manifest, both backbones.
# The cache metadata fixes the backbone, SAE release and layer range, so --model-size is the
# only switch. --manifest / --outdir override the defaults (e.g. for the full-1500 manifest).
set -e
# Resolve the checkout from this script's own location, so the driver works from any working
# directory and on any machine. SCOPE_ROOT overrides it.
ROOT="${SCOPE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT"
: "${HF_TOKEN:=$(cat "${HF_HOME:-$HOME/.cache/huggingface}/token" 2>/dev/null || true)}"
export HF_TOKEN

for SIZE in 1b 4b; do
  echo "############ ${SIZE^^} STATIC PR decode ############"
  python -m important_scripts.decode.run_decode_static_pr --model-size "$SIZE" "$@"
done
echo "ALL STATIC DECODE PR DONE"
