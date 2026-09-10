#!/bin/bash
# Regenerate the PROMPT+RESPONSE caches from the base (response-only) caches.
#
# The *_pr caches are just a small labels.npy + symlinks to the base caches' heavy arrays
# (features / col / val / lengths / roles / split / selection). So they are NOT synced to the
# bucket -- they are rebuilt here, deterministically, from the base caches.
#
# Prerequisites (pull per SETUP.md first):
#   cache/sae1500_10k_1b_prompt_response/        (base 1B static SAE-1500)
#   cache/sae1500_10k_gemma4b_prompt_response/   (base 4B static SAE-1500)
#   cache/dyn150_sparse_10k_1b/                  (base 1B dynamic-150 sparse)
#   runs/GPT_5.6_10k_2108/dataset.final.jsonl    (the 10k tagged dataset)
set -e
# Resolve the checkout from this script's own location, so the driver works from any working
# directory and on any machine. SCOPE_ROOT overrides it.
ROOT="${SCOPE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT"
: "${HF_TOKEN:=$(cat "${HF_HOME:-$HOME/.cache/huggingface}/token" 2>/dev/null || true)}"
export HF_TOKEN

# 1B + 4B static: relabel = write PR labels.npy + symlink the base features (non-destructive).
python -m important_scripts.cache.relabel_1b_prompt_response    # -> cache/sae1500_10k_1b_pr
python -m important_scripts.cache.relabel_4b_pr_nondestructive  # -> cache/sae1500_10k_gemma4b_pr

# 1B dynamic: identical token order to 1B static (verified byte-equal lengths/roles), so reuse
# those PR labels as int64 (dynamic analysis indexes with them) + symlink the sparse base arrays.
PR=cache/dyn150_sparse_10k_1b_pr; SRC=cache/dyn150_sparse_10k_1b
mkdir -p "$PR"
for f in col.npy val.npy tok_ptr.npy lengths.npy roles.npy split_indices.npz sae_index.npy selected_features.npz meta.json; do
    ln -sf "$(pwd)/$SRC/$f" "$PR/$f"
done
python -c "import numpy as np; np.save('$PR/labels.npy', np.load('cache/sae1500_10k_1b_pr/labels.npy').astype(np.int64))"

# 4B dynamic: same reuse, from the 4B static PR labels (byte-identical token order, verified). Needed
# for the 4B dynamic knockout. Base cache dyn150_sparse_10k_4b comes from scope-caches (see SETUP.md).
PR=cache/dyn150_sparse_10k_4b_pr; SRC=cache/dyn150_sparse_10k_4b
mkdir -p "$PR"
for f in col.npy val.npy tok_ptr.npy lengths.npy roles.npy split_indices.npz sae_index.npy selected_features.npz meta.json; do
    ln -sf "$(pwd)/$SRC/$f" "$PR/$f"
done
python -c "import numpy as np; np.save('$PR/labels.npy', np.load('cache/sae1500_10k_gemma4b_pr/labels.npy').astype(np.int64))"

echo "PR caches regenerated: sae1500_10k_1b_pr, sae1500_10k_gemma4b_pr, dyn150_sparse_10k_1b_pr, dyn150_sparse_10k_4b_pr"
echo "(static/dynamic each: prompt 515,081 tokens supervised + 8.21M response tokens)"
