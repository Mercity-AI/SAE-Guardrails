"""Optional Weights & Biases logging shared by the three trainers.

Logging is switched on by ``SCOPE_WANDB_PROJECT`` and draws no random numbers, so enabling it leaves a
run bit-for-bit unchanged.
"""

from __future__ import annotations

import os


def flatten_metrics(values, prefix: str) -> dict:
    """Flatten a nested metrics dict into ``prefix/a/b`` scalar keys for W&B."""
    flat = {}
    for key, value in values.items():
        name = f"{prefix}/{key}"
        if isinstance(value, dict):
            flat.update(flatten_metrics(value, name))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[name] = value
    return flat


def start_wandb(name: str, group: str, config: dict):
    """Open a W&B run when SCOPE_WANDB_PROJECT is set; otherwise return None.

    Logging draws no random numbers, so enabling it leaves training bit-for-bit unchanged.
    """
    project = os.environ.get("SCOPE_WANDB_PROJECT")
    if not project:
        return None
    import wandb

    return wandb.init(project=project, name=name, group=group, config=config, reinit="create_new")
