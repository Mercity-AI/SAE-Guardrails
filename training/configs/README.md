# Experiment configs

Static detectors (`train_static.py`), one config per backbone and selection rule; each trains the GRU,
the causal Transformer and the ConvNeXt from one cache load:

- `gemma1b_baselines_sae150_10k_pr.yaml`, `gemma4b_baselines_sae150_pr.yaml`: summation selection.
- `gemma1b_sae150_10k_pr_fclassif.yaml`, `gemma4b_sae150_10k_pr_fclassif.yaml`: F-statistic selection.
- `gemma1b_sae150_10k_pr_probe_gsparse.yaml`, `gemma4b_sae150_10k_pr_probe_gsparse.yaml`: group-lasso selection.

`dyn150_{gru,transformer,tcn}.yaml` record the dynamic detectors' settings; `train_dynamic.py` takes its
cache and output paths as `--cache` and `--out`.

All configs use prompt-and-response supervision, the persisted 70/15/15 split, and keep the epoch with
the best validation macro-F1. Each run writes its resolved config, best and final checkpoints and
metrics to its own results folder.
