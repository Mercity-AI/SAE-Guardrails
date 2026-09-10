# Reproducibility configurations

- `data_generation_1500.yaml` records how the teacher-generated prompt/response dataset was produced.
- `feature_extraction.yaml` records the Gemma prefill and sparse-feature extraction setup.
- `model_training.yaml` is the historical GRU/TCN/Transformer configuration and
  describes the old 80/20 comparison runs. It is retained for provenance only.
- `tcn_baseline_corrected.yaml` is the corrected baseline TCN run.
- `tcn_variants_corrected.yaml` compares Gated, MultiScale, and ConvNeXt TCNs.
- `tcn_ablations_corrected.yaml` records every depth, width, and learning-rate
  ablation for those three variants.
- `tcn_prompt_response_corrected.yaml` is the prompt+response supervision run.
- `transformer_corrected.yaml` contains the corrected causal Transformer runs.

Corrected TCN configs use the same cached 500-feature token sequences and the
persisted deterministic 70/15/15 response-level split. SAE feature selection is
fit on training records only, checkpoints are selected on validation macro-F1,
and final metrics are reported once on the untouched test partition. Boundary
metrics use original token coordinates even when prompt/template tokens are masked.

Every current run receives its own result directory containing the resolved `config.yaml`, a
validation-selected best checkpoint, a final checkpoint, and complete metrics. Use `train_gru.py`,
`train_transformer.py`, or `train_tcn.py` as the public entry point; each accepts repeated
`--config` arguments and selects only its own architecture family from a mixed YAML file.
