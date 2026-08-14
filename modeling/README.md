# Model training layout

This folder contains the maintained model-training code.

## Main files

- `models.py` defines every causal GRU, Transformer, and temporal-convolution architecture,
  including the joint topic-and-boundary versions.
- `train_gru.py` is the only GRU training entry point.
- `train_transformer.py` is the only Transformer training entry point.
- `train_tcn.py` is the only temporal-convolution training entry point. It supports the plain,
  gated, multi-scale, and ConvNeXt variants.
- `utils.py` provides cache validation/loading, persisted split loading, prediction, and metrics.
- `decoder_utils.py` contains the shared causal sequence decoders and the explicitly offline-only
  sticky-Viterbi decoder.
- `load_checkpoints.py` restores existing ordinary and joint checkpoints without changing their
  saved format.
- `configs/` contains the existing experiment configurations. Every completed run also saves its
  resolved configuration beside its best and final checkpoints.

`training.py` and `training_joint.py` are internal shared engines, not command-line entry points.
Keeping them shared prevents the three public training scripts from duplicating checkpoint and
metric logic.

## Running experiments

Run an architecture entry point with one or more configuration files:

```bash
python modeling/train_gru.py --config modeling/configs/gru_2k.yaml
python modeling/train_transformer.py --config modeling/configs/transformer_2k.yaml
python modeling/train_tcn.py --config modeling/configs/tcn_2k_clean.yaml
```

Mixed configuration files are safe: each entry point runs only its own architecture family. Before
training, the script validates the packed feature cache and prints that the existing cache is being
used. Training retains the validation-selected best checkpoint and the final checkpoint. Final
console output includes topic F1, Dice, and intersection over union; full metrics remain saved.
Decoder evaluation uses the experiment's exact training seed (42 in the maintained configurations)
and records that seed in the saved results.

The default remains raw per-token decoding, which exactly preserves previous behavior. A causal
decoder can be selected under `evaluation.decoder`, for example
`{method: minimum_duration, duration: 3}`. Available causal methods are `raw`,
`minimum_duration`, `hysteresis`, and `transition_penalty`. Viterbi is available only for offline
analysis and is rejected by the causal training/evaluation path.

## Loading a checkpoint

```bash
python modeling/load_checkpoints.py path/to/checkpoint_best.pt
```

The loader supports all ordinary and joint architectures and preserves the current checkpoint
contract.
