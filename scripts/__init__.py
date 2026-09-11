"""Self-contained pipeline for token-level topic detection on model activations.

Everything paper 1 needs lives under this package: the detector library (`model`),
cache builders (`cache`), training entry points (`train`), decode transfer (`decode`)
and the knockout/attribution analyses (`analysis`). Nothing here imports code from
outside the package. Run a script as a module from the project root, e.g.
`python -m important_scripts.decode.run_decode_static_pr --model-size 4b`."""
