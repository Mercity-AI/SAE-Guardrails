# Dataset generation

This folder contains the maintained topic-axis dataset-generation files:

- `prefill_mvp_topic_axis.yaml` defines the dataset, generation providers, and validation rules.
- `prefill_mvp_topic_axis_prompts.py` builds the generation and critique prompts.
- `build_topic_axis_taxonomy.py` writes and audits the balanced taxonomy and sampling strategy.

The YAML refers to the prompt module by filename, so the two files must remain together. Run the
taxonomy builder before a paid generation run so the balanced hand-authored taxonomy is present.
