# DnaRna Documentation

DnaRna is an open-source model and command-line workflow for predicting potential DNA-RNA interactions.

## What Is Included

- Sequence input and cleaning utilities
- Long-sequence windowing for DNA/RNA candidates
- DNABERT-2 and RNA-FM embedding workflows
- Pair-level prediction and window-level aggregation
- A trained pair model checkpoint for public release

## Documentation Sections

- Data formats: sequence and DNA-RNA pair inputs
- Pair model: training, inference, and aggregated scoring
- Paper methods: model architecture, training strategy, and limitations

## Local Usage

```bash
uv pip install .
python -m dnarna.models.pair.predict.infer --help
```

For an end-to-end template, edit and run:

```bash
bash scripts/pipeline-template.sh
```
