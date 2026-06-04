# DnaRna

DnaRna is an open-source model and command-line workflow for predicting potential DNA-RNA interactions. It standardizes input sequences, performs long-sequence windowing, extracts DNABERT-2/RNA-FM embeddings, and scores DNA x RNA pairs with a trained pair classifier.

## Repository Contents

- `src/dnarna/`: Python package for data processing, embedding, pair prediction, plotting, and job orchestration
- `assets/checkpoints/pair_model/model.pt`: trained pair model checkpoint included with this release
- `scripts/pipeline-template.sh`: command-line pipeline template for end-to-end prediction
- `docs/`: VitePress documentation site with English and Chinese content
- `tests/`: regression tests for data processing and prediction utilities

## Quick Start

```bash
uv pip install .
python -m dnarna.models.pair.predict.infer --help
```

For an end-to-end run, copy and edit the environment variables in [scripts/pipeline-template.sh](scripts/pipeline-template.sh), then run:

```bash
bash scripts/pipeline-template.sh
```

The template expects DNA/RNA input CSV files and writes windowed sequences, embeddings, raw pair predictions, and aggregated pair summaries.

## Included Model

The trained pair model used by this project is included at:

```text
assets/checkpoints/pair_model/model.pt
```

Only load model checkpoints and Hugging Face model code from trusted sources. DNABERT-2 uses Hugging Face remote model code, and local PyTorch checkpoints are deserialized during inference.

## Documentation

The documentation is managed by VitePress. To preview or build it locally:

```bash
pnpm install
pnpm docs:dev
pnpm docs:build
```

The generated static site can be deployed to GitHub Pages.

## Hugging Face Mirror

If you cannot access the original Hugging Face site, set these variables before running embedding or prediction commands:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=~/.cache/huggingface
```
