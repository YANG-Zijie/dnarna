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

## Optional Web UI

DnaRna also includes an optional Streamlit interface for interactive prediction jobs:

The Streamlit UI is an interactive wrapper around the same command-line prediction pipeline. It is suitable for small to medium trial runs. For large-scale jobs, server-side execution, or fine-grained parameter control, use the CLI pipeline directly.

```bash
uv pip install .
uv run streamlit run src/dnarna/app/streamlit_app.py
```

The default local URL is:

```text
http://localhost:8501
```

Keep the Streamlit process running while a job is active. Stopping the process will stop the job.
