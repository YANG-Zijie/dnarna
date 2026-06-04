#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DNA_INPUT="${DNA_INPUT:-/path/to/dna.csv}"
RNA_INPUT="${RNA_INPUT:-/path/to/rna.csv}"
WORK_DIR="${WORK_DIR:-/path/to/work_dir}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/output_dir}"

PAIR_CHECKPOINT="${PAIR_CHECKPOINT:-${REPO_ROOT}/assets/checkpoints/pair_model/model.pt}"

DNA_WINDOWED="${WORK_DIR}/dna.windowed.csv"
RNA_WINDOWED="${WORK_DIR}/rna.windowed.csv"
DNA_EMBEDDINGS="${WORK_DIR}/dna.windowed.embeddings.npz"
RNA_EMBEDDINGS="${WORK_DIR}/rna.windowed.embeddings.npz"

mkdir -p "${WORK_DIR}" "${OUTPUT_DIR}"

python -m dnarna.data.seq.window \
  --input_file "${DNA_INPUT}" \
  --output_dir "${WORK_DIR}" \
  --output_format csv \
  --window_size 1000 \
  --stride 500

python -m dnarna.models.dna.dnabert2.embed \
  --input_file "${DNA_WINDOWED}" \
  --output_dir "${WORK_DIR}" \
  --max_length 1000 \
  --batch_size 1024 \
  --device cuda:7

python -m dnarna.data.seq.window \
  --input_file "${RNA_INPUT}" \
  --output_dir "${WORK_DIR}" \
  --output_format csv \
  --window_size 1000 \
  --stride 500

python -m dnarna.models.rna.rnafm.embed \
  --input_file "${RNA_WINDOWED}" \
  --output_dir "${WORK_DIR}" \
  --max_length 1000 \
  --variant rna \
  --batch_size 256 \
  --device cuda:7

# Add --summary_only here if you only want pair_predictions.summary.csv and do not need raw pair_predictions.csv.
python -m dnarna.models.pair.predict.infer \
  --all_pairs \
  --dna_embeddings "${DNA_EMBEDDINGS}" \
  --rna_embeddings "${RNA_EMBEDDINGS}" \
  --dna_seq_file "${DNA_WINDOWED}" \
  --rna_seq_file "${RNA_WINDOWED}" \
  --checkpoint "${PAIR_CHECKPOINT}" \
  --output_dir "${OUTPUT_DIR}" \
  --feature_mode concat \
  --dna_block_size 64 \
  --rna_block_size 64 \
  --batch_size 4096 \
  --device cuda:7
