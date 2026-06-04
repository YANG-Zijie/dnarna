"""
Generate an aggregated pair summary from an existing pair predictions file.

Examples:
  python -m dnarna.models.pair.predict.summarize \
    --predictions pair_predictions.csv

  python -m dnarna.models.pair.predict.summarize \
    --predictions pair_predictions.csv \
    --dna_seq_file processed/dna/dna.windowed.csv \
    --rna_seq_file processed/rna/rna.windowed.csv

  python -m dnarna.models.pair.predict.summarize \
    --predictions pair_predictions.csv \
    --pairs_file processed/pairs/pairs.windowed.csv \
    --output pair_predictions.summary.csv
"""

from __future__ import annotations

import argparse
import logging

from dnarna.models.pair.predict.aggregate import (
    summarize_pair_predictions_with_thresholds,
)

LOGGER = logging.getLogger("pair_summary")


def _parse_threshold_values(raw: str) -> list[float]:
    value = raw.strip()
    if not value:
        return []
    parts = [part.strip() for part in value.split(",")]
    return [float(part) for part in parts if part]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an aggregated summary from an existing pair predictions file."
        )
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Existing pair predictions file (csv/tsv/parquet).",
    )
    parser.add_argument(
        "--output",
        default="",
        help=(
            "Optional output path for the summary file. "
            "Default: <predictions>.summary.<ext>."
        ),
    )
    parser.add_argument(
        "--pairs_file",
        default="",
        help=(
            "Optional windowed pairs file to restore pair_parent_id and window metadata."
        ),
    )
    parser.add_argument(
        "--dna_seq_file",
        default="",
        help="Optional DNA windowed sequence file with id/parent_id/window metadata.",
    )
    parser.add_argument(
        "--rna_seq_file",
        default="",
        help="Optional RNA windowed sequence file with id/parent_id/window metadata.",
    )
    parser.add_argument("--pair_id_col", default="pair_id")
    parser.add_argument("--dna_id_col", default="dna_id")
    parser.add_argument("--rna_id_col", default="rna_id")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help=(
            "Threshold used for combined_pred and, when --ignore_pred_col is set, "
            "for positive_window_pair_count/fraction."
        ),
    )
    parser.add_argument(
        "--ignore_pred_col",
        action="store_true",
        help=(
            "Ignore existing pred column and recompute positive counts from prob >= threshold."
        ),
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help=(
            "Chunk size for streaming summary over CSV/TSV files. "
            "Use 0 to disable streaming and load the full file into memory."
        ),
    )
    parser.add_argument(
        "--thresholds",
        default="",
        help=(
            "Optional comma-separated thresholds such as 0.5,0.6,0.7. "
            "When set, an additional long-format summary file "
            "<summary>.by_threshold.<ext> is written."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    threshold_values = _parse_threshold_values(args.thresholds)
    (
        summary_df,
        output_path,
        threshold_df,
        threshold_output_path,
    ) = summarize_pair_predictions_with_thresholds(
        predictions_path=args.predictions,
        output_path=args.output or None,
        pairs_file=args.pairs_file or None,
        dna_seq_file=args.dna_seq_file or None,
        rna_seq_file=args.rna_seq_file or None,
        pair_id_col=args.pair_id_col,
        dna_id_col=args.dna_id_col,
        rna_id_col=args.rna_id_col,
        threshold=float(args.threshold),
        ignore_pred_col=bool(args.ignore_pred_col),
        chunksize=int(args.chunksize),
        thresholds=threshold_values,
    )
    LOGGER.info("Wrote %d summary rows to %s", len(summary_df), output_path)
    if threshold_df is not None and threshold_output_path is not None:
        LOGGER.info(
            "Wrote %d threshold summary rows to %s",
            len(threshold_df),
            threshold_output_path,
        )
    print(output_path)


if __name__ == "__main__":
    main()
