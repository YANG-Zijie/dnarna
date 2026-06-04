"""
Export top/bottom ranked DNA-RNA pair predictions to smaller CSV files.

Example:
  python -m dnarna.models.pair.predict.export_top_bottom_pairs \
    --predictions pair_predictions.csv \
    --top_n 200 \
    --bottom_n 200
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from dnarna.models.pair.predict.utils import (
    CSV_EXTENSIONS,
    PARQUET_EXTENSIONS,
    infer_extension,
)

LOGGER = logging.getLogger("pair_top_bottom")


def _iter_predictions(
    path: Path, *, chunksize: int | None
) -> Iterable[pd.DataFrame]:
    ext = infer_extension(path)
    if ext in PARQUET_EXTENSIONS:
        yield pd.read_parquet(path)
        return
    if ext in CSV_EXTENSIONS or not ext:
        sep = "\t" if ext.startswith(".tsv") else ","
        if chunksize is not None and int(chunksize) > 0:
            for chunk in pd.read_csv(path, sep=sep, chunksize=int(chunksize)):
                yield chunk
        else:
            yield pd.read_csv(path, sep=sep)
        return
    raise ValueError(
        f"Unsupported predictions file extension '{ext}' for '{path}'. "
        f"Supported: {sorted(PARQUET_EXTENSIONS | CSV_EXTENSIONS)}"
    )


def _strip_known_extension(path: Path) -> str:
    name = path.name
    ext = infer_extension(path)
    if ext and name.lower().endswith(ext):
        return name[: -len(ext)]
    return path.stem


def _write_csv(df: pd.DataFrame, output: Path) -> None:
    ext = output.suffix.lower()
    if ext not in {".csv", ".tsv"}:
        raise ValueError(
            f"Unsupported output extension '{ext}' for '{output}'. Use .csv or .tsv."
        )
    sep = "\t" if ext == ".tsv" else ","
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, sep=sep)


def export_top_bottom_pairs(
    *,
    predictions_path: str | Path,
    top_n: int = 100,
    bottom_n: int = 100,
    top_output: str | Path | None = None,
    bottom_output: str | Path | None = None,
    output_dir: str | Path | None = None,
    prob_col: str = "prob",
    chunksize: int | None = 200_000,
) -> tuple[Path | None, Path | None]:
    """
    Extract the top-N and bottom-N pairs by probability.

    Returns:
        (top_output_path, bottom_output_path) for the files written.
    """
    predictions_path = Path(predictions_path).expanduser()
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    top_n = int(top_n)
    bottom_n = int(bottom_n)
    if top_n <= 0 and bottom_n <= 0:
        raise ValueError("At least one of top_n or bottom_n must be > 0.")

    base_name = _strip_known_extension(predictions_path)
    target_dir = (
        Path(output_dir).expanduser()
        if output_dir
        else predictions_path.parent
    )

    if top_n > 0:
        top_output_path = (
            Path(top_output).expanduser()
            if top_output
            else target_dir / f"{base_name}.top{top_n}.csv"
        )
    else:
        if top_output:
            raise ValueError("top_output provided but top_n <= 0.")
        top_output_path = None

    if bottom_n > 0:
        bottom_output_path = (
            Path(bottom_output).expanduser()
            if bottom_output
            else target_dir / f"{base_name}.bottom{bottom_n}.csv"
        )
    else:
        if bottom_output:
            raise ValueError("bottom_output provided but bottom_n <= 0.")
        bottom_output_path = None

    top_df: pd.DataFrame | None = None
    bottom_df: pd.DataFrame | None = None
    total_rows = 0
    valid_rows = 0

    for chunk in _iter_predictions(predictions_path, chunksize=chunksize):
        total_rows += len(chunk)
        if prob_col not in chunk.columns:
            raise KeyError(
                f"Predictions file missing column '{prob_col}'. "
                f"Available: {list(chunk.columns)}"
            )
        probs = pd.to_numeric(chunk[prob_col], errors="coerce")
        mask = probs.notna()
        if not mask.any():
            continue
        chunk = chunk.loc[mask].copy()
        chunk[prob_col] = probs.loc[mask].astype(float)
        valid_rows += len(chunk)

        if top_n > 0:
            candidate = chunk.nlargest(top_n, prob_col)
            if top_df is None:
                top_df = candidate
            else:
                top_df = pd.concat([top_df, candidate], ignore_index=True).nlargest(
                    top_n, prob_col
                )

        if bottom_n > 0:
            candidate = chunk.nsmallest(bottom_n, prob_col)
            if bottom_df is None:
                bottom_df = candidate
            else:
                bottom_df = pd.concat(
                    [bottom_df, candidate], ignore_index=True
                ).nsmallest(bottom_n, prob_col)

    if top_n > 0:
        if top_df is None or top_df.empty:
            raise ValueError("No valid rows found for top-N selection.")
        _write_csv(top_df, top_output_path)
        LOGGER.info("Wrote top %d pairs to %s", len(top_df), top_output_path)

    if bottom_n > 0:
        if bottom_df is None or bottom_df.empty:
            raise ValueError("No valid rows found for bottom-N selection.")
        _write_csv(bottom_df, bottom_output_path)
        LOGGER.info(
            "Wrote bottom %d pairs to %s", len(bottom_df), bottom_output_path
        )

    LOGGER.info(
        "Scanned %d rows (%d with valid '%s').", total_rows, valid_rows, prob_col
    )
    return top_output_path, bottom_output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export top/bottom ranked pairs from pair_predictions.csv."
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Pair predictions file (csv/tsv/parquet).",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=100,
        help="Number of top-scoring pairs to export (default: 100).",
    )
    parser.add_argument(
        "--bottom_n",
        type=int,
        default=100,
        help="Number of lowest-scoring pairs to export (default: 100).",
    )
    parser.add_argument(
        "--top_output",
        default="",
        help="Output CSV for top-N pairs (default: <input>.topN.csv).",
    )
    parser.add_argument(
        "--bottom_output",
        default="",
        help="Output CSV for bottom-N pairs (default: <input>.bottomN.csv).",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        help="Output directory (default: predictions parent).",
    )
    parser.add_argument(
        "--prob_col",
        default="prob",
        help="Probability column name (default: prob).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="Chunk size for streaming CSV/TSV (0 = read all).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable info logging.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    export_top_bottom_pairs(
        predictions_path=args.predictions,
        top_n=args.top_n,
        bottom_n=args.bottom_n,
        top_output=args.top_output or None,
        bottom_output=args.bottom_output or None,
        output_dir=args.output_dir or None,
        prob_col=args.prob_col,
        chunksize=None if args.chunksize <= 0 else args.chunksize,
    )


if __name__ == "__main__":
    main()
