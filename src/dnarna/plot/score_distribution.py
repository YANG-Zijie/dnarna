"""
Plot histogram distribution of prediction scores from inference outputs.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dnarna.models.shared.predict.infer import (
    CSV_EXTENSIONS,
    PARQUET_EXTENSIONS,
    infer_extension,
)

LOGGER = logging.getLogger("predict.plot")

__all__ = ["plot_score_distribution", "plot_score_distribution_from_predictions"]


def _read_predictions(path: Path) -> pd.DataFrame:
    ext = infer_extension(path)
    if ext in PARQUET_EXTENSIONS:
        return pd.read_parquet(path)
    if ext in {".json", ".jsonl"}:
        return pd.read_json(path, orient="records", lines=ext == ".jsonl")
    if ext in CSV_EXTENSIONS or not ext:
        sep = "\t" if ext.startswith(".tsv") else ","
        return pd.read_csv(path, sep=sep)
    raise ValueError(f"Unsupported predictions file extension '{ext}' for '{path}'.")


def plot_score_distribution(
    scores: Sequence[float],
    *,
    output_path: Path,
    threshold: float | None = None,
    bins: int = 50,
    xlim: tuple[float, float] | None = None,
    title: str = "Score Distribution",
    xlabel: str = "Predicted probability",
) -> Path | None:
    if scores is None:
        LOGGER.warning("No scores provided for score distribution plot.")
        return None
    arr = np.asarray(scores, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        LOGGER.warning("No finite scores available for score distribution plot.")
        return None

    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(arr, bins=int(bins), color="#4c78a8", alpha=0.85, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, linestyle="--", alpha=0.3)

    if xlim is not None:
        ax.set_xlim(*xlim)

    if threshold is not None:
        ax.axvline(
            float(threshold),
            color="#f58518",
            linestyle="--",
            linewidth=1.5,
            label=f"threshold={float(threshold):.3f}",
        )
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_score_distribution_from_predictions(
    predictions_path: str | Path,
    *,
    output_dir: Path | None = None,
    filename: str = "score_distribution.pdf",
    prob_col: str = "prob",
    threshold: float | None = None,
    bins: int = 50,
    xlim: tuple[float, float] | None = (0.0, 1.0),
) -> Path | None:
    predictions_path = Path(predictions_path).expanduser()
    df = _read_predictions(predictions_path)
    if prob_col not in df.columns:
        raise KeyError(
            f"Predictions file missing column '{prob_col}'. Available: {list(df.columns)}"
        )
    scores = pd.to_numeric(df[prob_col], errors="coerce")
    scores = scores.dropna().to_numpy(dtype=float)
    if scores.size == 0:
        LOGGER.warning("No valid scores found in %s.", predictions_path)
        return None

    target_dir = output_dir or predictions_path.parent
    return plot_score_distribution(
        scores,
        output_path=target_dir / filename,
        threshold=threshold,
        bins=bins,
        xlim=xlim,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot score distribution from a predictions file."
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Predictions file (csv/tsv/parquet/json/jsonl).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to write the figure to (defaults to predictions parent).",
    )
    parser.add_argument(
        "--filename",
        default="score_distribution.pdf",
        help="Output figure name.",
    )
    parser.add_argument(
        "--prob_col",
        default="prob",
        help="Probability column name in predictions (default: prob).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional threshold to draw as a vertical line.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins.",
    )
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir).expanduser() if args.output_dir else None
    plot_score_distribution_from_predictions(
        args.predictions,
        output_dir=out_dir,
        filename=args.filename,
        prob_col=args.prob_col,
        threshold=args.threshold,
        bins=max(1, int(args.bins)),
    )


if __name__ == "__main__":
    _main()
