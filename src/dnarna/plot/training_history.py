"""
Visualization helpers for classifier training metrics (RNA-FM, DNABERT-2, etc.).

The plotting utilities can be reused to generate figures for historical runs by
loading metrics history from ``metrics.json`` or ``metrics.csv`` files that were
produced during training.

Example CLI usage:
    python -m dnarna.plot.training_history \
        --metrics_json outputs/training_run/metrics.json \
        --output_dir figures \
        --filename training_metrics.pdf
"""

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt

LOGGER = logging.getLogger("training.plot")


def plot_training_history(
    history: Sequence[Mapping[str, float | int]],
    *,
    output_dir: Path,
    filename: str = "metrics.pdf",
) -> Path | None:
    """
    Plot training/validation curves (loss, accuracy, precision, recall, F1) from metrics history.

    Args:
        history: Iterable of per-epoch metrics dictionaries.
        output_dir: Directory where the figure should be written.
        filename: Name of the output figure (default: ``metrics.pdf``).

    Returns:
        The path to the saved plot, or ``None`` if the plot could not be created.
    """
    if not history:
        LOGGER.warning("No history available to plot metrics.")
        return None

    epochs = [int(entry.get("epoch", idx + 1)) for idx, entry in enumerate(history)]
    train_loss = [float(entry.get("train_loss", math.nan)) for entry in history]
    train_f1 = [float(entry.get("train_f1", math.nan)) for entry in history]

    val_loss = [
        float(entry["val_loss"]) if "val_loss" in entry else math.nan
        for entry in history
    ]
    val_f1 = [
        float(entry["val_f1"]) if "val_f1" in entry else math.nan for entry in history
    ]

    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / filename

    train_acc = [float(entry.get("train_acc", math.nan)) for entry in history]
    val_acc = [
        float(entry["val_acc"]) if "val_acc" in entry else math.nan for entry in history
    ]
    train_precision = [
        float(entry.get("train_precision", math.nan)) for entry in history
    ]
    val_precision = [
        float(entry["val_precision"]) if "val_precision" in entry else math.nan
        for entry in history
    ]
    train_recall = [float(entry.get("train_recall", math.nan)) for entry in history]
    val_recall = [
        float(entry["val_recall"]) if "val_recall" in entry else math.nan
        for entry in history
    ]

    metric_plots = [
        ("Loss", train_loss, val_loss, "Loss"),
        ("Accuracy", train_acc, val_acc, "Accuracy"),
        ("Precision", train_precision, val_precision, "Precision"),
        ("Recall", train_recall, val_recall, "Recall"),
        ("F1 Score", train_f1, val_f1, "F1 score"),
    ]

    n_cols = 2
    total_panels = len(metric_plots)

    n_rows = math.ceil(total_panels / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4.5 * n_rows), sharex=True)
    axes = axes.flatten()

    for idx, (title, train_series, val_series, ylabel) in enumerate(metric_plots):
        ax = axes[idx]
        plotted = False
        if any(not math.isnan(value) for value in train_series):
            ax.plot(epochs, train_series, label="train", marker="o")
            plotted = True
        if any(not math.isnan(value) for value in val_series):
            ax.plot(epochs, val_series, label="val", marker="s")
            plotted = True

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Epoch")
        if plotted:
            ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)

    # Hide any unused axes (in case total_panels < len(axes))
    for extra_ax in axes[total_panels:]:
        extra_ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Training curves saved to %s", plot_path)

    return plot_path


def plot_from_metrics_json(
    metrics_json_path: str | Path,
    *,
    output_dir: Path | None = None,
    filename: str | None = None,
) -> Path | None:
    """
    Convenience helper to rebuild plots from an existing ``metrics.json`` file.

    Args:
        metrics_json_path: Path to the metrics JSON generated during training.
        output_dir: Directory to write the figure to (defaults to the JSON parent).
        filename: Name of the figure file (defaults to ``metrics_plot`` entry or
            ``metrics.pdf``).

    Returns:
        The path to the saved plot, or ``None`` if plotting was skipped.
    """
    metrics_json = Path(metrics_json_path).expanduser()
    if not metrics_json.exists():
        raise FileNotFoundError(f"metrics.json not found: {metrics_json}")

    with metrics_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    history = data.get("history") or []
    if not history:
        LOGGER.warning("metrics.json at %s does not contain history.", metrics_json)
        return None

    target_dir = output_dir or metrics_json.parent
    target_name = filename or data.get("metrics_plot", "metrics.pdf")

    return plot_training_history(
        history,
        output_dir=target_dir,
        filename=target_name,
    )


__all__ = ["plot_training_history", "plot_from_metrics_json"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate training curves from an existing metrics.json file."
    )
    parser.add_argument(
        "--metrics_json",
        required=True,
        help="Path to a metrics.json file emitted by the training script.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to write the figure to (defaults to metrics.json parent).",
    )
    parser.add_argument(
        "--filename",
        default=None,
        help="Name of the figure file (defaults to metrics_plot entry or metrics.pdf).",
    )
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else None
    plot_from_metrics_json(
        args.metrics_json,
        output_dir=output_dir,
        filename=args.filename,
    )


if __name__ == "__main__":
    _main()
