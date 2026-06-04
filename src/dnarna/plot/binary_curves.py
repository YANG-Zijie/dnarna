"""
ROC / Precision-Recall plotting utilities for binary classifiers.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

__all__ = [
    "compute_binary_curves",
    "extract_curve_arrays",
    "plot_binary_curves",
    "plot_binary_curves_from_metrics_json",
]


def compute_binary_curves(
    labels: Sequence[float | int] | None,
    scores: Sequence[float] | None,
) -> dict[str, Any] | None:
    """
    Compute ROC / PR curve statistics given ground-truth labels and predicted scores.
    Returns ``None`` when inputs are missing/invalid.

    Returned keys include curve points plus per-point thresholds:
    - ROC: ``fpr``, ``tpr``, ``roc_thresholds``
    - PR: ``precision``, ``recall``, ``pr_thresholds``
    """
    if labels is None or scores is None:
        return None
    try:
        y_true = np.asarray(labels, dtype=float)
        y_score = np.asarray(scores, dtype=float)
    except Exception:
        return None

    if y_true.shape[0] == 0 or y_true.shape != y_score.shape:
        return None
    if np.unique(y_true).size < 2:
        return None

    try:
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        roc_auc_val = auc(fpr, tpr)
        pr_auc_val = auc(recall, precision)
        ap_val = average_precision_score(y_true, y_score)
    except Exception:
        return None

    return {
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": roc_thresholds,
        "precision": precision,
        "recall": recall,
        "pr_thresholds": pr_thresholds,
        "roc_auc": roc_auc_val,
        "pr_auc": pr_auc_val,
        "ap": ap_val,
    }


def extract_curve_arrays(
    metrics: Mapping[str, Any],
) -> tuple[Sequence[float | int] | None, Sequence[float] | None]:
    """
    Best-effort extraction of label/score arrays from a metrics-like mapping.

    Supports common key pairs: (val_labels, val_probs), (labels, scores), (y_true, y_score).
    Returns (None, None) when no suitable pair is found.
    """
    candidate_pairs = [
        ("val_labels", "val_probs"),
        ("val_labels", "val_scores"),
        ("labels", "scores"),
        ("y_true", "y_score"),
        ("labels", "probs"),
    ]
    for label_key, score_key in candidate_pairs:
        if label_key in metrics and score_key in metrics:
            return metrics[label_key], metrics[score_key]
    preds = metrics.get("predictions")
    if isinstance(preds, list) and preds:
        first = preds[0]
        if isinstance(first, Mapping) and "label" in first:
            score_field = (
                "prob" if "prob" in first else "score" if "score" in first else None
            )
            if score_field:
                labels = [item["label"] for item in preds]
                scores = [item[score_field] for item in preds]
                return labels, scores
    return None, None


def _write_curve_csv(
    *,
    output_path: Path,
    header: Sequence[str],
    rows: Sequence[Sequence[object]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(header))
        writer.writerows(rows)


def plot_binary_curves(
    labels: Sequence[float | int],
    scores: Sequence[float],
    *,
    output_path: Path,
    split_label: str = "val",
    save_curve_csv: bool = True,
) -> Path | None:
    """
    Plot ROC + PR curves to a PDF (or any Matplotlib-supported path).

    When ``save_curve_csv`` is True (default), also writes raw curve points next to the
    figure: ``{output_path.stem}.roc.csv`` and ``{output_path.stem}.pr.csv``.
    """
    curves = compute_binary_curves(labels, scores)
    if curves is None:
        return None

    fpr, tpr, roc_thresholds, roc_auc_val = (
        curves["fpr"],
        curves["tpr"],
        curves["roc_thresholds"],
        curves["roc_auc"],
    )
    precision, recall, pr_auc_val, ap_val = (
        curves["precision"],
        curves["recall"],
        curves["pr_auc"],
        curves["ap"],
    )
    pr_thresholds = curves["pr_thresholds"]

    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if save_curve_csv:
        roc_csv = output_path.with_name(f"{output_path.stem}.roc.csv")
        pr_csv = output_path.with_name(f"{output_path.stem}.pr.csv")

        _write_curve_csv(
            output_path=roc_csv,
            header=["threshold", "fpr", "tpr"],
            rows=list(zip(roc_thresholds.tolist(), fpr.tolist(), tpr.tolist())),
        )

        pr_thresholds_aligned = np.concatenate(
            ([np.nan], np.asarray(pr_thresholds, dtype=float))
        )
        _write_curve_csv(
            output_path=pr_csv,
            header=["threshold", "recall", "precision"],
            rows=list(
                zip(
                    pr_thresholds_aligned.tolist(),
                    recall.tolist(),
                    precision.tolist(),
                )
            ),
        )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    roc_ax = axes[0]
    roc_ax.plot(fpr, tpr, label=f"{split_label} AUC={roc_auc_val:.3f}")
    roc_ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.6)
    roc_ax.set_title("ROC Curve")
    roc_ax.set_xlabel("False Positive Rate")
    roc_ax.set_ylabel("True Positive Rate")
    roc_ax.legend()
    roc_ax.grid(True, linestyle="--", alpha=0.3)

    pr_ax = axes[1]
    pr_ax.plot(
        recall, precision, label=f"{split_label} AP={ap_val:.3f} | AUC={pr_auc_val:.3f}"
    )
    pr_ax.set_title("Precision-Recall Curve")
    pr_ax.set_xlabel("Recall")
    pr_ax.set_ylabel("Precision")
    pr_ax.set_xlim(0, 1)
    pr_ax.set_ylim(0, 1)
    pr_ax.legend()
    pr_ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_binary_curves_from_metrics_json(
    metrics_json_path: str | Path,
    *,
    output_dir: Path | None = None,
    filename: str = "roc_pr.pdf",
    save_curve_csv: bool = True,
) -> Path | None:
    metrics_json = Path(metrics_json_path).expanduser()
    with metrics_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    labels, scores = extract_curve_arrays(data)
    if labels is None or scores is None:
        return None
    target_dir = output_dir or metrics_json.parent
    return plot_binary_curves(
        labels,
        scores,
        output_path=target_dir / filename,
        split_label="val",
        save_curve_csv=save_curve_csv,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ROC + PR curves from a metrics.json file that contains labels and probabilities."
    )
    parser.add_argument("--metrics_json", required=True, help="Path to metrics.json.")
    parser.add_argument("--output_dir", default=None, help="Directory for the figure.")
    parser.add_argument("--filename", default="roc_pr.pdf", help="Output figure name.")
    parser.add_argument(
        "--no_save_curve_csv",
        action="store_true",
        help="Disable saving ROC/PR curve points to CSV next to the figure.",
    )
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir).expanduser() if args.output_dir else None
    plot_binary_curves_from_metrics_json(
        args.metrics_json,
        output_dir=out_dir,
        filename=args.filename,
        save_curve_csv=not args.no_save_curve_csv,
    )


if __name__ == "__main__":
    _main()
