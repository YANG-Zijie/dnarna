"""
Train a binary classifier on RNA-FM embeddings produced by ``dnarna.models.rnafm.embed``.

Example:
    python -m dnarna.models.rnafm.predict.train \
        --embeddings_npz data/rna_embeddings.npz \
        --metadata_file data/splits.parquet \
        --output_dir outputs/rnafm_clf \
        --epochs 25 --hidden_dims 512,256 --device cuda:0

Required inputs:
    - embeddings_npz: Combined RNA-FM embeddings with ``embeddings`` + ``ids``.
    - metadata_file: Parquet/CSV with at least ``id`` and binary ``label`` columns
      (optionally ``split`` to control train/val selection).

Outputs:
    <output_dir>/model.pt
        - Torch checkpoint with classifier weights and feature normalization stats.
    <output_dir>/best_model.pt
        - Best validation checkpoint when enabled (see --monitor / --no_save_best).
    <output_dir>/train.log
        - Combined console logs saved to disk (unless disabled).
    <output_dir>/metrics.json
        - Training/validation metrics (loss, accuracy, precision, recall, F1) history with basic dataset stats.
    <output_dir>/metrics.csv
        - CSV log with per-epoch metrics (epoch, split, loss, accuracy, precision, recall, F1, TP, FP, TN, FN).
    <output_dir>/metrics.pdf
        - Training curves (loss and F1) for train/validation splits when available.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from dnarna.models.shared.embed import load_embeddings
from dnarna.models.shared.predict.train import (
    DEFAULT_EARLY_STOP_PATIENCE,
    DEFAULT_MONITOR,
    DEFAULT_SAVE_BEST,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_HEARTBEAT_SECONDS,
    DEFAULT_ID_COL,
    DEFAULT_LABEL_COL,
    DEFAULT_LR,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SEED,
    DEFAULT_SPLIT_COL,
    DEFAULT_TRAIN_SPLITS,
    DEFAULT_VAL_FRACTION,
    DEFAULT_VAL_SPLITS,
    DEFAULT_WEIGHT_DECAY,
    setup_logging,
    train_binary_classifier,
)

LOGGER = logging.getLogger("rnafm_train")


@dataclass
class TrainConfig:
    embeddings_npz: str
    metadata_file: str
    output_dir: str
    log_file: str | None
    epochs: int
    batch_size: int
    hidden_dims: list[int]
    lr: float
    weight_decay: float
    val_fraction: float
    seed: int
    device: str | None
    num_workers: int
    progress: bool
    heartbeat_seconds: float
    id_col: str
    label_col: str
    split_col: str | None
    train_splits: list[str]
    val_splits: list[str]
    save_best: bool
    monitor: str
    early_stop_patience: int


def _parse_split_arg(value: str | None) -> list[str]:
    if value is None:
        return []
    tokens = [part.strip() for part in value.split(",")]
    return [token for token in tokens if token]


def _parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train a binary classifier to predict RNA labels from embeddings."
    )
    parser.add_argument(
        "--embeddings_npz",
        required=True,
        help="Combined embeddings .npz with embeddings and ids.",
    )
    parser.add_argument(
        "--metadata_file",
        required=True,
        help="Parquet/CSV metadata file containing at least id and label columns (e.g., split output).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save the trained model and metadata.",
    )
    parser.add_argument(
        "--log_file",
        default="",
        help=(
            "Optional path to write logs (default: <output_dir>/train.log). "
            "Use --no_log_file to disable."
        ),
    )
    parser.add_argument(
        "--no_log_file",
        action="store_true",
        help="Disable writing logs to a file; console only.",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--hidden_dims",
        required=True,
        help="Comma-separated hidden layer widths (e.g. 512,256).",
    )
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate.")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=DEFAULT_VAL_FRACTION,
        help="Fraction of train data randomly held out when no explicit validation split is provided.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default=None, help="Torch device (e.g., cuda:0).")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="DataLoader workers (0 disables multiprocessing).",
    )
    parser.add_argument(
        "--id_col", default=DEFAULT_ID_COL, help="Identifier column in metadata."
    )
    parser.add_argument(
        "--label_col",
        default=DEFAULT_LABEL_COL,
        help="Label column in metadata (expects binary 0/1).",
    )
    parser.add_argument(
        "--split_col",
        default=DEFAULT_SPLIT_COL,
        help="Metadata column describing dataset split; provide empty string to ignore.",
    )
    parser.add_argument(
        "--train_splits",
        default=DEFAULT_TRAIN_SPLITS,
        help="Comma-separated split names to use for training when split_col is available. Empty string uses all rows.",
    )
    parser.add_argument(
        "--val_splits",
        default=DEFAULT_VAL_SPLITS,
        help="Comma-separated split names to use for validation when split_col is available. Empty string disables explicit validation splits.",
    )
    parser.add_argument(
        "--no_save_best",
        action="store_true",
        help="Disable saving <output_dir>/best_model.pt (best-by-validation checkpoint).",
    )
    parser.add_argument(
        "--monitor",
        default=DEFAULT_MONITOR,
        choices=["val_f1", "val_loss"],
        help="Metric used to select best checkpoint when saving best is enabled.",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=DEFAULT_EARLY_STOP_PATIENCE,
        help="Stop training after N epochs without improvement on --monitor (0 disables).",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable tqdm progress bar output.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed per-epoch logging even when tqdm is enabled.",
    )
    parser.add_argument(
        "--heartbeat_seconds",
        type=float,
        default=DEFAULT_HEARTBEAT_SECONDS,
        help="Emit a status line every N seconds during epochs when tqdm is not visible (<=0 disables).",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = None
    if not args.no_log_file:
        log_file = (
            Path(args.log_file).expanduser()
            if str(args.log_file).strip()
            else output_dir / "train.log"
        )
    setup_logging(
        log_file,
        level=logging.INFO,
        quiet_tqdm=(not args.no_progress and not args.verbose),
    )

    hidden_dims_str = str(args.hidden_dims).strip()
    parts = [p.strip() for p in hidden_dims_str.split(",")]
    dims = [int(p) for p in parts if p]
    hidden_dims = [d for d in dims if d > 0]
    if not hidden_dims:
        raise ValueError("--hidden_dims must contain at least one positive integer.")
    split_col = args.split_col.strip() or None
    train_splits = _parse_split_arg(args.train_splits if split_col else "")
    val_splits = _parse_split_arg(args.val_splits if split_col else "")
    return TrainConfig(
        embeddings_npz=args.embeddings_npz,
        metadata_file=args.metadata_file,
        output_dir=str(output_dir),
        log_file=str(log_file) if log_file is not None else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dims=hidden_dims,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_fraction=max(0.0, min(0.5, args.val_fraction)),
        seed=args.seed,
        device=args.device,
        num_workers=max(0, args.num_workers),
        progress=not args.no_progress,
        heartbeat_seconds=float(args.heartbeat_seconds),
        id_col=args.id_col,
        label_col=args.label_col,
        split_col=split_col,
        train_splits=train_splits,
        val_splits=val_splits,
        save_best=(DEFAULT_SAVE_BEST and (not args.no_save_best)),
        monitor=str(args.monitor),
        early_stop_patience=int(args.early_stop_patience),
    )


def _train(cfg: TrainConfig) -> None:
    train_binary_classifier(
        embeddings_npz=Path(cfg.embeddings_npz),
        load_embeddings=load_embeddings,
        metadata_file=Path(cfg.metadata_file),
        output_dir=Path(cfg.output_dir),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        hidden_dims=cfg.hidden_dims,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        val_fraction=cfg.val_fraction,
        seed=cfg.seed,
        device=cfg.device,
        num_workers=cfg.num_workers,
        progress=cfg.progress,
        heartbeat_seconds=cfg.heartbeat_seconds,
        id_col=cfg.id_col,
        label_col=cfg.label_col,
        split_col=cfg.split_col,
        train_splits=cfg.train_splits,
        val_splits=cfg.val_splits,
        save_best=cfg.save_best,
        monitor=cfg.monitor,
        early_stop_patience=cfg.early_stop_patience,
        config=asdict(cfg),
        logger=LOGGER,
    )


def main() -> None:
    cfg = _parse_args()
    _train(cfg)


__all__ = ["TrainConfig", "main", "_train"]


if __name__ == "__main__":
    main()
