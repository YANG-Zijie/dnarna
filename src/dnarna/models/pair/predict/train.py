"""
Train a DNA-RNA pair classifier from precomputed embeddings.

The input pair file is treated as positives by default. Negative pairs are sampled
randomly unless the pair file already contains negative labels.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dnarna.models.shared.embed import load_embeddings, save_embeddings_npz
from dnarna.models.shared.predict.train import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EARLY_STOP_PATIENCE,
    DEFAULT_EPOCHS,
    DEFAULT_HEARTBEAT_SECONDS,
    DEFAULT_LR,
    DEFAULT_MONITOR,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SAVE_BEST,
    DEFAULT_SEED,
    DEFAULT_VAL_FRACTION,
    DEFAULT_WEIGHT_DECAY,
    setup_logging,
    train_binary_classifier,
)
from dnarna.models.pair.predict.utils import (
    build_pair_features,
    load_table,
    normalize_pair_dataframe,
    sample_negative_pairs,
)

LOGGER = logging.getLogger("pair_train")


@dataclass
class TrainConfig:
    pairs_file: str
    dna_embeddings: str
    rna_embeddings: str
    output_dir: str
    pair_id_col: str
    dna_id_col: str
    rna_id_col: str
    label_col: str | None
    split_col: str | None
    feature_mode: str
    negative_ratio: float
    max_pairs: int
    chunk_size: int
    pair_embeddings_input: str | None
    metadata_input: str | None
    pair_embeddings_output: str | None
    metadata_output: str | None
    report_output: str | None
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
    save_best: bool
    monitor: str
    early_stop_patience: int


def _parse_hidden_dims(value: str) -> list[int]:
    tokens = [part.strip() for part in value.split(",")]
    dims = [int(tok) for tok in tokens if tok]
    if not dims:
        raise ValueError("hidden_dims must be a comma-separated list of integers.")
    return dims


def _parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train a DNA-RNA pair classifier using precomputed embeddings."
    )
    parser.add_argument("--pairs_file", required=True)
    parser.add_argument("--dna_embeddings", required=True)
    parser.add_argument("--rna_embeddings", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pair_id_col", default="pair_id")
    parser.add_argument("--dna_id_col", default="dna_id")
    parser.add_argument("--rna_id_col", default="rna_id")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--split_col", default="")
    parser.add_argument(
        "--feature_mode",
        default="concat",
        help="Pair feature mode: concat, absdiff, mul, all.",
    )
    parser.add_argument(
        "--negative_ratio",
        type=float,
        default=1.0,
        help="Negatives per positive when sampling (ignored if negatives already provided).",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=0,
        help="Optional cap on number of positive pairs (0 = no limit).",
    )
    parser.add_argument("--chunk_size", type=int, default=4096)
    parser.add_argument(
        "--pair_embeddings_input",
        default="",
        help="Reuse existing pair embeddings .npz (skips feature construction).",
    )
    parser.add_argument(
        "--metadata_input",
        default="",
        help="Reuse existing pair metadata .csv/.parquet (skips sampling/metadata generation).",
    )
    parser.add_argument("--pair_embeddings_output", default="")
    parser.add_argument("--metadata_output", default="")
    parser.add_argument("--report_output", default="")
    parser.add_argument(
        "--log_file",
        default="",
        help="Optional path to write logs (default: <output_dir>/train.log).",
    )
    parser.add_argument("--no_log_file", action="store_true")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--hidden_dims", required=True)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--val_fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--heartbeat_seconds", type=float, default=DEFAULT_HEARTBEAT_SECONDS)
    parser.add_argument("--no_save_best", action="store_true")
    parser.add_argument(
        "--monitor",
        default=DEFAULT_MONITOR,
        choices=["val_f1", "val_loss"],
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=DEFAULT_EARLY_STOP_PATIENCE,
    )
    args = parser.parse_args()

    log_file = None if args.no_log_file else (args.log_file or "")
    if log_file == "":
        log_file = str(Path(args.output_dir).expanduser() / "train.log")

    label_col = str(args.label_col).strip() if args.label_col else None
    if label_col == "":
        label_col = None
    split_col = str(args.split_col).strip() if args.split_col else None
    if split_col == "":
        split_col = None

    pair_embeddings_input = str(args.pair_embeddings_input).strip()
    if pair_embeddings_input == "":
        pair_embeddings_input = None
    metadata_input = str(args.metadata_input).strip()
    if metadata_input == "":
        metadata_input = None

    return TrainConfig(
        pairs_file=args.pairs_file,
        dna_embeddings=args.dna_embeddings,
        rna_embeddings=args.rna_embeddings,
        output_dir=args.output_dir,
        pair_id_col=args.pair_id_col,
        dna_id_col=args.dna_id_col,
        rna_id_col=args.rna_id_col,
        label_col=label_col,
        split_col=split_col,
        feature_mode=args.feature_mode,
        negative_ratio=float(args.negative_ratio),
        max_pairs=int(args.max_pairs),
        chunk_size=int(args.chunk_size),
        pair_embeddings_input=pair_embeddings_input,
        metadata_input=metadata_input,
        pair_embeddings_output=args.pair_embeddings_output or None,
        metadata_output=args.metadata_output or None,
        report_output=args.report_output or None,
        log_file=log_file,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        hidden_dims=_parse_hidden_dims(args.hidden_dims),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
        device=args.device,
        num_workers=int(args.num_workers),
        progress=not args.no_progress,
        heartbeat_seconds=float(args.heartbeat_seconds),
        save_best=not args.no_save_best,
        monitor=str(args.monitor),
        early_stop_patience=int(args.early_stop_patience),
    )


def _resolve_output_path(
    output_dir: Path, override: str | None, stem: str, suffix: str
) -> Path:
    if override:
        return Path(override).expanduser()
    return output_dir / f"{stem}{suffix}"


def _select_positive_pairs(
    df: pd.DataFrame,
    *,
    label_col: str | None,
    max_pairs: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if label_col and label_col in df.columns:
        pos_df = df[df[label_col] == 1].copy()
        neg_df = df[df[label_col] == 0].copy()
    else:
        pos_df = df.copy()
        neg_df = df.iloc[0:0].copy()
        neg_df["label"] = []

    if max_pairs and max_pairs > 0 and len(pos_df) > max_pairs:
        pos_df = pos_df.sample(n=max_pairs, random_state=seed).copy()
    return pos_df, neg_df


def main() -> None:
    cfg = _parse_args()
    output_dir = Path(cfg.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(Path(cfg.log_file) if cfg.log_file else None, quiet_tqdm=not cfg.progress)
    LOGGER.info("Starting pair classifier training.")

    if cfg.pair_embeddings_input or cfg.metadata_input:
        if not cfg.pair_embeddings_input or not cfg.metadata_input:
            raise ValueError(
                "Both --pair_embeddings_input and --metadata_input are required to reuse."
            )
        if cfg.label_col is None:
            raise ValueError(
                "label_col is required when reusing pair metadata for training."
            )

        pair_embeddings_path = Path(cfg.pair_embeddings_input).expanduser()
        metadata_path = Path(cfg.metadata_input).expanduser()
        if not pair_embeddings_path.exists():
            raise FileNotFoundError(
                f"Pair embeddings not found: {pair_embeddings_path}"
            )
        if not metadata_path.exists():
            raise FileNotFoundError(f"Pair metadata not found: {metadata_path}")

        LOGGER.info("Reusing pair embeddings: %s", pair_embeddings_path)
        LOGGER.info("Reusing pair metadata: %s", metadata_path)
        LOGGER.info(
            "Reuse mode enabled; skipping pair feature construction."
        )
        if cfg.negative_ratio or cfg.max_pairs:
            LOGGER.info(
                "Skipping negative sampling and max_pairs (reusing precomputed features)."
            )

        train_binary_classifier(
            embeddings_npz=pair_embeddings_path,
            load_embeddings=load_embeddings,
            metadata_file=metadata_path,
            output_dir=output_dir,
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
            id_col=cfg.pair_id_col,
            label_col=cfg.label_col,
            split_col=cfg.split_col,
            train_splits=[],
            val_splits=[],
            save_best=cfg.save_best,
            monitor=cfg.monitor,
            early_stop_patience=cfg.early_stop_patience,
            config={**asdict(cfg), "pair_feature_mode": cfg.feature_mode},
            logger=LOGGER,
        )
        return

    dna_embeddings, dna_ids = load_embeddings(Path(cfg.dna_embeddings).expanduser())
    rna_embeddings, rna_ids = load_embeddings(Path(cfg.rna_embeddings).expanduser())
    dna_embeddings = dna_embeddings.astype(np.float32, copy=False)
    rna_embeddings = rna_embeddings.astype(np.float32, copy=False)
    dna_id_to_idx = {str(dna_id): idx for idx, dna_id in enumerate(dna_ids)}
    rna_id_to_idx = {str(rna_id): idx for idx, rna_id in enumerate(rna_ids)}

    pairs_path = Path(cfg.pairs_file).expanduser()
    raw_pairs = load_table(pairs_path)
    cleaned_pairs, clean_report = normalize_pair_dataframe(
        raw_pairs,
        pair_id_col=cfg.pair_id_col,
        dna_id_col=cfg.dna_id_col,
        rna_id_col=cfg.rna_id_col,
        label_col=cfg.label_col,
    )

    pos_df, neg_df_existing = _select_positive_pairs(
        cleaned_pairs,
        label_col=cfg.label_col,
        max_pairs=cfg.max_pairs,
        seed=cfg.seed,
    )
    pos_df = pos_df.copy()
    pos_df["label"] = 1
    neg_df_existing = neg_df_existing.copy()
    if not neg_df_existing.empty:
        neg_df_existing["label"] = 0

    if pos_df.empty:
        raise ValueError("No positive pairs available after cleaning.")

    pos_df["dna_idx"] = pos_df[cfg.dna_id_col].map(dna_id_to_idx)
    pos_df["rna_idx"] = pos_df[cfg.rna_id_col].map(rna_id_to_idx)
    missing_pos = pos_df["dna_idx"].isna() | pos_df["rna_idx"].isna()
    pos_df = pos_df.loc[~missing_pos].copy()
    if pos_df.empty:
        raise ValueError(
            "No positive pairs remain after filtering for available embeddings."
        )

    neg_df_existing["dna_idx"] = neg_df_existing[cfg.dna_id_col].map(dna_id_to_idx)
    neg_df_existing["rna_idx"] = neg_df_existing[cfg.rna_id_col].map(rna_id_to_idx)
    missing_neg = neg_df_existing["dna_idx"].isna() | neg_df_existing["rna_idx"].isna()
    neg_df_existing = neg_df_existing.loc[~missing_neg].copy()

    n_pos = int(len(pos_df))
    n_neg_existing = int(len(neg_df_existing))
    n_neg_target = int(max(0, round(n_pos * cfg.negative_ratio)))
    needs_sampling = n_neg_existing == 0 and n_neg_target > 0
    neg_df_sampled = None

    if needs_sampling:
        pos_pairs_set = set(
            zip(pos_df[cfg.dna_id_col].astype(str), pos_df[cfg.rna_id_col].astype(str))
        )
        sampled = sample_negative_pairs(
            pos_pairs=pos_pairs_set,
            dna_ids=list(dna_id_to_idx.keys()),
            rna_ids=list(rna_id_to_idx.keys()),
            n_samples=n_neg_target,
            seed=cfg.seed,
        )
        if sampled:
            neg_df_sampled = pd.DataFrame(sampled, columns=[cfg.dna_id_col, cfg.rna_id_col])
            neg_df_sampled[cfg.pair_id_col] = [
                f"neg_{i+1}" for i in range(len(neg_df_sampled))
            ]
            neg_df_sampled["label"] = 0
            neg_df_sampled["dna_idx"] = neg_df_sampled[cfg.dna_id_col].map(dna_id_to_idx)
            neg_df_sampled["rna_idx"] = neg_df_sampled[cfg.rna_id_col].map(rna_id_to_idx)
    elif n_neg_existing > 0 and cfg.negative_ratio > 0:
        LOGGER.info(
            "Negative labels already present (%d rows); ignoring --negative_ratio.",
            n_neg_existing,
        )

    neg_parts = [neg_df_existing]
    if neg_df_sampled is not None and not neg_df_sampled.empty:
        neg_parts.append(neg_df_sampled)
    neg_df = pd.concat(neg_parts, ignore_index=True) if neg_parts else neg_df_existing

    if neg_df.empty:
        raise ValueError(
            "No negative pairs available. Provide labeled negatives or set --negative_ratio > 0."
        )

    pos_df["source"] = "positive"
    neg_df["source"] = "negative"
    pairs_df = pd.concat([pos_df, neg_df], ignore_index=True)

    dna_indices = pairs_df["dna_idx"].astype(int).to_numpy()
    rna_indices = pairs_df["rna_idx"].astype(int).to_numpy()
    pair_ids = pairs_df[cfg.pair_id_col].astype(str).tolist()

    LOGGER.info(
        "Building pair features (pairs=%d, mode=%s, chunk_size=%d).",
        len(pairs_df),
        cfg.feature_mode,
        cfg.chunk_size,
    )
    features = build_pair_features(
        dna_embeddings,
        rna_embeddings,
        dna_indices,
        rna_indices,
        mode=cfg.feature_mode,
        chunk_size=cfg.chunk_size,
        show_progress=cfg.progress,
    )

    stem = pairs_path.stem
    pair_embeddings_path = _resolve_output_path(
        output_dir, cfg.pair_embeddings_output, stem, ".pair_embeddings.npz"
    )
    metadata_path = _resolve_output_path(
        output_dir, cfg.metadata_output, stem, ".pair_metadata.csv"
    )
    report_path = _resolve_output_path(
        output_dir, cfg.report_output, stem, ".pair_report.json"
    )

    save_embeddings_npz(pair_embeddings_path, pair_ids, features)
    split_col = (
        cfg.split_col if cfg.split_col and cfg.split_col in pairs_df.columns else None
    )
    if split_col and neg_df_sampled is not None:
        LOGGER.warning("split_col ignored because negatives were sampled.")
        split_col = None

    metadata_cols = [
        cfg.pair_id_col,
        cfg.dna_id_col,
        cfg.rna_id_col,
        "label",
        "source",
    ]
    if split_col:
        metadata_cols.append(split_col)
    pairs_df[metadata_cols].to_csv(metadata_path, index=False)

    report = {
        "cleaning": clean_report,
        "input_pairs": int(len(raw_pairs)),
        "kept_pairs": int(len(cleaned_pairs)),
        "positives": int(len(pos_df)),
        "positives_missing_embeddings": int(missing_pos.sum()),
        "negatives_missing_embeddings": int(missing_neg.sum()),
        "negatives_existing": int(n_neg_existing),
        "negatives_sampled": int(0 if neg_df_sampled is None else len(neg_df_sampled)),
        "negatives_total": int(len(neg_df)),
        "feature_mode": str(cfg.feature_mode),
        "feature_dim": int(features.shape[1]) if features.size else 0,
        "pair_embeddings": str(pair_embeddings_path),
        "metadata_file": str(metadata_path),
    }
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=True, indent=2)

    LOGGER.info("Pair embeddings saved to %s", pair_embeddings_path)
    LOGGER.info("Pair metadata saved to %s", metadata_path)
    LOGGER.info("Pair report saved to %s", report_path)

    train_binary_classifier(
        embeddings_npz=pair_embeddings_path,
        load_embeddings=load_embeddings,
        metadata_file=metadata_path,
        output_dir=output_dir,
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
        id_col=cfg.pair_id_col,
        label_col="label",
        split_col=split_col,
        train_splits=[],
        val_splits=[],
        save_best=cfg.save_best,
        monitor=cfg.monitor,
        early_stop_patience=cfg.early_stop_patience,
        config={**asdict(cfg), "pair_feature_mode": cfg.feature_mode},
        logger=LOGGER,
    )


if __name__ == "__main__":
    main()
