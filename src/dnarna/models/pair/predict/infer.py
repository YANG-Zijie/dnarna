"""
Run inference for DNA-RNA pairs using a trained pair classifier.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from dnarna.models.shared.embed import load_embeddings
from dnarna.models.pair.predict.aggregate import (
    BEST_DNA_ID_COLUMN,
    BEST_DNA_WINDOW_END_COLUMN,
    BEST_DNA_WINDOW_INDEX_COLUMN,
    BEST_DNA_WINDOW_START_COLUMN,
    BEST_RNA_ID_COLUMN,
    BEST_RNA_WINDOW_END_COLUMN,
    BEST_RNA_WINDOW_INDEX_COLUMN,
    BEST_RNA_WINDOW_START_COLUMN,
    BEST_WINDOW_PAIR_ID_COLUMN,
    BEST_WINDOW_PROB_COLUMN,
    COMBINED_PRED_COLUMN,
    COMBINED_SCORE_COLUMN,
    DNA_PARENT_ID_COLUMN,
    DNA_WINDOW_COUNT_COLUMN,
    DNA_WINDOW_INDEX_COLUMN,
    PAIR_GROUP_ID_COLUMN,
    POSITIVE_WINDOW_PAIR_COUNT_COLUMN,
    POSITIVE_WINDOW_PAIR_FRACTION_COLUMN,
    PROB_MAX_COLUMN,
    PROB_MEAN_COLUMN,
    RNA_PARENT_ID_COLUMN,
    RNA_WINDOW_COUNT_COLUMN,
    RNA_WINDOW_INDEX_COLUMN,
    WINDOW_PAIR_COUNT_COLUMN,
    aggregate_pair_predictions,
    attach_parent_metadata,
    load_sequence_window_metadata,
    resolve_pair_summary_output_path,
    write_pair_summary,
)
from dnarna.models.shared.predict.infer import (
    load_classifier_checkpoint,
    normalize_embeddings,
    predict_probabilities,
)
from dnarna.models.shared.predict.train import LogisticMLP
from dnarna.models.pair.predict.utils import (
    build_pair_features,
    infer_extension,
    load_table,
    normalize_pair_dataframe,
)

LOGGER = logging.getLogger("pair_infer")


@dataclass
class InferConfig:
    pairs_file: str
    dna_embeddings: str
    rna_embeddings: str
    checkpoint: str
    output: str
    output_dir: str
    dna_seq_file: str
    rna_seq_file: str
    pair_id_col: str
    dna_id_col: str
    rna_id_col: str
    feature_mode: str
    chunk_size: int
    num_workers: int
    max_pairs: int
    all_pairs: bool
    dna_block_size: int
    rna_block_size: int
    max_dna: int
    max_rna: int
    batch_size: int
    device: str | None
    threshold: float
    progress: bool
    summary_only: bool


def _parse_args() -> InferConfig:
    parser = argparse.ArgumentParser(
        description="Predict binding scores for DNA-RNA pairs using a trained pair classifier."
    )
    parser.add_argument("--pairs_file", default="")
    parser.add_argument("--dna_embeddings", required=True)
    parser.add_argument("--rna_embeddings", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--output_name", default="pair_predictions.csv")
    parser.add_argument(
        "--dna_seq_file",
        default="",
        help="Optional DNA CSV/Parquet with window metadata for summary aggregation.",
    )
    parser.add_argument(
        "--rna_seq_file",
        default="",
        help="Optional RNA CSV/Parquet with window metadata for summary aggregation.",
    )
    parser.add_argument("--pair_id_col", default="pair_id")
    parser.add_argument("--dna_id_col", default="dna_id")
    parser.add_argument("--rna_id_col", default="rna_id")
    parser.add_argument(
        "--feature_mode",
        default="concat",
        help="Pair feature mode: concat, absdiff, mul, all.",
    )
    parser.add_argument("--chunk_size", type=int, default=4096)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="CPU threads for pair feature construction.",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=0,
        help="Optional cap on number of pairs from pairs_file (0 = no limit).",
    )
    parser.add_argument(
        "--all_pairs",
        action="store_true",
        help="Ignore pairs_file and score all DNA x RNA pairs.",
    )
    parser.add_argument(
        "--dna_block_size",
        type=int,
        default=128,
        help="DNA block size when generating all pairs.",
    )
    parser.add_argument(
        "--rna_block_size",
        type=int,
        default=128,
        help="RNA block size when generating all pairs.",
    )
    parser.add_argument(
        "--max_dna",
        type=int,
        default=0,
        help="Optional cap on number of DNA embeddings in all_pairs mode (0 = no limit).",
    )
    parser.add_argument(
        "--max_rna",
        type=int,
        default=0,
        help="Optional cap on number of RNA embeddings in all_pairs mode (0 = no limit).",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--summary_only",
        "--skip_raw_output",
        dest="summary_only",
        action="store_true",
        help=(
            "all_pairs only: skip writing raw pair_predictions.csv and write "
            "only the aggregated summary. Requires --dna_seq_file and --rna_seq_file."
        ),
    )
    parser.add_argument("--no_progress", action="store_true")
    args = parser.parse_args()

    output_value = str(args.output).strip()
    output_dir = str(args.output_dir).strip()
    output_name = str(args.output_name).strip() or "pair_predictions.csv"
    if output_dir:
        output_path = str(Path(output_dir).expanduser() / output_name)
    elif output_value:
        output_path = output_value
        output_dir = str(Path(output_path).expanduser().parent)
    else:
        raise ValueError("--output or --output_dir must be provided.")

    return InferConfig(
        pairs_file=str(args.pairs_file).strip(),
        dna_embeddings=args.dna_embeddings,
        rna_embeddings=args.rna_embeddings,
        checkpoint=args.checkpoint,
        output=output_path,
        output_dir=output_dir,
        dna_seq_file=str(args.dna_seq_file).strip(),
        rna_seq_file=str(args.rna_seq_file).strip(),
        pair_id_col=args.pair_id_col,
        dna_id_col=args.dna_id_col,
        rna_id_col=args.rna_id_col,
        feature_mode=args.feature_mode,
        chunk_size=int(args.chunk_size),
        num_workers=int(args.num_workers),
        max_pairs=int(args.max_pairs),
        all_pairs=bool(args.all_pairs),
        dna_block_size=int(args.dna_block_size),
        rna_block_size=int(args.rna_block_size),
        max_dna=int(args.max_dna),
        max_rna=int(args.max_rna),
        batch_size=int(args.batch_size),
        device=args.device,
        threshold=float(args.threshold),
        progress=not args.no_progress,
        summary_only=bool(args.summary_only),
    )


def _setup_logging(log_path: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def _write_meta(output_path: Path, payload: dict) -> Path:
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=True, indent=2)
    return meta_path


def _build_prediction_output_df(
    pairs: pd.DataFrame,
    probs: np.ndarray,
    *,
    pair_id_col: str,
    dna_id_col: str,
    rna_id_col: str,
    threshold: float,
) -> pd.DataFrame:
    preds = (probs >= threshold).astype(int)
    out_df = pairs.copy()
    out_df = out_df.drop(columns=[c for c in ("dna_idx", "rna_idx") if c in out_df.columns])
    out_df[pair_id_col] = out_df[pair_id_col].astype(str)
    out_df[dna_id_col] = out_df[dna_id_col].astype(str)
    out_df[rna_id_col] = out_df[rna_id_col].astype(str)
    out_df["prob"] = probs.astype(np.float32)
    out_df["pred"] = preds
    return out_df


def _write_prediction_output(
    out_df: pd.DataFrame,
    output: Path,
    *,
    threshold: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    ext = infer_extension(output)
    if ext in {".parquet", ".parq", ".pq"}:
        out_df.to_parquet(output, index=False)
    elif ext.endswith(".tsv"):
        out_df.to_csv(output, index=False, sep="\t")
    else:
        out_df.to_csv(output, index=False)
    LOGGER.info(
        "Wrote predictions to %s (pos=%d neg=%d threshold=%.3f)",
        output,
        int((out_df["pred"].astype(int) == 1).sum()),
        int((out_df["pred"].astype(int) == 0).sum()),
        threshold,
    )


def _append_predictions(
    *,
    pair_ids: np.ndarray,
    dna_ids: np.ndarray,
    rna_ids: np.ndarray,
    probs: np.ndarray,
    output: Path,
    threshold: float,
    header: bool,
    pair_id_col: str,
    dna_id_col: str,
    rna_id_col: str,
) -> None:
    preds = (probs >= threshold).astype(int)
    out_df = pd.DataFrame(
        {
            pair_id_col: pair_ids,
            dna_id_col: dna_ids,
            rna_id_col: rna_ids,
            "prob": probs.astype(np.float32),
            "pred": preds,
        }
    )
    sep = "\t" if output.suffix.lower().endswith(".tsv") else ","
    out_df.to_csv(output, index=False, mode="a", header=header, sep=sep)


def _build_summary_stats(summary_df: pd.DataFrame | None) -> dict[str, Any] | None:
    if summary_df is None or summary_df.empty:
        return None
    return {
        "group_count": int(len(summary_df)),
        "mean_combined_score": float(summary_df[COMBINED_SCORE_COLUMN].mean()),
        "max_combined_score": float(summary_df[COMBINED_SCORE_COLUMN].max()),
        "positive_group_count": int(summary_df[COMBINED_PRED_COLUMN].sum()),
    }


def _accumulate_summary_stats(
    stats: dict[str, Any],
    summary_df: pd.DataFrame | None,
) -> None:
    if summary_df is None or summary_df.empty:
        return
    stats["group_count"] = int(stats.get("group_count", 0)) + int(len(summary_df))
    stats["combined_score_sum"] = float(stats.get("combined_score_sum", 0.0)) + float(
        summary_df[COMBINED_SCORE_COLUMN].sum()
    )
    max_score = float(summary_df[COMBINED_SCORE_COLUMN].max())
    current_max = stats.get("max_combined_score")
    stats["max_combined_score"] = (
        max_score if current_max is None else max(float(current_max), max_score)
    )
    stats["positive_group_count"] = int(
        stats.get("positive_group_count", 0)
    ) + int(summary_df[COMBINED_PRED_COLUMN].sum())


def _finalize_summary_stats(stats: dict[str, Any]) -> dict[str, Any] | None:
    group_count = int(stats.get("group_count", 0))
    if group_count <= 0:
        return None
    return {
        "group_count": group_count,
        "mean_combined_score": float(stats.get("combined_score_sum", 0.0))
        / group_count,
        "max_combined_score": float(stats["max_combined_score"]),
        "positive_group_count": int(stats.get("positive_group_count", 0)),
    }


def _append_summary_output(
    summary_df: pd.DataFrame,
    output_path: Path,
    *,
    header: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sep = "\t" if output_path.suffix.lower() == ".tsv" else ","
    summary_df.to_csv(output_path, index=False, mode="a", header=header, sep=sep)


def _build_parent_complete_index_blocks(
    indexed_meta: pd.DataFrame,
    *,
    ordered_ids: np.ndarray,
    id_col: str,
    parent_col: str,
    max_windows: int,
) -> list[np.ndarray]:
    max_windows = max(1, int(max_windows))
    ordered_ids = np.asarray(ordered_ids, dtype=str)
    aligned = indexed_meta.reindex(ordered_ids)
    missing = aligned[parent_col].isna()
    if missing.any():
        missing_ids = ordered_ids[missing.to_numpy()].astype(str).tolist()
        raise KeyError(
            f"Missing summary metadata for {len(missing_ids)} ids in column "
            f"'{id_col}' (first 10: {missing_ids[:10]})"
        )

    parent_to_indices: dict[str, list[int]] = {}
    parent_order: list[str] = []
    for idx, parent_id in enumerate(aligned[parent_col].astype(str).to_numpy()):
        if parent_id not in parent_to_indices:
            parent_to_indices[parent_id] = []
            parent_order.append(parent_id)
        parent_to_indices[parent_id].append(idx)

    blocks: list[np.ndarray] = []
    current: list[int] = []
    current_size = 0
    for parent_id in parent_order:
        parent_indices = parent_to_indices[parent_id]
        parent_size = len(parent_indices)
        if current and current_size + parent_size > max_windows:
            blocks.append(np.asarray(current, dtype=int))
            current = []
            current_size = 0
        current.extend(parent_indices)
        current_size += parent_size
    if current:
        blocks.append(np.asarray(current, dtype=int))
    return blocks


def _update_group_summary(
    states: dict[tuple[str, str], dict[str, Any]],
    block_df: pd.DataFrame,
) -> None:
    grouped = block_df.groupby([DNA_PARENT_ID_COLUMN, RNA_PARENT_ID_COLUMN], sort=False)
    for (dna_parent_id, rna_parent_id), group in grouped:
        key = (str(dna_parent_id), str(rna_parent_id))
        state = states.setdefault(
            key,
            {
                "window_pair_count": 0,
                "positive_window_pair_count": 0,
                "prob_sum": 0.0,
                "log_not_sum": 0.0,
                "prob_max": -1.0,
                "best_window_pair_id": None,
                "best_dna_id": None,
                "best_rna_id": None,
                "best_dna_window_index": None,
                "best_rna_window_index": None,
                "best_dna_window_start": None,
                "best_dna_window_end": None,
                "best_rna_window_start": None,
                "best_rna_window_end": None,
            },
        )

        probs = group["prob"].to_numpy(dtype=np.float64)
        clipped = np.clip(probs, 0.0, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_not = np.log1p(-clipped)
        state["window_pair_count"] += int(len(group))
        state["positive_window_pair_count"] += int(group["pred"].sum())
        state["prob_sum"] += float(np.sum(probs))
        state["log_not_sum"] += float(np.sum(log_not))

        best_idx = int(group["prob"].idxmax())
        best_row = group.loc[best_idx]
        best_prob = float(best_row["prob"])
        if best_prob > float(state["prob_max"]):
            state["prob_max"] = best_prob
            state["best_window_pair_id"] = str(best_row["pair_id"])
            state["best_dna_id"] = str(best_row["dna_id"])
            state["best_rna_id"] = str(best_row["rna_id"])
            for col in (
                "dna_window_index",
                "rna_window_index",
                "dna_window_start",
                "dna_window_end",
                "rna_window_start",
                "rna_window_end",
            ):
                if col in group.columns:
                    state[f"best_{col}"] = best_row[col]


def _finalize_group_summary(
    states: dict[tuple[str, str], dict[str, Any]],
    *,
    dna_window_counts: dict[str, int],
    rna_window_counts: dict[str, int],
    threshold: float,
) -> pd.DataFrame | None:
    if not states:
        return None

    records: list[dict[str, Any]] = []
    for (dna_parent_id, rna_parent_id), state in states.items():
        count = int(state["window_pair_count"])
        combined = float(-np.expm1(float(state["log_not_sum"]))) if count else float("nan")
        positive_count = int(state["positive_window_pair_count"])
        records.append(
            {
                PAIR_GROUP_ID_COLUMN: f"{dna_parent_id}__{rna_parent_id}",
                DNA_PARENT_ID_COLUMN: dna_parent_id,
                RNA_PARENT_ID_COLUMN: rna_parent_id,
                DNA_WINDOW_COUNT_COLUMN: int(dna_window_counts.get(dna_parent_id, 0)),
                RNA_WINDOW_COUNT_COLUMN: int(rna_window_counts.get(rna_parent_id, 0)),
                WINDOW_PAIR_COUNT_COLUMN: count,
                POSITIVE_WINDOW_PAIR_COUNT_COLUMN: positive_count,
                POSITIVE_WINDOW_PAIR_FRACTION_COLUMN: (
                    positive_count / count if count else np.nan
                ),
                PROB_MEAN_COLUMN: (float(state["prob_sum"]) / count) if count else np.nan,
                PROB_MAX_COLUMN: float(state["prob_max"]) if count else np.nan,
                COMBINED_SCORE_COLUMN: combined,
                COMBINED_PRED_COLUMN: int(combined >= float(threshold)) if count else 0,
                BEST_WINDOW_PAIR_ID_COLUMN: state["best_window_pair_id"],
                BEST_WINDOW_PROB_COLUMN: float(state["prob_max"]) if count else np.nan,
                BEST_DNA_ID_COLUMN: state["best_dna_id"],
                BEST_RNA_ID_COLUMN: state["best_rna_id"],
                BEST_DNA_WINDOW_INDEX_COLUMN: state["best_dna_window_index"],
                BEST_RNA_WINDOW_INDEX_COLUMN: state["best_rna_window_index"],
                BEST_DNA_WINDOW_START_COLUMN: state["best_dna_window_start"],
                BEST_DNA_WINDOW_END_COLUMN: state["best_dna_window_end"],
                BEST_RNA_WINDOW_START_COLUMN: state["best_rna_window_start"],
                BEST_RNA_WINDOW_END_COLUMN: state["best_rna_window_end"],
            }
        )

    summary_df = pd.DataFrame.from_records(records)
    summary_df = summary_df.sort_values(
        [COMBINED_SCORE_COLUMN, PROB_MAX_COLUMN, PROB_MEAN_COLUMN],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return summary_df


def _infer_from_pairs(
    cfg: InferConfig,
    *,
    dna_embeddings: np.ndarray,
    dna_ids: list[str],
    rna_embeddings: np.ndarray,
    rna_ids: list[str],
    model: torch.nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    output_path: Path,
) -> dict:
    pairs_path = Path(cfg.pairs_file).expanduser()
    raw_pairs = load_table(pairs_path)
    pairs, _ = normalize_pair_dataframe(
        raw_pairs,
        pair_id_col=cfg.pair_id_col,
        dna_id_col=cfg.dna_id_col,
        rna_id_col=cfg.rna_id_col,
        label_col=None,
    )

    dna_id_to_idx = {str(dna_id): idx for idx, dna_id in enumerate(dna_ids)}
    rna_id_to_idx = {str(rna_id): idx for idx, rna_id in enumerate(rna_ids)}
    pairs["dna_idx"] = pairs[cfg.dna_id_col].map(dna_id_to_idx)
    pairs["rna_idx"] = pairs[cfg.rna_id_col].map(rna_id_to_idx)
    missing = pairs["dna_idx"].isna() | pairs["rna_idx"].isna()
    if missing.any():
        LOGGER.warning(
            "Dropping %d pairs without matching embeddings.", int(missing.sum())
        )
        pairs = pairs.loc[~missing].copy()
    dropped_missing = int(missing.sum())
    pairs_after_missing = int(len(pairs))
    if cfg.max_pairs > 0 and pairs_after_missing > cfg.max_pairs:
        LOGGER.info(
            "Capping pairs to first %d rows (from %d).",
            cfg.max_pairs,
            pairs_after_missing,
        )
        pairs = pairs.iloc[: cfg.max_pairs].copy()

    if pairs.empty:
        raise ValueError("No pairs remain after filtering for available embeddings.")

    dna_indices = pairs["dna_idx"].astype(int).to_numpy()
    rna_indices = pairs["rna_idx"].astype(int).to_numpy()
    LOGGER.info(
        "Building pair features (pairs=%d, mode=%s, chunk_size=%d).",
        len(pairs),
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
        num_workers=cfg.num_workers,
    )

    X = normalize_embeddings(features.astype(np.float32), mean, std)
    ds = TensorDataset(torch.from_numpy(X).float())
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    probs = predict_probabilities(model, loader, device, show_progress=cfg.progress)

    out_df = _build_prediction_output_df(
        pairs,
        probs,
        pair_id_col=cfg.pair_id_col,
        dna_id_col=cfg.dna_id_col,
        rna_id_col=cfg.rna_id_col,
        threshold=cfg.threshold,
    )
    _write_prediction_output(out_df, output_path, threshold=cfg.threshold)

    summary_path = None
    summary_input_df = out_df
    if cfg.dna_seq_file and cfg.rna_seq_file:
        try:
            dna_meta = load_sequence_window_metadata(
                cfg.dna_seq_file,
                merge_id_col=cfg.dna_id_col,
                prefix="dna",
            )
            rna_meta = load_sequence_window_metadata(
                cfg.rna_seq_file,
                merge_id_col=cfg.rna_id_col,
                prefix="rna",
            )
            summary_input_df = attach_parent_metadata(
                out_df,
                dna_meta=dna_meta,
                rna_meta=rna_meta,
                dna_id_col=cfg.dna_id_col,
                rna_id_col=cfg.rna_id_col,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Skipping aggregated summary metadata enrichment: %s", exc)
            summary_input_df = out_df

    summary_df = aggregate_pair_predictions(
        summary_input_df,
        pair_id_col=cfg.pair_id_col,
        dna_id_col=cfg.dna_id_col,
        rna_id_col=cfg.rna_id_col,
        threshold=cfg.threshold,
    )
    if summary_df is not None and not summary_df.empty:
        summary_path = write_pair_summary(summary_df, output_path=output_path)
        LOGGER.info("Wrote aggregated summary to %s", summary_path)

    return {
        "mode": "pairs",
        "input_pairs": int(len(raw_pairs)),
        "pairs_after_missing": pairs_after_missing,
        "output_pairs": int(len(pairs)),
        "dropped_missing_embeddings": dropped_missing,
        "max_pairs": int(cfg.max_pairs) if cfg.max_pairs else None,
        "mean_prob": float(np.mean(probs)) if len(probs) else None,
        "min_prob": float(np.min(probs)) if len(probs) else None,
        "max_prob": float(np.max(probs)) if len(probs) else None,
        "summary_output": str(summary_path) if summary_path is not None else None,
        "summary": _build_summary_stats(summary_df),
    }


def _infer_all_pairs(
    cfg: InferConfig,
    *,
    dna_embeddings: np.ndarray,
    dna_ids: list[str],
    rna_embeddings: np.ndarray,
    rna_ids: list[str],
    model: torch.nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> dict:
    output_path = Path(cfg.output).expanduser()
    ext = infer_extension(output_path)
    if cfg.summary_only and not cfg.all_pairs:
        raise ValueError("--summary_only is only supported with --all_pairs.")
    if cfg.summary_only and not (cfg.dna_seq_file and cfg.rna_seq_file):
        raise ValueError(
            "--summary_only requires --dna_seq_file and --rna_seq_file so summary metadata can be restored."
        )
    if ext not in {".csv", ".tsv", ""}:
        raise ValueError(
            "all_pairs mode only supports CSV/TSV output for streaming writes."
        )

    n_dna = len(dna_ids)
    n_rna = len(rna_ids)
    total_pairs = n_dna * n_rna
    LOGGER.info(
        "Scoring all pairs: dna=%d rna=%d total=%d (blocks: dna=%d rna=%d).",
        n_dna,
        n_rna,
        total_pairs,
        cfg.dna_block_size,
        cfg.rna_block_size,
    )
    if cfg.summary_only:
        LOGGER.info("summary_only enabled: raw pair predictions will not be written.")

    header = True
    progress = None
    if cfg.progress:
        from tqdm.auto import tqdm

        progress = tqdm(total=total_pairs, desc="all pairs", unit="pair")

    dna_ids_arr = np.asarray(dna_ids, dtype=str)
    rna_ids_arr = np.asarray(rna_ids, dtype=str)
    sum_probs = 0.0
    min_prob = None
    max_prob = None
    processed = 0
    summary_states: dict[tuple[str, str], dict[str, Any]] = {}
    summary_enabled = False
    summary_path = None
    summary_df = None
    dna_meta_block_df = None
    rna_meta_block_df = None
    dna_window_counts: dict[str, int] = {}
    rna_window_counts: dict[str, int] = {}

    if cfg.dna_seq_file and cfg.rna_seq_file:
        try:
            dna_meta_block_df = load_sequence_window_metadata(
                cfg.dna_seq_file,
                merge_id_col=cfg.dna_id_col,
                prefix="dna",
            ).set_index(cfg.dna_id_col, drop=False)
            rna_meta_block_df = load_sequence_window_metadata(
                cfg.rna_seq_file,
                merge_id_col=cfg.rna_id_col,
                prefix="rna",
            ).set_index(cfg.rna_id_col, drop=False)
            dna_window_counts = (
                dna_meta_block_df.groupby(DNA_PARENT_ID_COLUMN)[cfg.dna_id_col]
                .nunique()
                .astype(int)
                .to_dict()
            )
            rna_window_counts = (
                rna_meta_block_df.groupby(RNA_PARENT_ID_COLUMN)[cfg.rna_id_col]
                .nunique()
                .astype(int)
                .to_dict()
            )
            summary_enabled = True
            LOGGER.info(
                "Aggregated summary enabled using %s and %s.",
                cfg.dna_seq_file,
                cfg.rna_seq_file,
            )
        except Exception as exc:  # noqa: BLE001
            if cfg.summary_only:
                raise ValueError(
                    "Unable to initialize summary aggregation for --summary_only."
                ) from exc
            LOGGER.warning("Skipping aggregated summary for all_pairs: %s", exc)

    if cfg.summary_only:
        if (
            not summary_enabled
            or dna_meta_block_df is None
            or rna_meta_block_df is None
        ):
            raise ValueError("Summary aggregation is required for --summary_only.")

        dna_parent_blocks = _build_parent_complete_index_blocks(
            dna_meta_block_df,
            ordered_ids=dna_ids_arr,
            id_col=cfg.dna_id_col,
            parent_col=DNA_PARENT_ID_COLUMN,
            max_windows=cfg.dna_block_size,
        )
        rna_parent_blocks = _build_parent_complete_index_blocks(
            rna_meta_block_df,
            ordered_ids=rna_ids_arr,
            id_col=cfg.rna_id_col,
            parent_col=RNA_PARENT_ID_COLUMN,
            max_windows=cfg.rna_block_size,
        )
        LOGGER.info(
            "Streaming summary in parent-complete blocks: dna_blocks=%d rna_blocks=%d.",
            len(dna_parent_blocks),
            len(rna_parent_blocks),
        )

        summary_path = resolve_pair_summary_output_path(output_path)
        if summary_path.exists():
            summary_path.unlink()
        summary_header = True
        summary_stats: dict[str, Any] = {}

        for dna_idx_block in dna_parent_blocks:
            dna_id_block = dna_ids_arr[dna_idx_block]
            dna_meta_slice = dna_meta_block_df.reindex(dna_id_block)
            for rna_idx_block in rna_parent_blocks:
                rna_id_block = rna_ids_arr[rna_idx_block]
                rna_meta_slice = rna_meta_block_df.reindex(rna_id_block)

                dna_idx_pairs = np.repeat(dna_idx_block, len(rna_idx_block))
                rna_idx_pairs = np.tile(rna_idx_block, len(dna_idx_block))

                features = build_pair_features(
                    dna_embeddings,
                    rna_embeddings,
                    dna_idx_pairs,
                    rna_idx_pairs,
                    mode=cfg.feature_mode,
                    chunk_size=cfg.chunk_size,
                    show_progress=False,
                    num_workers=cfg.num_workers,
                )
                X = normalize_embeddings(features.astype(np.float32), mean, std)
                ds = TensorDataset(torch.from_numpy(X).float())
                loader = DataLoader(
                    ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False
                )
                probs = predict_probabilities(
                    model, loader, device, show_progress=False
                )
                preds = (probs >= cfg.threshold).astype(int)

                dna_id_pairs = np.repeat(dna_id_block, len(rna_idx_block))
                rna_id_pairs = np.tile(rna_id_block, len(dna_idx_block))
                pair_ids = np.char.add(
                    np.char.add(dna_id_pairs, "__"), rna_id_pairs
                )

                processed += len(probs)
                if len(probs):
                    sum_probs += float(np.sum(probs))
                    block_min = float(np.min(probs))
                    block_max = float(np.max(probs))
                    min_prob = block_min if min_prob is None else min(min_prob, block_min)
                    max_prob = block_max if max_prob is None else max(max_prob, block_max)
                if progress is not None:
                    progress.update(len(dna_id_pairs))

                block_summary_df = pd.DataFrame(
                    {
                        "pair_id": pair_ids.astype(str),
                        "dna_id": dna_id_pairs.astype(str),
                        "rna_id": rna_id_pairs.astype(str),
                        "prob": probs.astype(np.float32),
                        "pred": preds.astype(int),
                        DNA_PARENT_ID_COLUMN: np.repeat(
                            dna_meta_slice[DNA_PARENT_ID_COLUMN]
                            .astype(str)
                            .to_numpy(),
                            len(rna_idx_block),
                        ),
                        RNA_PARENT_ID_COLUMN: np.tile(
                            rna_meta_slice[RNA_PARENT_ID_COLUMN]
                            .astype(str)
                            .to_numpy(),
                            len(dna_idx_block),
                        ),
                    }
                )
                for src, dst in (
                    ("dna_window_index", DNA_WINDOW_INDEX_COLUMN),
                    ("dna_window_start", "dna_window_start"),
                    ("dna_window_end", "dna_window_end"),
                ):
                    if src in dna_meta_slice.columns:
                        block_summary_df[dst] = np.repeat(
                            dna_meta_slice[src].to_numpy(), len(rna_idx_block)
                        )
                for src, dst in (
                    ("rna_window_index", RNA_WINDOW_INDEX_COLUMN),
                    ("rna_window_start", "rna_window_start"),
                    ("rna_window_end", "rna_window_end"),
                ):
                    if src in rna_meta_slice.columns:
                        block_summary_df[dst] = np.tile(
                            rna_meta_slice[src].to_numpy(), len(dna_idx_block)
                        )

                block_states: dict[tuple[str, str], dict[str, Any]] = {}
                _update_group_summary(block_states, block_summary_df)
                block_output_df = _finalize_group_summary(
                    block_states,
                    dna_window_counts=dna_window_counts,
                    rna_window_counts=rna_window_counts,
                    threshold=cfg.threshold,
                )
                if block_output_df is not None and not block_output_df.empty:
                    _append_summary_output(
                        block_output_df,
                        summary_path,
                        header=summary_header,
                    )
                    summary_header = False
                    _accumulate_summary_stats(summary_stats, block_output_df)

        if progress is not None:
            progress.close()
        if summary_header:
            raise ValueError("No summary file was produced for --summary_only.")
        mean_prob = (sum_probs / processed) if processed else None
        LOGGER.info("Wrote streaming aggregated summary to %s", summary_path)
        return {
            "mode": "all_pairs",
            "dna_count": int(n_dna),
            "rna_count": int(n_rna),
            "total_pairs": int(total_pairs),
            "processed_pairs": int(processed),
            "mean_prob": float(mean_prob) if mean_prob is not None else None,
            "min_prob": float(min_prob) if min_prob is not None else None,
            "max_prob": float(max_prob) if max_prob is not None else None,
            "summary_only": True,
            "summary_output": str(summary_path),
            "summary": _finalize_summary_stats(summary_stats),
        }

    for dna_start in range(0, n_dna, cfg.dna_block_size):
        dna_end = min(dna_start + cfg.dna_block_size, n_dna)
        dna_idx_block = np.arange(dna_start, dna_end, dtype=int)
        dna_id_block = dna_ids_arr[dna_start:dna_end]
        for rna_start in range(0, n_rna, cfg.rna_block_size):
            rna_end = min(rna_start + cfg.rna_block_size, n_rna)
            rna_idx_block = np.arange(rna_start, rna_end, dtype=int)
            rna_id_block = rna_ids_arr[rna_start:rna_end]

            dna_idx_pairs = np.repeat(dna_idx_block, len(rna_idx_block))
            rna_idx_pairs = np.tile(rna_idx_block, len(dna_idx_block))

            features = build_pair_features(
                dna_embeddings,
                rna_embeddings,
                dna_idx_pairs,
                rna_idx_pairs,
                mode=cfg.feature_mode,
                chunk_size=cfg.chunk_size,
                show_progress=False,
                num_workers=cfg.num_workers,
            )
            X = normalize_embeddings(features.astype(np.float32), mean, std)
            ds = TensorDataset(torch.from_numpy(X).float())
            loader = DataLoader(
                ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False
            )
            probs = predict_probabilities(
                model, loader, device, show_progress=False
            )
            preds = (probs >= cfg.threshold).astype(int)

            dna_id_pairs = np.repeat(dna_id_block, len(rna_idx_block))
            rna_id_pairs = np.tile(rna_id_block, len(dna_idx_block))
            pair_ids = np.char.add(
                np.char.add(dna_id_pairs, "__"), rna_id_pairs
            )

            if not cfg.summary_only:
                _append_predictions(
                    pair_ids=pair_ids,
                    dna_ids=dna_id_pairs,
                    rna_ids=rna_id_pairs,
                    probs=probs,
                    output=output_path,
                    threshold=cfg.threshold,
                    header=header,
                    pair_id_col=cfg.pair_id_col,
                    dna_id_col=cfg.dna_id_col,
                    rna_id_col=cfg.rna_id_col,
                )
                header = False
            processed += len(probs)
            if len(probs):
                sum_probs += float(np.sum(probs))
                block_min = float(np.min(probs))
                block_max = float(np.max(probs))
                min_prob = block_min if min_prob is None else min(min_prob, block_min)
                max_prob = block_max if max_prob is None else max(max_prob, block_max)
            if progress is not None:
                progress.update(len(dna_id_pairs))

            if summary_enabled and dna_meta_block_df is not None and rna_meta_block_df is not None:
                try:
                    dna_meta_slice = dna_meta_block_df.reindex(dna_id_block)
                    rna_meta_slice = rna_meta_block_df.reindex(rna_id_block)
                    if dna_meta_slice[DNA_PARENT_ID_COLUMN].isna().any():
                        missing_ids = dna_meta_slice.loc[
                            dna_meta_slice[DNA_PARENT_ID_COLUMN].isna(), cfg.dna_id_col
                        ].astype(str).tolist()
                        raise KeyError(
                            f"Missing DNA summary metadata for IDs: {missing_ids[:10]}"
                        )
                    if rna_meta_slice[RNA_PARENT_ID_COLUMN].isna().any():
                        missing_ids = rna_meta_slice.loc[
                            rna_meta_slice[RNA_PARENT_ID_COLUMN].isna(), cfg.rna_id_col
                        ].astype(str).tolist()
                        raise KeyError(
                            f"Missing RNA summary metadata for IDs: {missing_ids[:10]}"
                        )

                    block_summary_df = pd.DataFrame(
                        {
                            "pair_id": pair_ids.astype(str),
                            "dna_id": dna_id_pairs.astype(str),
                            "rna_id": rna_id_pairs.astype(str),
                            "prob": probs.astype(np.float32),
                            "pred": preds.astype(int),
                            DNA_PARENT_ID_COLUMN: np.repeat(
                                dna_meta_slice[DNA_PARENT_ID_COLUMN]
                                .astype(str)
                                .to_numpy(),
                                len(rna_idx_block),
                            ),
                            RNA_PARENT_ID_COLUMN: np.tile(
                                rna_meta_slice[RNA_PARENT_ID_COLUMN]
                                .astype(str)
                                .to_numpy(),
                                len(dna_idx_block),
                            ),
                        }
                    )
                    for src, dst in (
                        ("dna_window_index", DNA_WINDOW_INDEX_COLUMN),
                        ("dna_window_start", "dna_window_start"),
                        ("dna_window_end", "dna_window_end"),
                    ):
                        if src in dna_meta_slice.columns:
                            block_summary_df[dst] = np.repeat(
                                dna_meta_slice[src].to_numpy(), len(rna_idx_block)
                            )
                    for src, dst in (
                        ("rna_window_index", RNA_WINDOW_INDEX_COLUMN),
                        ("rna_window_start", "rna_window_start"),
                        ("rna_window_end", "rna_window_end"),
                    ):
                        if src in rna_meta_slice.columns:
                            block_summary_df[dst] = np.tile(
                                rna_meta_slice[src].to_numpy(), len(dna_idx_block)
                            )
                    _update_group_summary(summary_states, block_summary_df)
                except Exception as exc:  # noqa: BLE001
                    if cfg.summary_only:
                        raise ValueError(
                            "Summary aggregation failed during --summary_only run."
                        ) from exc
                    LOGGER.warning("Disabling aggregated summary for all_pairs: %s", exc)
                    summary_enabled = False
                    summary_states.clear()

    if progress is not None:
        progress.close()
    mean_prob = (sum_probs / processed) if processed else None
    if summary_enabled:
        summary_df = _finalize_group_summary(
            summary_states,
            dna_window_counts=dna_window_counts,
            rna_window_counts=rna_window_counts,
            threshold=cfg.threshold,
        )
        if summary_df is not None and not summary_df.empty:
            summary_path = write_pair_summary(summary_df, output_path=output_path)
            LOGGER.info("Wrote aggregated summary to %s", summary_path)
    if cfg.summary_only and summary_path is None:
        raise ValueError("No summary file was produced for --summary_only.")
    return {
        "mode": "all_pairs",
        "dna_count": int(n_dna),
        "rna_count": int(n_rna),
        "total_pairs": int(total_pairs),
        "processed_pairs": int(processed),
        "mean_prob": float(mean_prob) if mean_prob is not None else None,
        "min_prob": float(min_prob) if min_prob is not None else None,
        "max_prob": float(max_prob) if max_prob is not None else None,
        "summary_only": bool(cfg.summary_only),
        "summary_output": str(summary_path) if summary_path is not None else None,
        "summary": _build_summary_stats(summary_df),
    }


def main() -> None:
    cfg = _parse_args()
    output_path = Path(cfg.output).expanduser()
    log_path = output_path.with_suffix(output_path.suffix + ".log")
    _setup_logging(log_path)
    if cfg.summary_only and not cfg.all_pairs:
        raise ValueError("--summary_only is only supported with --all_pairs.")

    LOGGER.info("Loading DNA embeddings (npz may take time to decompress).")
    start = time.perf_counter()
    dna_embeddings, dna_ids = load_embeddings(Path(cfg.dna_embeddings).expanduser())
    elapsed = time.perf_counter() - start
    LOGGER.info("Loaded DNA embeddings in %.2fs.", elapsed)
    LOGGER.info("Loading RNA embeddings (npz may take time to decompress).")
    start = time.perf_counter()
    rna_embeddings, rna_ids = load_embeddings(Path(cfg.rna_embeddings).expanduser())
    elapsed = time.perf_counter() - start
    LOGGER.info("Loaded RNA embeddings in %.2fs.", elapsed)
    dna_embeddings = dna_embeddings.astype(np.float32, copy=False)
    rna_embeddings = rna_embeddings.astype(np.float32, copy=False)
    dna_ids = [str(x) for x in dna_ids]
    rna_ids = [str(x) for x in rna_ids]
    total_dna = len(dna_ids)
    total_rna = len(rna_ids)
    LOGGER.info("Loaded embeddings: dna=%d rna=%d", total_dna, total_rna)
    if cfg.all_pairs:
        if cfg.max_dna > 0:
            if cfg.max_dna < total_dna:
                LOGGER.info(
                    "Limiting DNA embeddings to first %d entries (from %d).",
                    cfg.max_dna,
                    total_dna,
                )
            dna_embeddings = dna_embeddings[: cfg.max_dna]
            dna_ids = dna_ids[: cfg.max_dna]
        if cfg.max_rna > 0:
            if cfg.max_rna < total_rna:
                LOGGER.info(
                    "Limiting RNA embeddings to first %d entries (from %d).",
                    cfg.max_rna,
                    total_rna,
                )
            rna_embeddings = rna_embeddings[: cfg.max_rna]
            rna_ids = rna_ids[: cfg.max_rna]
    elif cfg.max_dna > 0 or cfg.max_rna > 0:
        LOGGER.info("max_dna/max_rna ignored because --all_pairs is not set.")
    device = torch.device(
        cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    ckpt = load_classifier_checkpoint(Path(cfg.checkpoint).expanduser(), device)
    LOGGER.info("Loaded checkpoint: %s", Path(cfg.checkpoint).expanduser())
    input_dim = int(ckpt["input_dim"])
    training_cfg = ckpt.get("config", {}) or {}
    hidden_dims_raw = training_cfg.get("hidden_dims")
    if not hidden_dims_raw:
        raise KeyError(
            "Checkpoint config missing required 'hidden_dims'."
        )
    hidden_dims = [int(d) for d in hidden_dims_raw if int(d) > 0]
    if not hidden_dims:
        raise ValueError("Checkpoint hidden_dims must contain positive integers.")

    trained_mode = training_cfg.get("pair_feature_mode")
    if trained_mode and str(trained_mode) != str(cfg.feature_mode):
        LOGGER.warning(
            "Feature mode mismatch: checkpoint=%s current=%s",
            trained_mode,
            cfg.feature_mode,
        )

    model = LogisticMLP(input_dim, hidden_dims).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    LOGGER.info("Model ready: input_dim=%d device=%s", input_dim, device)

    mean = ckpt["feature_mean"].cpu().numpy().astype(np.float32)
    std = ckpt["feature_std"].cpu().numpy().astype(np.float32)

    LOGGER.info(
        "Starting inference: mode=%s output=%s",
        "all_pairs" if cfg.all_pairs else "pairs_file",
        output_path,
    )
    if cfg.all_pairs:
        stats = _infer_all_pairs(
            cfg,
            dna_embeddings=dna_embeddings,
            dna_ids=dna_ids,
            rna_embeddings=rna_embeddings,
            rna_ids=rna_ids,
            model=model,
            mean=mean,
            std=std,
            device=device,
        )
    else:
        if not cfg.pairs_file:
            raise ValueError("--pairs_file is required unless --all_pairs is set.")
        stats = _infer_from_pairs(
            cfg,
            dna_embeddings=dna_embeddings,
            dna_ids=dna_ids,
            rna_embeddings=rna_embeddings,
            rna_ids=rna_ids,
            model=model,
            mean=mean,
            std=std,
            device=device,
            output_path=output_path,
        )
    LOGGER.info("Inference completed.")

    meta = {
        "output": str(output_path),
        "output_dir": str(output_path.parent),
        "pairs_file": str(cfg.pairs_file) if cfg.pairs_file else None,
        "dna_embeddings": str(Path(cfg.dna_embeddings).expanduser()),
        "rna_embeddings": str(Path(cfg.rna_embeddings).expanduser()),
        "dna_seq_file": str(Path(cfg.dna_seq_file).expanduser())
        if cfg.dna_seq_file
        else None,
        "rna_seq_file": str(Path(cfg.rna_seq_file).expanduser())
        if cfg.rna_seq_file
        else None,
        "checkpoint": str(Path(cfg.checkpoint).expanduser()),
        "feature_mode": str(cfg.feature_mode),
        "threshold": float(cfg.threshold),
        "batch_size": int(cfg.batch_size),
        "chunk_size": int(cfg.chunk_size),
        "num_workers": int(cfg.num_workers),
        "device": str(device),
        "all_pairs": bool(cfg.all_pairs),
        "summary_only": bool(cfg.summary_only),
        "raw_output_skipped": bool(cfg.summary_only),
        "max_pairs": int(cfg.max_pairs) if cfg.max_pairs else None,
        "max_dna": int(cfg.max_dna) if cfg.max_dna else None,
        "max_rna": int(cfg.max_rna) if cfg.max_rna else None,
        "dna_embeddings_count": int(total_dna),
        "rna_embeddings_count": int(total_rna),
        "dna_used_count": int(len(dna_ids)),
        "rna_used_count": int(len(rna_ids)),
        "stats": stats,
    }
    meta_path = _write_meta(output_path, meta)
    LOGGER.info("Metadata written to %s", meta_path)


if __name__ == "__main__":
    main()
