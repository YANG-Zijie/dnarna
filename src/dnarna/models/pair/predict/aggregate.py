from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd

from dnarna.models.pair.predict.utils import (
    CSV_EXTENSIONS,
    PARQUET_EXTENSIONS,
    infer_extension,
    load_table,
)

PAIR_GROUP_ID_COLUMN = "pair_group_id"
PAIR_PARENT_ID_COLUMN = "pair_parent_id"
DNA_PARENT_ID_COLUMN = "dna_parent_id"
RNA_PARENT_ID_COLUMN = "rna_parent_id"
DNA_WINDOW_INDEX_COLUMN = "dna_window_index"
RNA_WINDOW_INDEX_COLUMN = "rna_window_index"
DNA_WINDOW_START_COLUMN = "dna_window_start"
DNA_WINDOW_END_COLUMN = "dna_window_end"
RNA_WINDOW_START_COLUMN = "rna_window_start"
RNA_WINDOW_END_COLUMN = "rna_window_end"
WINDOW_PAIR_COUNT_COLUMN = "window_pair_count"
POSITIVE_WINDOW_PAIR_COUNT_COLUMN = "positive_window_pair_count"
POSITIVE_WINDOW_PAIR_FRACTION_COLUMN = "positive_window_pair_fraction"
DNA_WINDOW_COUNT_COLUMN = "dna_window_count"
RNA_WINDOW_COUNT_COLUMN = "rna_window_count"
THRESHOLD_COLUMN = "threshold"
PROB_MEAN_COLUMN = "prob_mean"
PROB_MAX_COLUMN = "prob_max"
COMBINED_SCORE_COLUMN = "combined_score"
COMBINED_PRED_COLUMN = "combined_pred"
BEST_WINDOW_PAIR_ID_COLUMN = "best_window_pair_id"
BEST_WINDOW_PROB_COLUMN = "best_window_prob"
BEST_DNA_ID_COLUMN = "best_dna_id"
BEST_RNA_ID_COLUMN = "best_rna_id"
BEST_DNA_WINDOW_INDEX_COLUMN = "best_dna_window_index"
BEST_RNA_WINDOW_INDEX_COLUMN = "best_rna_window_index"
BEST_DNA_WINDOW_START_COLUMN = "best_dna_window_start"
BEST_DNA_WINDOW_END_COLUMN = "best_dna_window_end"
BEST_RNA_WINDOW_START_COLUMN = "best_rna_window_start"
BEST_RNA_WINDOW_END_COLUMN = "best_rna_window_end"

LOGGER = logging.getLogger("pair_aggregate")
WINDOW_ID_SUFFIX_PATTERN = re.compile(r"_win_\d+$")


def _resolve_summary_output_path(output_path: Path) -> Path:
    name = output_path.name
    suffixes = output_path.suffixes
    if suffixes:
        combined_suffix = "".join(suffixes)
        stem = name[: -len(combined_suffix)]
        return output_path.with_name(f"{stem}.summary{combined_suffix}")
    return output_path.with_name(f"{name}.summary.csv")


def resolve_pair_summary_output_path(output_path: str | Path) -> Path:
    return _resolve_summary_output_path(Path(output_path).expanduser())


def _resolve_variant_output_path(output_path: Path, variant: str) -> Path:
    name = output_path.name
    suffix = output_path.suffix
    if suffix:
        stem = name[: -len(suffix)]
        return output_path.with_name(f"{stem}.{variant}{suffix}")
    return output_path.with_name(f"{name}.{variant}.csv")


def _noisy_or_probability(values: pd.Series) -> float:
    probs = np.clip(pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64), 0.0, 1.0)
    if probs.size == 0:
        return float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        log_not = np.log1p(-probs)
    return float(-np.expm1(np.sum(log_not)))


def iter_table_chunks(
    path: str | Path,
    *,
    chunksize: int | None,
    columns: list[str] | None = None,
) -> Iterable[pd.DataFrame]:
    table_path = Path(path).expanduser()
    ext = infer_extension(table_path)

    if ext in CSV_EXTENSIONS or not ext:
        sep = "\t" if ext.startswith(".tsv") else ","
        if chunksize is None or int(chunksize) <= 0:
            yield pd.read_csv(table_path, sep=sep, usecols=columns)
            return
        for chunk in pd.read_csv(
            table_path,
            sep=sep,
            usecols=columns,
            chunksize=int(chunksize),
        ):
            yield chunk
        return

    if ext in PARQUET_EXTENSIONS:
        if chunksize is None or int(chunksize) <= 0:
            yield pd.read_parquet(table_path, columns=columns)
            return
        try:
            import fastparquet
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Streaming parquet summary requires fastparquet to be installed."
            ) from exc
        parquet_file = fastparquet.ParquetFile(table_path)
        for chunk in parquet_file.iter_row_groups(columns=columns):
            yield chunk
        return

    raise ValueError(
        f"Unsupported input extension '{ext}'. "
        f"Supported: {sorted(PARQUET_EXTENSIONS | CSV_EXTENSIONS)}"
    )


def list_table_columns(path: str | Path) -> list[str]:
    table_path = Path(path).expanduser()
    ext = infer_extension(table_path)

    if ext in CSV_EXTENSIONS or not ext:
        sep = "\t" if ext.startswith(".tsv") else ","
        return pd.read_csv(table_path, sep=sep, nrows=0).columns.astype(str).tolist()

    if ext in PARQUET_EXTENSIONS:
        try:
            import fastparquet
        except ImportError:  # pragma: no cover
            return pd.read_parquet(table_path).columns.astype(str).tolist()
        parquet_file = fastparquet.ParquetFile(table_path)
        return [str(col) for col in parquet_file.columns]

    raise ValueError(
        f"Unsupported input extension '{ext}'. "
        f"Supported: {sorted(PARQUET_EXTENSIONS | CSV_EXTENSIONS)}"
    )


def _load_selected_columns(
    path: str | Path,
    *,
    columns: list[str],
) -> pd.DataFrame:
    table_path = Path(path).expanduser()
    ext = infer_extension(table_path)

    if ext in CSV_EXTENSIONS or not ext:
        sep = "\t" if ext.startswith(".tsv") else ","
        return pd.read_csv(table_path, sep=sep, usecols=columns)

    if ext in PARQUET_EXTENSIONS:
        return pd.read_parquet(table_path, columns=columns)

    raise ValueError(
        f"Unsupported input extension '{ext}'. "
        f"Supported: {sorted(PARQUET_EXTENSIONS | CSV_EXTENSIONS)}"
    )


def _resolve_group_columns(df: pd.DataFrame) -> list[str] | None:
    if PAIR_PARENT_ID_COLUMN in df.columns:
        group_cols = [PAIR_PARENT_ID_COLUMN]
        if DNA_PARENT_ID_COLUMN in df.columns:
            group_cols.append(DNA_PARENT_ID_COLUMN)
        if RNA_PARENT_ID_COLUMN in df.columns:
            group_cols.append(RNA_PARENT_ID_COLUMN)
        return group_cols
    if {DNA_PARENT_ID_COLUMN, RNA_PARENT_ID_COLUMN}.issubset(df.columns):
        return [DNA_PARENT_ID_COLUMN, RNA_PARENT_ID_COLUMN]
    return None


def _infer_parent_id_from_window_id(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return WINDOW_ID_SUFFIX_PATTERN.sub("", text)


def attach_inferred_parent_ids(
    df: pd.DataFrame,
    *,
    dna_id_col: str,
    rna_id_col: str,
) -> pd.DataFrame:
    enriched = df.copy()
    if DNA_PARENT_ID_COLUMN not in enriched.columns and dna_id_col in enriched.columns:
        enriched[DNA_PARENT_ID_COLUMN] = enriched[dna_id_col].map(
            _infer_parent_id_from_window_id
        )
    if RNA_PARENT_ID_COLUMN not in enriched.columns and rna_id_col in enriched.columns:
        enriched[RNA_PARENT_ID_COLUMN] = enriched[rna_id_col].map(
            _infer_parent_id_from_window_id
        )
    return enriched


def _attach_indexed_metadata(
    df: pd.DataFrame,
    *,
    id_col: str,
    indexed_meta: pd.DataFrame | None,
) -> pd.DataFrame:
    if indexed_meta is None:
        return df

    enriched = df.copy()
    enriched[id_col] = enriched[id_col].astype(str)
    added_cols = [
        col for col in indexed_meta.columns if col != id_col and col not in enriched.columns
    ]
    if not added_cols:
        return enriched

    for col in added_cols:
        enriched[col] = enriched[id_col].map(indexed_meta[col])
    missing = enriched[added_cols].isna().all(axis=1)
    if missing.any():
        missing_ids = enriched.loc[missing, id_col].astype(str).drop_duplicates().tolist()
        raise KeyError(
            f"Missing metadata for {len(missing_ids)} ids in column '{id_col}' "
            f"(first 10: {missing_ids[:10]})"
        )
    return enriched


def _default_prediction_columns(
    *,
    pair_id_col: str,
    dna_id_col: str,
    rna_id_col: str,
    include_pred: bool,
) -> list[str]:
    cols = [
        pair_id_col,
        dna_id_col,
        rna_id_col,
        "prob",
    ]
    if include_pred:
        cols.append("pred")
    cols.extend(
        [
            PAIR_PARENT_ID_COLUMN,
            DNA_PARENT_ID_COLUMN,
            RNA_PARENT_ID_COLUMN,
            DNA_WINDOW_INDEX_COLUMN,
            RNA_WINDOW_INDEX_COLUMN,
            DNA_WINDOW_START_COLUMN,
            DNA_WINDOW_END_COLUMN,
            RNA_WINDOW_START_COLUMN,
            RNA_WINDOW_END_COLUMN,
        ]
    )
    seen: set[str] = set()
    ordered: list[str] = []
    for col in cols:
        if col not in seen:
            seen.add(col)
            ordered.append(col)
    return ordered


def _normalize_thresholds(thresholds: Iterable[float] | None) -> list[float]:
    if thresholds is None:
        return []

    normalized: list[float] = []
    seen: set[float] = set()
    for value in thresholds:
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError(f"Invalid threshold value: {value}")
        if numeric < 0.0 or numeric > 1.0:
            raise ValueError(
                f"Threshold must be within [0, 1], got {numeric}"
            )
        if numeric not in seen:
            seen.add(numeric)
            normalized.append(numeric)
    return sorted(normalized)


def load_sequence_window_metadata(
    path: str | Path,
    *,
    merge_id_col: str,
    prefix: str,
) -> pd.DataFrame:
    seq_path = Path(path).expanduser()
    available_columns = set(list_table_columns(seq_path))
    if "id" not in available_columns:
        raise KeyError(f"Sequence file missing required column 'id': {seq_path}")
    selected_columns = [
        col
        for col in ("id", "parent_id", "window_index", "window_start", "window_end")
        if col in available_columns
    ]
    df = _load_selected_columns(seq_path, columns=selected_columns)

    meta = pd.DataFrame({merge_id_col: df["id"].astype(str)})
    parent_values = (
        df["parent_id"].astype(str)
        if "parent_id" in df.columns
        else df["id"].astype(str)
    )
    meta[f"{prefix}_parent_id"] = parent_values

    optional_columns = {
        "window_index": f"{prefix}_window_index",
        "window_start": f"{prefix}_window_start",
        "window_end": f"{prefix}_window_end",
    }
    for src, dst in optional_columns.items():
        if src in df.columns:
            meta[dst] = df[src]

    if meta[merge_id_col].duplicated().any():
        dup_ids = meta.loc[meta[merge_id_col].duplicated(), merge_id_col].astype(str).tolist()
        raise ValueError(
            f"Sequence file contains duplicated IDs for summary mapping: {dup_ids[:10]}"
        )
    return meta


def load_pair_window_metadata(
    path: str | Path,
    *,
    pair_id_col: str,
) -> pd.DataFrame:
    pair_path = Path(path).expanduser()
    available_columns = set(list_table_columns(pair_path))
    if pair_id_col not in available_columns:
        raise KeyError(
            f"Pair file missing required column '{pair_id_col}': {pair_path}"
        )

    candidate_cols = [
        PAIR_PARENT_ID_COLUMN,
        DNA_PARENT_ID_COLUMN,
        RNA_PARENT_ID_COLUMN,
        DNA_WINDOW_INDEX_COLUMN,
        RNA_WINDOW_INDEX_COLUMN,
        DNA_WINDOW_START_COLUMN,
        DNA_WINDOW_END_COLUMN,
        RNA_WINDOW_START_COLUMN,
        RNA_WINDOW_END_COLUMN,
    ]
    keep_cols = [pair_id_col] + [col for col in candidate_cols if col in available_columns]
    df = _load_selected_columns(pair_path, columns=keep_cols)
    meta = df[keep_cols].copy()
    meta[pair_id_col] = meta[pair_id_col].astype(str)
    if meta[pair_id_col].duplicated().any():
        dup_ids = meta.loc[meta[pair_id_col].duplicated(), pair_id_col].tolist()
        raise ValueError(
            f"Pair file contains duplicated pair ids for summary mapping: {dup_ids[:10]}"
        )
    return meta


def attach_pair_metadata(
    df: pd.DataFrame,
    *,
    pair_meta: pd.DataFrame | None,
    pair_id_col: str,
) -> pd.DataFrame:
    if pair_meta is None:
        return df

    enriched = df.copy()
    merge_cols = [pair_id_col] + [
        col for col in pair_meta.columns if col != pair_id_col and col not in enriched.columns
    ]
    if len(merge_cols) == 1:
        return enriched

    enriched[pair_id_col] = enriched[pair_id_col].astype(str)
    enriched = enriched.merge(pair_meta[merge_cols], on=pair_id_col, how="left")
    newly_added = [col for col in merge_cols if col != pair_id_col]
    if newly_added:
        missing = enriched[newly_added].isna().all(axis=1)
        if missing.any():
            missing_ids = (
                enriched.loc[missing, pair_id_col].astype(str).drop_duplicates().tolist()
            )
            raise KeyError(
                f"Missing pair metadata for {len(missing_ids)} pair ids "
                f"(first 10: {missing_ids[:10]})"
            )
    return enriched


def attach_parent_metadata(
    df: pd.DataFrame,
    *,
    dna_meta: pd.DataFrame | None,
    rna_meta: pd.DataFrame | None,
    dna_id_col: str,
    rna_id_col: str,
) -> pd.DataFrame:
    enriched = df.copy()
    if dna_meta is not None and DNA_PARENT_ID_COLUMN not in enriched.columns:
        enriched = enriched.merge(dna_meta, on=dna_id_col, how="left")
        missing = enriched[DNA_PARENT_ID_COLUMN].isna()
        if missing.any():
            missing_ids = (
                enriched.loc[missing, dna_id_col].astype(str).drop_duplicates().tolist()
            )
            raise KeyError(
                f"Missing DNA metadata for {len(missing_ids)} IDs (first 10: {missing_ids[:10]})"
            )

    if rna_meta is not None and RNA_PARENT_ID_COLUMN not in enriched.columns:
        enriched = enriched.merge(rna_meta, on=rna_id_col, how="left")
        missing = enriched[RNA_PARENT_ID_COLUMN].isna()
        if missing.any():
            missing_ids = (
                enriched.loc[missing, rna_id_col].astype(str).drop_duplicates().tolist()
            )
            raise KeyError(
                f"Missing RNA metadata for {len(missing_ids)} IDs (first 10: {missing_ids[:10]})"
            )

    return enriched


def _update_group_states(
    states: dict[tuple[str, ...], dict[str, Any]],
    block_df: pd.DataFrame,
    *,
    group_cols: list[str],
    pair_id_col: str,
    dna_id_col: str,
    rna_id_col: str,
    track_window_sets: bool,
    thresholds: list[float] | None = None,
) -> None:
    grouped = block_df.groupby(group_cols, sort=False, dropna=False)
    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        key = tuple(str(value) for value in group_key)
        state = states.setdefault(
            key,
            {
                "group_values": {
                    col: key[idx] for idx, col in enumerate(group_cols)
                },
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
                "dna_ids": set() if track_window_sets else None,
                "rna_ids": set() if track_window_sets else None,
                "positive_window_pair_count_by_threshold": (
                    {threshold: 0 for threshold in thresholds}
                    if thresholds
                    else None
                ),
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
        if thresholds:
            counts_by_threshold = state["positive_window_pair_count_by_threshold"]
            for threshold in thresholds:
                counts_by_threshold[threshold] += int(np.count_nonzero(probs >= threshold))

        if track_window_sets:
            state["dna_ids"].update(group[dna_id_col].astype(str).tolist())
            state["rna_ids"].update(group[rna_id_col].astype(str).tolist())

        best_idx = int(group["prob"].idxmax())
        best_row = group.loc[best_idx]
        best_prob = float(best_row["prob"])
        if best_prob > float(state["prob_max"]):
            state["prob_max"] = best_prob
            state["best_window_pair_id"] = str(best_row[pair_id_col])
            state["best_dna_id"] = str(best_row[dna_id_col])
            state["best_rna_id"] = str(best_row[rna_id_col])
            field_map = {
                DNA_WINDOW_INDEX_COLUMN: BEST_DNA_WINDOW_INDEX_COLUMN,
                RNA_WINDOW_INDEX_COLUMN: BEST_RNA_WINDOW_INDEX_COLUMN,
                DNA_WINDOW_START_COLUMN: BEST_DNA_WINDOW_START_COLUMN,
                DNA_WINDOW_END_COLUMN: BEST_DNA_WINDOW_END_COLUMN,
                RNA_WINDOW_START_COLUMN: BEST_RNA_WINDOW_START_COLUMN,
                RNA_WINDOW_END_COLUMN: BEST_RNA_WINDOW_END_COLUMN,
            }
            for src, dst in field_map.items():
                if src in group.columns:
                    state[dst] = best_row[src]


def _finalize_group_states(
    states: dict[tuple[str, ...], dict[str, Any]],
    *,
    threshold: float,
    group_cols: list[str],
    dna_window_counts: dict[str, int] | None = None,
    rna_window_counts: dict[str, int] | None = None,
) -> pd.DataFrame | None:
    if not states:
        return None

    records: list[dict[str, Any]] = []
    for state in states.values():
        group_values = state["group_values"]
        count = int(state["window_pair_count"])
        positive_count = int(state["positive_window_pair_count"])
        combined = float(-np.expm1(float(state["log_not_sum"]))) if count else float("nan")
        record: dict[str, Any] = {col: group_values[col] for col in group_cols}

        dna_parent_value = group_values.get(DNA_PARENT_ID_COLUMN)
        rna_parent_value = group_values.get(RNA_PARENT_ID_COLUMN)
        if dna_window_counts is not None and dna_parent_value is not None:
            dna_window_count = int(dna_window_counts.get(str(dna_parent_value), 0))
        else:
            dna_ids = state.get("dna_ids")
            dna_window_count = len(dna_ids) if dna_ids is not None else 0
        if rna_window_counts is not None and rna_parent_value is not None:
            rna_window_count = int(rna_window_counts.get(str(rna_parent_value), 0))
        else:
            rna_ids = state.get("rna_ids")
            rna_window_count = len(rna_ids) if rna_ids is not None else 0

        record.update(
            {
                DNA_WINDOW_COUNT_COLUMN: dna_window_count,
                RNA_WINDOW_COUNT_COLUMN: rna_window_count,
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

        if PAIR_PARENT_ID_COLUMN in group_values:
            record[PAIR_GROUP_ID_COLUMN] = str(group_values[PAIR_PARENT_ID_COLUMN])
        elif dna_parent_value is not None and rna_parent_value is not None:
            record[PAIR_GROUP_ID_COLUMN] = f"{dna_parent_value}__{rna_parent_value}"
        records.append(record)

    summary_df = pd.DataFrame.from_records(records)
    preferred = [
        PAIR_GROUP_ID_COLUMN,
        PAIR_PARENT_ID_COLUMN,
        DNA_PARENT_ID_COLUMN,
        RNA_PARENT_ID_COLUMN,
        DNA_WINDOW_COUNT_COLUMN,
        RNA_WINDOW_COUNT_COLUMN,
        WINDOW_PAIR_COUNT_COLUMN,
        POSITIVE_WINDOW_PAIR_COUNT_COLUMN,
        POSITIVE_WINDOW_PAIR_FRACTION_COLUMN,
        PROB_MEAN_COLUMN,
        PROB_MAX_COLUMN,
        COMBINED_SCORE_COLUMN,
        COMBINED_PRED_COLUMN,
        BEST_WINDOW_PAIR_ID_COLUMN,
        BEST_WINDOW_PROB_COLUMN,
        BEST_DNA_ID_COLUMN,
        BEST_RNA_ID_COLUMN,
        BEST_DNA_WINDOW_INDEX_COLUMN,
        BEST_RNA_WINDOW_INDEX_COLUMN,
        BEST_DNA_WINDOW_START_COLUMN,
        BEST_DNA_WINDOW_END_COLUMN,
        BEST_RNA_WINDOW_START_COLUMN,
        BEST_RNA_WINDOW_END_COLUMN,
    ]
    ordered = [col for col in preferred if col in summary_df.columns]
    remaining = [col for col in summary_df.columns if col not in ordered]
    summary_df = summary_df[ordered + remaining]
    summary_df = summary_df.sort_values(
        [COMBINED_SCORE_COLUMN, PROB_MAX_COLUMN, PROB_MEAN_COLUMN],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return summary_df


def _finalize_group_threshold_states(
    states: dict[tuple[str, ...], dict[str, Any]],
    *,
    thresholds: list[float],
    group_cols: list[str],
) -> pd.DataFrame | None:
    if not states or not thresholds:
        return None

    records: list[dict[str, Any]] = []
    for state in states.values():
        group_values = state["group_values"]
        count = int(state["window_pair_count"])
        positive_counts = state.get("positive_window_pair_count_by_threshold") or {}
        dna_parent_value = group_values.get(DNA_PARENT_ID_COLUMN)
        rna_parent_value = group_values.get(RNA_PARENT_ID_COLUMN)
        if PAIR_PARENT_ID_COLUMN in group_values:
            pair_group_id = str(group_values[PAIR_PARENT_ID_COLUMN])
        elif dna_parent_value is not None and rna_parent_value is not None:
            pair_group_id = f"{dna_parent_value}__{rna_parent_value}"
        else:
            pair_group_id = None

        for threshold in thresholds:
            positive_count = int(positive_counts.get(threshold, 0))
            record: dict[str, Any] = {col: group_values[col] for col in group_cols}
            record.update(
                {
                    PAIR_GROUP_ID_COLUMN: pair_group_id,
                    THRESHOLD_COLUMN: float(threshold),
                    WINDOW_PAIR_COUNT_COLUMN: count,
                    POSITIVE_WINDOW_PAIR_COUNT_COLUMN: positive_count,
                    POSITIVE_WINDOW_PAIR_FRACTION_COLUMN: (
                        positive_count / count if count else np.nan
                    ),
                }
            )
            records.append(record)

    threshold_df = pd.DataFrame.from_records(records)
    preferred = [
        PAIR_GROUP_ID_COLUMN,
        PAIR_PARENT_ID_COLUMN,
        DNA_PARENT_ID_COLUMN,
        RNA_PARENT_ID_COLUMN,
        THRESHOLD_COLUMN,
        WINDOW_PAIR_COUNT_COLUMN,
        POSITIVE_WINDOW_PAIR_COUNT_COLUMN,
        POSITIVE_WINDOW_PAIR_FRACTION_COLUMN,
    ]
    ordered = [col for col in preferred if col in threshold_df.columns]
    remaining = [col for col in threshold_df.columns if col not in ordered]
    threshold_df = threshold_df[ordered + remaining]
    threshold_df = threshold_df.sort_values(
        [THRESHOLD_COLUMN, PAIR_GROUP_ID_COLUMN],
        ascending=[True, True],
    ).reset_index(drop=True)
    return threshold_df


def _prepare_prediction_summary_input(
    df: pd.DataFrame,
    *,
    pair_id_col: str,
    dna_id_col: str,
    rna_id_col: str,
    threshold: float,
) -> tuple[pd.DataFrame | None, list[str] | None]:
    if "prob" not in df.columns:
        raise KeyError("Prediction dataframe missing required column 'prob'.")

    work = df.copy()
    work["prob"] = pd.to_numeric(work["prob"], errors="coerce")
    work = work.dropna(subset=["prob"]).copy()
    if work.empty:
        return None, None

    work = attach_inferred_parent_ids(
        work,
        dna_id_col=dna_id_col,
        rna_id_col=rna_id_col,
    )

    if "pred" in work.columns:
        work["pred"] = pd.to_numeric(work["pred"], errors="coerce").fillna(0).astype(int)
    else:
        work["pred"] = (work["prob"] >= float(threshold)).astype(int)

    group_cols = _resolve_group_columns(work)
    if group_cols is None:
        return None, None

    if PAIR_PARENT_ID_COLUMN in work.columns:
        work[PAIR_GROUP_ID_COLUMN] = work[PAIR_PARENT_ID_COLUMN].astype(str)
    else:
        work[PAIR_GROUP_ID_COLUMN] = (
            work[DNA_PARENT_ID_COLUMN].astype(str)
            + "__"
            + work[RNA_PARENT_ID_COLUMN].astype(str)
        )
    return work, group_cols


def aggregate_pair_predictions(
    df: pd.DataFrame,
    *,
    pair_id_col: str,
    dna_id_col: str,
    rna_id_col: str,
    threshold: float,
) -> pd.DataFrame | None:
    work, group_cols = _prepare_prediction_summary_input(
        df,
        pair_id_col=pair_id_col,
        dna_id_col=dna_id_col,
        rna_id_col=rna_id_col,
        threshold=threshold,
    )
    if work is None or group_cols is None:
        return None

    grouped = work.groupby(group_cols, sort=False, dropna=False)
    summary = grouped.agg(
        **{
            WINDOW_PAIR_COUNT_COLUMN: ("prob", "size"),
            POSITIVE_WINDOW_PAIR_COUNT_COLUMN: ("pred", "sum"),
            DNA_WINDOW_COUNT_COLUMN: (dna_id_col, "nunique"),
            RNA_WINDOW_COUNT_COLUMN: (rna_id_col, "nunique"),
            PROB_MEAN_COLUMN: ("prob", "mean"),
            PROB_MAX_COLUMN: ("prob", "max"),
        }
    ).reset_index()

    summary[POSITIVE_WINDOW_PAIR_FRACTION_COLUMN] = (
        summary[POSITIVE_WINDOW_PAIR_COUNT_COLUMN] / summary[WINDOW_PAIR_COUNT_COLUMN]
    )
    combined_scores = (
        grouped["prob"].apply(_noisy_or_probability).reset_index(name=COMBINED_SCORE_COLUMN)
    )
    summary = summary.merge(combined_scores, on=group_cols, how="left")
    summary[COMBINED_SCORE_COLUMN] = summary[COMBINED_SCORE_COLUMN].astype(np.float32)
    summary[COMBINED_PRED_COLUMN] = (
        summary[COMBINED_SCORE_COLUMN] >= float(threshold)
    ).astype(int)

    best_idx = grouped["prob"].idxmax()
    best_rows = work.loc[best_idx].copy()
    best_rows = best_rows[group_cols + [
        pair_id_col,
        dna_id_col,
        rna_id_col,
        "prob",
        *[
            col
            for col in (
                DNA_WINDOW_INDEX_COLUMN,
                RNA_WINDOW_INDEX_COLUMN,
                DNA_WINDOW_START_COLUMN,
                DNA_WINDOW_END_COLUMN,
                RNA_WINDOW_START_COLUMN,
                RNA_WINDOW_END_COLUMN,
            )
            if col in best_rows.columns
        ],
    ]]
    rename_map = {
        pair_id_col: BEST_WINDOW_PAIR_ID_COLUMN,
        "prob": BEST_WINDOW_PROB_COLUMN,
        dna_id_col: BEST_DNA_ID_COLUMN,
        rna_id_col: BEST_RNA_ID_COLUMN,
        DNA_WINDOW_INDEX_COLUMN: BEST_DNA_WINDOW_INDEX_COLUMN,
        RNA_WINDOW_INDEX_COLUMN: BEST_RNA_WINDOW_INDEX_COLUMN,
        DNA_WINDOW_START_COLUMN: BEST_DNA_WINDOW_START_COLUMN,
        DNA_WINDOW_END_COLUMN: BEST_DNA_WINDOW_END_COLUMN,
        RNA_WINDOW_START_COLUMN: BEST_RNA_WINDOW_START_COLUMN,
        RNA_WINDOW_END_COLUMN: BEST_RNA_WINDOW_END_COLUMN,
    }
    best_rows = best_rows.rename(columns=rename_map)
    summary = summary.merge(best_rows, on=group_cols, how="left")

    if PAIR_PARENT_ID_COLUMN in summary.columns:
        summary[PAIR_GROUP_ID_COLUMN] = summary[PAIR_PARENT_ID_COLUMN].astype(str)
    else:
        summary[PAIR_GROUP_ID_COLUMN] = (
            summary[DNA_PARENT_ID_COLUMN].astype(str)
            + "__"
            + summary[RNA_PARENT_ID_COLUMN].astype(str)
        )

    preferred = [
        PAIR_GROUP_ID_COLUMN,
        PAIR_PARENT_ID_COLUMN,
        DNA_PARENT_ID_COLUMN,
        RNA_PARENT_ID_COLUMN,
        DNA_WINDOW_COUNT_COLUMN,
        RNA_WINDOW_COUNT_COLUMN,
        WINDOW_PAIR_COUNT_COLUMN,
        POSITIVE_WINDOW_PAIR_COUNT_COLUMN,
        POSITIVE_WINDOW_PAIR_FRACTION_COLUMN,
        PROB_MEAN_COLUMN,
        PROB_MAX_COLUMN,
        COMBINED_SCORE_COLUMN,
        COMBINED_PRED_COLUMN,
        BEST_WINDOW_PAIR_ID_COLUMN,
        BEST_WINDOW_PROB_COLUMN,
        BEST_DNA_ID_COLUMN,
        BEST_RNA_ID_COLUMN,
        BEST_DNA_WINDOW_INDEX_COLUMN,
        BEST_RNA_WINDOW_INDEX_COLUMN,
        BEST_DNA_WINDOW_START_COLUMN,
        BEST_DNA_WINDOW_END_COLUMN,
        BEST_RNA_WINDOW_START_COLUMN,
        BEST_RNA_WINDOW_END_COLUMN,
    ]
    ordered = [col for col in preferred if col in summary.columns]
    remaining = [col for col in summary.columns if col not in ordered]
    summary = summary[ordered + remaining]
    summary = summary.sort_values(
        [COMBINED_SCORE_COLUMN, PROB_MAX_COLUMN, PROB_MEAN_COLUMN],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return summary


def aggregate_pair_predictions_by_threshold(
    df: pd.DataFrame,
    *,
    pair_id_col: str,
    dna_id_col: str,
    rna_id_col: str,
    threshold: float,
    thresholds: Iterable[float],
) -> pd.DataFrame | None:
    normalized_thresholds = _normalize_thresholds(thresholds)
    if not normalized_thresholds:
        return None

    work, group_cols = _prepare_prediction_summary_input(
        df,
        pair_id_col=pair_id_col,
        dna_id_col=dna_id_col,
        rna_id_col=rna_id_col,
        threshold=threshold,
    )
    if work is None or group_cols is None:
        return None

    states: dict[tuple[str, ...], dict[str, Any]] = {}
    _update_group_states(
        states,
        work,
        group_cols=group_cols,
        pair_id_col=pair_id_col,
        dna_id_col=dna_id_col,
        rna_id_col=rna_id_col,
        track_window_sets=False,
        thresholds=normalized_thresholds,
    )
    return _finalize_group_threshold_states(
        states,
        thresholds=normalized_thresholds,
        group_cols=group_cols,
    )


def _write_summary_dataframe(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    elif suffix == ".tsv":
        df.to_csv(output_path, index=False, sep="\t")
    else:
        df.to_csv(output_path, index=False)
    return output_path


def write_pair_summary(
    df: pd.DataFrame,
    *,
    output_path: str | Path,
    add_summary_suffix: bool = True,
) -> Path:
    raw_output_path = Path(output_path).expanduser()
    summary_path = (
        resolve_pair_summary_output_path(raw_output_path)
        if add_summary_suffix
        else raw_output_path
    )
    return _write_summary_dataframe(df, summary_path)


def write_pair_threshold_summary(
    df: pd.DataFrame,
    *,
    output_path: str | Path,
    add_summary_suffix: bool = True,
) -> Path:
    raw_output_path = Path(output_path).expanduser()
    base_output_path = (
        _resolve_summary_output_path(raw_output_path)
        if add_summary_suffix
        else raw_output_path
    )
    threshold_path = _resolve_variant_output_path(base_output_path, "by_threshold")
    return _write_summary_dataframe(df, threshold_path)


def _summarize_pair_predictions_impl(
    *,
    predictions_path: str | Path,
    output_path: str | Path | None = None,
    pairs_file: str | Path | None = None,
    dna_seq_file: str | Path | None = None,
    rna_seq_file: str | Path | None = None,
    pair_id_col: str = "pair_id",
    dna_id_col: str = "dna_id",
    rna_id_col: str = "rna_id",
    threshold: float = 0.5,
    ignore_pred_col: bool = False,
    chunksize: int | None = 200_000,
    thresholds: Iterable[float] | None = None,
) -> tuple[pd.DataFrame, Path, pd.DataFrame | None, Path | None]:
    normalized_thresholds = _normalize_thresholds(thresholds)
    predictions_path = Path(predictions_path).expanduser()
    if chunksize is not None and int(chunksize) <= 0:
        chunksize = None

    threshold_summary_df: pd.DataFrame | None = None
    threshold_summary_path: Path | None = None

    if chunksize is None:
        pred_df = load_table(predictions_path)

        required_cols = {pair_id_col, dna_id_col, rna_id_col, "prob"} - set(pred_df.columns)
        if required_cols:
            raise KeyError(
                f"Predictions file missing columns: {sorted(required_cols)}. "
                f"Available: {list(pred_df.columns)}"
            )

        pred_df = pred_df.copy()
        pred_df[pair_id_col] = pred_df[pair_id_col].astype(str)
        pred_df[dna_id_col] = pred_df[dna_id_col].astype(str)
        pred_df[rna_id_col] = pred_df[rna_id_col].astype(str)
        if ignore_pred_col and "pred" in pred_df.columns:
            pred_df = pred_df.drop(columns=["pred"])

        if pairs_file:
            pair_meta = load_pair_window_metadata(pairs_file, pair_id_col=pair_id_col)
            pred_df = attach_pair_metadata(
                pred_df,
                pair_meta=pair_meta,
                pair_id_col=pair_id_col,
            )

        dna_meta = (
            load_sequence_window_metadata(
                dna_seq_file,
                merge_id_col=dna_id_col,
                prefix="dna",
            )
            if dna_seq_file
            else None
        )
        rna_meta = (
            load_sequence_window_metadata(
                rna_seq_file,
                merge_id_col=rna_id_col,
                prefix="rna",
            )
            if rna_seq_file
            else None
        )
        pred_df = attach_parent_metadata(
            pred_df,
            dna_meta=dna_meta,
            rna_meta=rna_meta,
            dna_id_col=dna_id_col,
            rna_id_col=rna_id_col,
        )

        summary_df = aggregate_pair_predictions(
            pred_df,
            pair_id_col=pair_id_col,
            dna_id_col=dna_id_col,
            rna_id_col=rna_id_col,
            threshold=threshold,
        )
        if summary_df is None or summary_df.empty:
            raise ValueError(
                "Unable to build summary from the provided predictions. "
                "Provide window metadata via --pairs_file and/or --dna_seq_file/--rna_seq_file."
            )

        if normalized_thresholds:
            threshold_summary_df = aggregate_pair_predictions_by_threshold(
                pred_df,
                pair_id_col=pair_id_col,
                dna_id_col=dna_id_col,
                rna_id_col=rna_id_col,
                threshold=threshold,
                thresholds=normalized_thresholds,
            )

        final_output_path = output_path or predictions_path
        summary_path = write_pair_summary(
            summary_df,
            output_path=final_output_path,
            add_summary_suffix=output_path is None,
        )
        if threshold_summary_df is not None and not threshold_summary_df.empty:
            threshold_summary_path = write_pair_threshold_summary(
                threshold_summary_df,
                output_path=final_output_path,
                add_summary_suffix=output_path is None,
            )
        return summary_df, summary_path, threshold_summary_df, threshold_summary_path

    pair_meta_indexed: pd.DataFrame | None = None
    if pairs_file:
        LOGGER.warning(
            "Loading pairs_file into memory to restore pair metadata. "
            "For very large jobs, prefer --dna_seq_file/--rna_seq_file or predictions "
            "that already contain parent columns."
        )
        pair_meta_indexed = load_pair_window_metadata(
            pairs_file,
            pair_id_col=pair_id_col,
        ).set_index(pair_id_col, drop=False)

    dna_meta_indexed: pd.DataFrame | None = None
    rna_meta_indexed: pd.DataFrame | None = None
    dna_window_counts: dict[str, int] | None = None
    rna_window_counts: dict[str, int] | None = None
    if dna_seq_file:
        dna_meta_indexed = load_sequence_window_metadata(
            dna_seq_file,
            merge_id_col=dna_id_col,
            prefix="dna",
        ).set_index(dna_id_col, drop=False)
        dna_window_counts = (
            dna_meta_indexed.groupby(DNA_PARENT_ID_COLUMN)[dna_id_col]
            .nunique()
            .astype(int)
            .to_dict()
        )
    if rna_seq_file:
        rna_meta_indexed = load_sequence_window_metadata(
            rna_seq_file,
            merge_id_col=rna_id_col,
            prefix="rna",
        ).set_index(rna_id_col, drop=False)
        rna_window_counts = (
            rna_meta_indexed.groupby(RNA_PARENT_ID_COLUMN)[rna_id_col]
            .nunique()
            .astype(int)
            .to_dict()
        )

    available_columns = set(list_table_columns(predictions_path))
    chunk_columns = _default_prediction_columns(
        pair_id_col=pair_id_col,
        dna_id_col=dna_id_col,
        rna_id_col=rna_id_col,
        include_pred=not ignore_pred_col,
    )
    chunk_columns = [col for col in chunk_columns if col in available_columns]
    states: dict[tuple[str, ...], dict[str, Any]] = {}
    resolved_group_cols: list[str] | None = None
    processed_rows = 0

    for chunk in iter_table_chunks(
        predictions_path,
        chunksize=chunksize,
        columns=chunk_columns,
    ):
        required_cols = {pair_id_col, dna_id_col, rna_id_col, "prob"} - set(chunk.columns)
        if required_cols:
            raise KeyError(
                f"Predictions file missing columns: {sorted(required_cols)}. "
                f"Available: {list(chunk.columns)}"
            )

        chunk = chunk.copy()
        chunk[pair_id_col] = chunk[pair_id_col].astype(str)
        chunk[dna_id_col] = chunk[dna_id_col].astype(str)
        chunk[rna_id_col] = chunk[rna_id_col].astype(str)
        chunk["prob"] = pd.to_numeric(chunk["prob"], errors="coerce")
        chunk = chunk.dropna(subset=["prob"]).copy()
        if chunk.empty:
            continue

        if ignore_pred_col or "pred" not in chunk.columns:
            chunk["pred"] = (chunk["prob"] >= float(threshold)).astype(int)
        else:
            chunk["pred"] = (
                pd.to_numeric(chunk["pred"], errors="coerce").fillna(0).astype(int)
            )

        chunk = _attach_indexed_metadata(
            chunk,
            id_col=pair_id_col,
            indexed_meta=pair_meta_indexed,
        )
        chunk = _attach_indexed_metadata(
            chunk,
            id_col=dna_id_col,
            indexed_meta=dna_meta_indexed,
        )
        chunk = _attach_indexed_metadata(
            chunk,
            id_col=rna_id_col,
            indexed_meta=rna_meta_indexed,
        )
        chunk = attach_inferred_parent_ids(
            chunk,
            dna_id_col=dna_id_col,
            rna_id_col=rna_id_col,
        )

        group_cols = _resolve_group_columns(chunk)
        if group_cols is None:
            raise ValueError(
                "Unable to build summary from the provided predictions. "
                "Provide window metadata via --pairs_file and/or --dna_seq_file/--rna_seq_file."
            )
        if resolved_group_cols is None:
            resolved_group_cols = group_cols
        elif group_cols != resolved_group_cols:
            raise ValueError(
                f"Inconsistent grouping columns across chunks: "
                f"{resolved_group_cols} vs {group_cols}"
            )

        track_window_sets = dna_window_counts is None or rna_window_counts is None
        _update_group_states(
            states,
            chunk,
            group_cols=resolved_group_cols,
            pair_id_col=pair_id_col,
            dna_id_col=dna_id_col,
            rna_id_col=rna_id_col,
            track_window_sets=track_window_sets,
            thresholds=normalized_thresholds,
        )
        processed_rows += int(len(chunk))

    if not states:
        raise ValueError("No valid prediction rows found in the provided file.")

    summary_df = _finalize_group_states(
        states,
        threshold=threshold,
        group_cols=resolved_group_cols or [],
        dna_window_counts=dna_window_counts,
        rna_window_counts=rna_window_counts,
    )
    if summary_df is None or summary_df.empty:
        raise ValueError("No summary rows were produced from the provided predictions.")

    if normalized_thresholds:
        threshold_summary_df = _finalize_group_threshold_states(
            states,
            thresholds=normalized_thresholds,
            group_cols=resolved_group_cols or [],
        )

    LOGGER.info(
        "Processed %d prediction rows into %d summary groups.",
        processed_rows,
        len(summary_df),
    )
    final_output_path = output_path or predictions_path
    summary_path = write_pair_summary(
        summary_df,
        output_path=final_output_path,
        add_summary_suffix=output_path is None,
    )
    if threshold_summary_df is not None and not threshold_summary_df.empty:
        threshold_summary_path = write_pair_threshold_summary(
            threshold_summary_df,
            output_path=final_output_path,
            add_summary_suffix=output_path is None,
        )
    return summary_df, summary_path, threshold_summary_df, threshold_summary_path


def summarize_pair_predictions_with_thresholds(
    *,
    predictions_path: str | Path,
    output_path: str | Path | None = None,
    pairs_file: str | Path | None = None,
    dna_seq_file: str | Path | None = None,
    rna_seq_file: str | Path | None = None,
    pair_id_col: str = "pair_id",
    dna_id_col: str = "dna_id",
    rna_id_col: str = "rna_id",
    threshold: float = 0.5,
    ignore_pred_col: bool = False,
    chunksize: int | None = 200_000,
    thresholds: Iterable[float] | None = None,
) -> tuple[pd.DataFrame, Path, pd.DataFrame | None, Path | None]:
    return _summarize_pair_predictions_impl(
        predictions_path=predictions_path,
        output_path=output_path,
        pairs_file=pairs_file,
        dna_seq_file=dna_seq_file,
        rna_seq_file=rna_seq_file,
        pair_id_col=pair_id_col,
        dna_id_col=dna_id_col,
        rna_id_col=rna_id_col,
        threshold=threshold,
        ignore_pred_col=ignore_pred_col,
        chunksize=chunksize,
        thresholds=thresholds,
    )


def summarize_pair_predictions(
    *,
    predictions_path: str | Path,
    output_path: str | Path | None = None,
    pairs_file: str | Path | None = None,
    dna_seq_file: str | Path | None = None,
    rna_seq_file: str | Path | None = None,
    pair_id_col: str = "pair_id",
    dna_id_col: str = "dna_id",
    rna_id_col: str = "rna_id",
    threshold: float = 0.5,
    ignore_pred_col: bool = False,
    chunksize: int | None = 200_000,
    thresholds: Iterable[float] | None = None,
) -> tuple[pd.DataFrame, Path]:
    summary_df, summary_path, _, _ = _summarize_pair_predictions_impl(
        predictions_path=predictions_path,
        output_path=output_path,
        pairs_file=pairs_file,
        dna_seq_file=dna_seq_file,
        rna_seq_file=rna_seq_file,
        pair_id_col=pair_id_col,
        dna_id_col=dna_id_col,
        rna_id_col=rna_id_col,
        threshold=threshold,
        ignore_pred_col=ignore_pred_col,
        chunksize=chunksize,
        thresholds=thresholds,
    )
    return summary_df, summary_path
