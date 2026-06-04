from __future__ import annotations

import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("pair_predict_utils")

PARQUET_EXTENSIONS = {".parquet", ".parq", ".pq"}
CSV_EXTENSIONS = {".csv", ".tsv", ".csv.gz", ".tsv.gz"}


def infer_extension(path: Path) -> str:
    lower = path.name.lower()
    for ext in PARQUET_EXTENSIONS | CSV_EXTENSIONS:
        if lower.endswith(ext):
            return ext
    return path.suffix.lower()


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    ext = infer_extension(path)
    if ext in PARQUET_EXTENSIONS:
        return pd.read_parquet(path)
    if ext in CSV_EXTENSIONS or not ext:
        sep = "\t" if ext.startswith(".tsv") else ","
        return pd.read_csv(path, sep=sep)
    raise ValueError(
        f"Unsupported input extension '{ext}'. "
        f"Supported: {sorted(PARQUET_EXTENSIONS | CSV_EXTENSIONS)}"
    )


def _normalize_id(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def dedupe_ids(ids: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    deduped: list[str] = []
    for raw in ids:
        key = str(raw)
        n = counts.get(key, 0)
        if n:
            deduped.append(f"{key}__dup{n}")
        else:
            deduped.append(key)
        counts[key] = n + 1
    return deduped


def normalize_pair_dataframe(
    df: pd.DataFrame,
    *,
    pair_id_col: str,
    dna_id_col: str,
    rna_id_col: str,
    label_col: str | None,
) -> tuple[pd.DataFrame, dict]:
    missing_cols = [c for c in (dna_id_col, rna_id_col) if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Pair file missing required columns: {', '.join(sorted(missing_cols))}"
        )

    cleaned = df.copy()
    if pair_id_col not in cleaned.columns:
        cleaned[pair_id_col] = [f"pair_{i+1}" for i in range(len(cleaned))]

    cleaned[pair_id_col] = cleaned[pair_id_col].map(_normalize_id)
    cleaned[dna_id_col] = cleaned[dna_id_col].map(_normalize_id)
    cleaned[rna_id_col] = cleaned[rna_id_col].map(_normalize_id)

    missing_pair = cleaned[pair_id_col].isna()
    if missing_pair.any():
        cleaned.loc[missing_pair, pair_id_col] = [
            f"pair_{i+1}"
            for i in range(int(missing_pair.sum()))
        ]

    missing_required = cleaned[dna_id_col].isna() | cleaned[rna_id_col].isna()
    dropped_missing = int(missing_required.sum())
    cleaned = cleaned.loc[~missing_required].copy()

    cleaned[pair_id_col] = dedupe_ids(cleaned[pair_id_col].astype(str).tolist())

    dropped_invalid_label = 0
    if label_col and label_col in cleaned.columns:
        labels = cleaned[label_col]
        labels = pd.to_numeric(labels, errors="coerce")
        valid_mask = labels.isin([0, 1])
        dropped_invalid_label = int((~valid_mask).sum())
        cleaned = cleaned.loc[valid_mask].copy()
        cleaned[label_col] = labels.loc[valid_mask].astype(int).values

    report = {
        "input_rows": int(len(df)),
        "kept_rows": int(len(cleaned)),
        "dropped_missing_ids": dropped_missing,
        "dropped_invalid_label": dropped_invalid_label,
    }
    return cleaned, report


def build_pair_features(
    dna_embeddings: np.ndarray,
    rna_embeddings: np.ndarray,
    dna_indices: np.ndarray,
    rna_indices: np.ndarray,
    *,
    mode: str,
    chunk_size: int = 4096,
    show_progress: bool = False,
    num_workers: int = 1,
) -> np.ndarray:
    dna_dim = int(dna_embeddings.shape[1])
    rna_dim = int(rna_embeddings.shape[1])
    mode_key = str(mode).lower().strip()

    if mode_key in {"concat", "cat"}:
        feature_dim = dna_dim + rna_dim
        combine = "concat"
    elif mode_key in {"absdiff", "abs_diff", "diff"}:
        if dna_dim != rna_dim:
            raise ValueError(
                f"Feature mode '{mode}' requires dna/rna dims to match "
                f"(dna={dna_dim}, rna={rna_dim})."
            )
        feature_dim = dna_dim
        combine = "absdiff"
    elif mode_key in {"mul", "product"}:
        if dna_dim != rna_dim:
            raise ValueError(
                f"Feature mode '{mode}' requires dna/rna dims to match "
                f"(dna={dna_dim}, rna={rna_dim})."
            )
        feature_dim = dna_dim
        combine = "mul"
    elif mode_key in {"all", "concat_absdiff_mul"}:
        if dna_dim != rna_dim:
            raise ValueError(
                f"Feature mode '{mode}' requires dna/rna dims to match "
                f"(dna={dna_dim}, rna={rna_dim})."
            )
        feature_dim = dna_dim + rna_dim + 2 * dna_dim
        combine = "all"
    else:
        raise ValueError(
            "feature mode must be one of: concat, absdiff, mul, all"
        )

    n_pairs = int(len(dna_indices))
    features = np.empty((n_pairs, feature_dim), dtype=np.float32)
    if n_pairs == 0:
        return features

    if chunk_size <= 0:
        chunk_size = n_pairs

    if num_workers < 1:
        num_workers = 1

    progress = None
    if show_progress:
        from tqdm.auto import tqdm

        progress = tqdm(total=n_pairs, desc="pair features", unit="pair")

    def _compute_slice(start: int, end: int) -> int:
        dna_batch = dna_embeddings[dna_indices[start:end]]
        rna_batch = rna_embeddings[rna_indices[start:end]]

        if combine == "concat":
            features[start:end] = np.concatenate([dna_batch, rna_batch], axis=1)
        elif combine == "absdiff":
            features[start:end] = np.abs(dna_batch - rna_batch)
        elif combine == "mul":
            features[start:end] = dna_batch * rna_batch
        else:
            diff = np.abs(dna_batch - rna_batch)
            prod = dna_batch * rna_batch
            features[start:end] = np.concatenate(
                [dna_batch, rna_batch, diff, prod], axis=1
            )
        return end - start

    if num_workers > 1 and n_pairs > chunk_size:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(_compute_slice, start, min(start + chunk_size, n_pairs))
                for start in range(0, n_pairs, chunk_size)
            ]
            for future in as_completed(futures):
                done = future.result()
                if progress is not None:
                    progress.update(done)
    else:
        for start in range(0, n_pairs, chunk_size):
            end = min(start + chunk_size, n_pairs)
            done = _compute_slice(start, end)
            if progress is not None:
                progress.update(done)

    if progress is not None:
        progress.close()

    return features


def sample_negative_pairs(
    *,
    pos_pairs: set[tuple[str, str]],
    dna_ids: list[str],
    rna_ids: list[str],
    n_samples: int,
    seed: int,
) -> list[tuple[str, str]]:
    if n_samples <= 0:
        return []

    rng = random.Random(seed)
    negatives: set[tuple[str, str]] = set()
    max_attempts = max(1000, n_samples * 20)
    attempts = 0

    while len(negatives) < n_samples and attempts < max_attempts:
        dna_id = rng.choice(dna_ids)
        rna_id = rng.choice(rna_ids)
        pair = (dna_id, rna_id)
        if pair in pos_pairs or pair in negatives:
            attempts += 1
            continue
        negatives.add(pair)
        attempts += 1

    if len(negatives) < n_samples:
        LOGGER.warning(
            "Only sampled %d/%d negatives after %d attempts; "
            "dataset might be close to saturated.",
            len(negatives),
            n_samples,
            attempts,
        )
    return list(negatives)
