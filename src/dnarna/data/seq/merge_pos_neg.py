"""
## 输入数据格式

输入数据为 2 个 CSV 文件，分别包含正样本和负样本。每个文件至少包含以下列：

- `id`: 序列的唯一标识符
- `seq`: 序列字符串。目前只支持 ATCG 4 种碱基字符

若 `id` 列缺失或某些行为空，会在读取阶段自动生成唯一 ID。同一个序列在所有文件中始终共享同一个 ID。正负样本文件中若出现相同序列，会在合并前全部移除。

## 输出数据格式

输出数据为 1 个 Parquet 文件，包含以下列：

- `id`: 序列的唯一标识符
- `seq`: 序列字符串
- `len`: 序列长度
- `label`: 序列的标签，0 表示负样本，1 表示正样本

其中 Positive 样本的 label 为 1，Negative 样本的 label 为 0。

## 使用方法
```bash
python -m dnarna.data.seq.merge_pos_neg \
    --pos_file path/to/positive_samples.csv \
    --neg_file path/to/negative_samples.csv \
    --output_dir path/to/output_dir \
    --output_format parquet
```
"""

import argparse
import logging
from collections.abc import Hashable
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from .validate import LABEL_COLUMN, SEQ_COLUMN, SEQ_ID_COLUMN, VALID_BASES

__all__ = [
    "merge_positive_negative",
    "parse_args",
    "main",
]

POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0
REQUIRED_COLUMNS: tuple[str, ...] = (SEQ_COLUMN,)
OUTPUT_COLUMNS: tuple[str, ...] = (SEQ_ID_COLUMN, SEQ_COLUMN, "len", LABEL_COLUMN)
LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    LOGGER.addHandler(logging.NullHandler())


def _configure_file_logging(log_path: Path) -> None:
    """Attach a file handler to the module logger for persistent logs."""
    for handler in LOGGER.handlers:
        if (
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename) == log_path
        ):
            return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s")
    )
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    LOGGER.addHandler(file_handler)


class UnsupportedBasesError(ValueError):
    """Raised when a sequence contains unsupported bases."""


class SequenceIdAllocator:
    """Ensure a single, unique ID is assigned to every unique sequence."""

    def __init__(self, prefix: str = "seq_") -> None:
        self._seq_to_id: dict[str, str] = {}
        self._id_to_seq: dict[str, str] = {}
        self._counter = 1
        self._prefix = prefix

    def assign(
        self,
        sequence: str,
        candidate_id: str | None,
        row_idx: Hashable,
        source_label: str,
    ) -> str:
        """Return a stable ID for the sequence, generating one if needed."""
        if sequence in self._seq_to_id:
            existing = self._seq_to_id[sequence]
            if candidate_id and candidate_id != existing:
                LOGGER.warning(
                    "%s row %s: sequence already assigned id %r, ignoring conflicting id %r",
                    source_label,
                    row_idx,
                    existing,
                    candidate_id,
                )
            return existing

        seq_id = self._resolve_id(sequence, candidate_id, row_idx, source_label)
        self._seq_to_id[sequence] = seq_id
        self._id_to_seq[seq_id] = sequence
        return seq_id

    def _resolve_id(
        self,
        sequence: str,
        candidate_id: str | None,
        row_idx: Hashable,
        source_label: str,
    ) -> str:
        if candidate_id:
            existing = self._id_to_seq.get(candidate_id)
            if existing and existing != sequence:
                raise ValueError(
                    f"{source_label} row {row_idx}: {SEQ_ID_COLUMN}={candidate_id!r} is already used "
                    f"for a different sequence"
                )
            return candidate_id
        return self._generate_unique_id()

    def _generate_unique_id(self) -> str:
        while True:
            seq_id = f"{self._prefix}{self._counter:06d}"
            self._counter += 1
            if seq_id not in self._id_to_seq:
                return seq_id

    def has_sequence(self, sequence: str) -> bool:
        return sequence in self._seq_to_id


def _format_for_message(value: str, max_length: int = 80) -> str:
    """Return a readable representation that is truncated if necessary."""
    if len(value) <= max_length:
        return value
    head = value[: max_length - 10]
    tail = value[-7:]
    return f"{head}...{tail} (len={len(value)})"


def _load_sample_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not csv_path.is_file():
        raise ValueError(f"CSV path must refer to a file: {csv_path}")
    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv file, got: {csv_path}")
    try:
        return pd.read_csv(csv_path)
    except Exception as exc:  # pragma: no cover - pandas read_csv specific
        raise ValueError(f"Failed to read CSV file: {csv_path}") from exc


def _ensure_required_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"CSV file missing required columns: {', '.join(sorted(missing))}"
        )


def _normalize_sequence(
    value: object,
    row_idx: Hashable,
    seq_id: str | None = None,
    original_seq_id: Any | None = None,
) -> str:
    seq = str(value).strip().upper()
    if not seq:
        raise ValueError(f"Row {row_idx}: {SEQ_COLUMN} must not be empty")
    invalid = sorted({ch for ch in seq if ch not in VALID_BASES})
    if invalid:
        prefix = f"Row {row_idx}"
        if seq_id is not None:
            raw_id_str = str(original_seq_id) if original_seq_id is not None else seq_id
            truncated_id = _format_for_message(raw_id_str)
            prefix += f" ({SEQ_ID_COLUMN}={truncated_id!r})"
        raise UnsupportedBasesError(
            f"{prefix}: {SEQ_COLUMN} contains unsupported bases: {', '.join(invalid)}"
        )
    return seq


def _extract_seq_id(value: Any) -> str | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    seq_id = str(value).strip()
    return seq_id or None


def _prepare_labeled_dataframe(
    df: pd.DataFrame,
    label: int,
    allocator: SequenceIdAllocator,
    source_name: str | None = None,
) -> pd.DataFrame:
    _ensure_required_columns(df)
    source_label = source_name or "input dataframe"
    total_rows = len(df)
    skipped_null = 0
    skipped_invalid = 0
    auto_assigned = 0
    records: list[dict[str, object]] = []
    has_id_column = SEQ_ID_COLUMN in df.columns

    row_iter = df.iterrows()
    if tqdm is not None:
        row_iter = tqdm(
            row_iter,
            total=total_rows,
            desc=f"Processing {source_label}",
            leave=False,
        )

    for idx, row in row_iter:
        raw_seq = row[SEQ_COLUMN]
        if pd.isna(raw_seq):
            skipped_null += 1
            continue
        raw_seq_id = row[SEQ_ID_COLUMN] if has_id_column else None
        candidate_id = _extract_seq_id(raw_seq_id)
        try:
            normalized_seq = _normalize_sequence(
                raw_seq, idx, seq_id=candidate_id, original_seq_id=raw_seq_id
            )
        except UnsupportedBasesError as exc:
            skipped_invalid += 1
            LOGGER.debug("%s", exc)
            continue
        is_known_sequence = allocator.has_sequence(normalized_seq)
        seq_id = allocator.assign(
            normalized_seq,
            candidate_id,
            row_idx=idx,
            source_label=source_label,
        )
        if not is_known_sequence and candidate_id is None:
            auto_assigned += 1
        records.append(
            {
                SEQ_ID_COLUMN: seq_id,
                SEQ_COLUMN: normalized_seq,
                "len": len(normalized_seq),
                LABEL_COLUMN: label,
            }
        )

    valid_rows = len(records)
    message = (
        f"{source_label}: processed {total_rows} rows; retained {valid_rows} valid sequences "
        f"(skipped {skipped_null} null {SEQ_COLUMN} values, {skipped_invalid} sequences with unsupported bases)"
    )
    if auto_assigned:
        message += f"; auto-assigned {auto_assigned} {SEQ_ID_COLUMN} values"
    LOGGER.info(message)
    print(message)

    return pd.DataFrame(records, columns=OUTPUT_COLUMNS)


def _remove_cross_label_duplicates(
    pos_df: pd.DataFrame, neg_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pos_df.empty or neg_df.empty:
        return pos_df, neg_df
    overlapping_sequences = set(pos_df[SEQ_COLUMN]) & set(neg_df[SEQ_COLUMN])
    if not overlapping_sequences:
        return pos_df, neg_df
    pos_mask = ~pos_df[SEQ_COLUMN].isin(overlapping_sequences)
    neg_mask = ~neg_df[SEQ_COLUMN].isin(overlapping_sequences)
    removed_pos = int((~pos_mask).sum())
    removed_neg = int((~neg_mask).sum())
    message = (
        f"Removed {removed_pos} positive and {removed_neg} negative sequences "
        f"that share identical {SEQ_COLUMN} values between files "
        f"({len(overlapping_sequences)} unique overlapping sequences)."
    )
    LOGGER.info(message)
    print(message)
    return (
        pos_df.loc[pos_mask].reset_index(drop=True),
        neg_df.loc[neg_mask].reset_index(drop=True),
    )


def merge_positive_negative(
    pos_file: str | Path,
    neg_file: str | Path,
) -> pd.DataFrame:
    """Load positive/negative CSV files, validate them, and combine into one DataFrame."""
    pos_raw = _load_sample_csv(pos_file)
    neg_raw = _load_sample_csv(neg_file)
    allocator = SequenceIdAllocator()
    pos_source = f"{Path(pos_file)} (positive)"
    neg_source = f"{Path(neg_file)} (negative)"
    pos_df = _prepare_labeled_dataframe(
        pos_raw, POSITIVE_LABEL, allocator, source_name=pos_source
    )
    neg_df = _prepare_labeled_dataframe(
        neg_raw, NEGATIVE_LABEL, allocator, source_name=neg_source
    )
    pos_df, neg_df = _remove_cross_label_duplicates(pos_df, neg_df)
    if pos_df.empty and neg_df.empty:
        raise ValueError("Both positive and negative CSV files are empty.")
    merged = pd.concat([pos_df, neg_df], ignore_index=True)
    total_input_rows = len(pos_raw) + len(neg_raw)
    total_valid_rows = len(merged)
    summary_message = f"Combined dataset: processed {total_input_rows} rows; retained {total_valid_rows} valid sequences"
    LOGGER.info(summary_message)
    print(summary_message)
    return merged.loc[:, list(OUTPUT_COLUMNS)]


def _log_and_print(message: str) -> None:
    LOGGER.info(message)
    print(message)


def _resolve_output_path(*, output_dir: str, output_format: str) -> Path:
    output_dir_path = Path(output_dir)
    fmt = output_format.strip().lower()
    if fmt != "parquet":
        raise ValueError(
            f"Unsupported --output_format: {output_format!r} (only 'parquet' is supported)"
        )
    return output_dir_path / "merged.parquet"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge positive/negative sequence CSVs into a labeled Parquet dataset."
    )
    parser.add_argument(
        "--pos_file",
        required=True,
        help="Path to the positive sample CSV file.",
    )
    parser.add_argument(
        "--neg_file",
        required=True,
        help="Path to the negative sample CSV file.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write outputs (parquet + log).",
    )
    parser.add_argument(
        "--output_format",
        required=True,
        choices=["parquet"],
        help="Output file format (required). Currently only supports: parquet.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_path = _resolve_output_path(
        output_dir=args.output_dir,
        output_format=args.output_format,
    )
    log_path = output_path.with_suffix(output_path.suffix + ".log")
    _configure_file_logging(log_path)
    merged = merge_positive_negative(args.pos_file, args.neg_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    _log_and_print(f"Wrote merged dataset to {output_path}")
    _log_and_print(f"Detailed log saved to {log_path}")


if __name__ == "__main__":
    main()
