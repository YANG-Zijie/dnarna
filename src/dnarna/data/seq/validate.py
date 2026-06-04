"""
# 序列文件验证

本模块用于验证 DNA / RNA 序列文件的格式和内容是否符合最基本的要求，确保后续
处理和分析的正确性。

## 输入数据结构

输入数据为一个 Parquet 文件，至少包含以下列：

- id: 序列的唯一标识符
- seq: 序列字符串。目前默认只支持 ATCG 4 种碱基字符。
- label: 序列的标签，0 表示负样本，1 表示正样本。只能取 0 或 1。
- len: (可选）序列长度，如果存在则必须与 `seq` 的实际长度一致。
- split: (可选）如果该列存在，其值必须为 train/val/test 之一，表示所属数据集划分。

## 用法

本模块提供 `validate_sequence_file`，可用于其他模块进行序列文件的验证。如果验证失
败，会抛出异常并给出错误信息，如果验证通过，则继续后续处理。
"""

__all__ = [
    "LABEL_COLUMN",
    "LEN_COLUMN",
    "SEQ_COLUMN",
    "SEQ_ID_COLUMN",
    "SPLIT_COLUMN",
    "VALID_BASES",
    "VALID_BASES_WITH_N",
    "sanitize_sequence_dataframe",
    "validate_sequence_dataframe",
    "validate_sequence_file",
]

import numbers
from collections.abc import Hashable, Iterable
from pathlib import Path
from typing import Literal

import pandas as pd

VALID_BASES: set[str] = {"A", "T", "C", "G"}
VALID_BASES_WITH_N: set[str] = {"A", "T", "C", "G", "N"}
SEQ_ID_COLUMN = "id"
SEQ_COLUMN = "seq"
LABEL_COLUMN = "label"
LEN_COLUMN = "len"
SPLIT_COLUMN = "split"
REQUIRED_COLUMNS: tuple[str, ...] = (SEQ_ID_COLUMN, SEQ_COLUMN, LABEL_COLUMN)
ALLOWED_SPLITS: set[str] = {"train", "val", "test"}
DedupBy = Literal[None, "id", "seq", "id_seq", "id_or_seq"]


def _ensure_columns(columns: Iterable[str], required: Iterable[str]) -> None:
    missing = [c for c in required if c not in columns]
    if missing:
        raise ValueError(f"File missing required columns: {', '.join(sorted(missing))}")


def _normalize_sequence(seq: str, row_idx: Hashable, *, allow_n: bool = False) -> str:
    normalized = seq.strip().upper()
    if not normalized:
        raise ValueError(f"Row {row_idx}: sequence must not be empty")
    allowed = VALID_BASES_WITH_N if allow_n else VALID_BASES
    invalid = sorted({base for base in normalized if base not in allowed})
    if invalid:
        raise ValueError(
            f"Row {row_idx}: sequence contains unsupported bases: {', '.join(invalid)}"
        )
    return normalized


def _normalize_sequence_or_none(seq: str, *, allow_n: bool = False) -> str | None:
    normalized = str(seq).strip().upper()
    if not normalized:
        return None
    allowed = VALID_BASES_WITH_N if allow_n else VALID_BASES
    invalid = {base for base in normalized if base not in allowed}
    if invalid:
        return None
    return normalized


def _normalize_id_value(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        if float(value).is_integer():
            return str(int(value))
    text = str(value).strip()
    return text or None


def validate_sequence_file(
    path: str | Path, *, require_label: bool = True, allow_n: bool = False
) -> pd.DataFrame:
    """Validate a DNA/RNA Parquet file and return the loaded dataframe on success."""
    parquet_path = Path(path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Sequence file not found: {parquet_path}")
    if not parquet_path.is_file():
        raise ValueError(f"Sequence file path must be a file: {parquet_path}")
    if parquet_path.suffix.lower() != ".parquet":
        raise ValueError(
            f"Sequence file must be a Parquet file with '.parquet' extension: {parquet_path}"
        )

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as exc:  # pragma: no cover - pandas-specific errors
        raise ValueError(f"Failed to read Parquet file: {parquet_path}") from exc

    required = (SEQ_ID_COLUMN, SEQ_COLUMN)
    if require_label:
        required = required + (LABEL_COLUMN,)
    return validate_sequence_dataframe(df, require_label=require_label, allow_n=allow_n)


def validate_sequence_dataframe(
    df: pd.DataFrame, *, require_label: bool = True, allow_n: bool = False
) -> pd.DataFrame:
    """Validate a DNA/RNA dataframe and return a standardized copy on success."""
    required = (SEQ_ID_COLUMN, SEQ_COLUMN)
    if require_label:
        required = required + (LABEL_COLUMN,)
    _ensure_columns(df.columns, required)

    if df.empty:
        raise ValueError("Sequence file is empty")

    validated_sequences: list[str] = []
    validated_ids: list[str] = []
    seen_ids: set[str] = set()
    seen_seqs: dict[str, str] = {}
    labels: list[int] | None = [] if LABEL_COLUMN in df.columns else None
    splits: list[str] | None = [] if SPLIT_COLUMN in df.columns else None
    lengths: list[int] | None = [] if LEN_COLUMN in df.columns else None

    for idx, row in df.iterrows():
        # Validate id
        normalized_id = _normalize_id_value(row.get(SEQ_ID_COLUMN))
        if normalized_id is None:
            raise ValueError(f"Row {idx}: {SEQ_ID_COLUMN} must not be null")
        if normalized_id in seen_ids:
            raise ValueError(f"Row {idx}: duplicate {SEQ_ID_COLUMN} '{normalized_id}'")
        seen_ids.add(normalized_id)
        validated_ids.append(normalized_id)

        # Validate sequence
        seq_value = row[SEQ_COLUMN]
        if pd.isna(seq_value):
            raise ValueError(f"Row {idx}: {SEQ_COLUMN} must not be null")
        normalized_seq = _normalize_sequence(str(seq_value), idx, allow_n=allow_n)
        existing_id = seen_seqs.get(normalized_seq)
        if existing_id is not None and existing_id != normalized_id:
            raise ValueError(
                f"Row {idx}: {SEQ_COLUMN} already assigned to id '{existing_id}'"
            )
        seen_seqs[normalized_seq] = normalized_id
        validated_sequences.append(normalized_seq)
        seq_length = len(normalized_seq)

        if lengths is not None:
            len_value = row[LEN_COLUMN]
            if pd.isna(len_value):
                lengths.append(seq_length)
            else:
                try:
                    reported_len = int(len_value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Row {idx}: {LEN_COLUMN} must be an integer"
                    ) from exc
                if reported_len != seq_length:
                    raise ValueError(
                        f"Row {idx}: {LEN_COLUMN} must equal len({SEQ_COLUMN}); "
                        f"expected {seq_length}, got {reported_len}"
                    )
                lengths.append(reported_len)

        if labels is not None:
            # Validate label if column exists.
            label_value = row[LABEL_COLUMN]
            if pd.isna(label_value):
                raise ValueError(f"Row {idx}: label must not be null")
            try:
                label_int = int(label_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Row {idx}: label must be 0 or 1") from exc
            if label_int not in (0, 1):
                raise ValueError(f"Row {idx}: label must be 0 or 1")
            labels.append(label_int)

        if splits is not None:
            split_value = row[SPLIT_COLUMN]
            if pd.isna(split_value):
                raise ValueError(f"Row {idx}: split must not be null")
            normalized_split = str(split_value).strip().lower()
            if normalized_split not in ALLOWED_SPLITS:
                raise ValueError(
                    f"Row {idx}: split must be one of {sorted(ALLOWED_SPLITS)}, "
                    f"got {split_value!r}"
                )
            splits.append(normalized_split)

    # Mutate dataframe in-place to ensure standardized values.
    df = df.copy()
    df[SEQ_COLUMN] = validated_sequences
    df[SEQ_ID_COLUMN] = validated_ids
    if labels is not None:
        df[LABEL_COLUMN] = labels
    if splits is not None:
        df[SPLIT_COLUMN] = splits
    if lengths is not None:
        df[LEN_COLUMN] = lengths

    return df


def sanitize_sequence_dataframe(
    df: pd.DataFrame,
    *,
    require_label: bool = False,
    allow_n: bool = False,
    require_split: bool = False,
    dedup_by: DedupBy = "id_or_seq",
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Sanitize a dataframe by skipping invalid rows and returning a cleaned copy.
    Duplicates can be skipped via dedup_by (default: "id_or_seq").

    dedup_by modes:
    - None: keep all rows (no de-duplication)
    - "id": skip rows with duplicate id
    - "seq": skip rows with duplicate sequence
    - "id_seq": skip rows with duplicate (id, seq) pairs
    - "id_or_seq": skip rows with duplicate id or duplicate sequence

    Returns:
        (clean_df, skipped_rows) where skipped_rows is a list of dicts with
        row_index/id/reason (and optional details).
    """
    required = (SEQ_ID_COLUMN, SEQ_COLUMN)
    if require_label:
        required = required + (LABEL_COLUMN,)
    if require_split:
        required = required + (SPLIT_COLUMN,)
    _ensure_columns(df.columns, required)

    skipped: list[dict] = []
    cleaned_rows: list[dict] = []

    if dedup_by not in (None, "id", "seq", "id_seq", "id_or_seq"):
        raise ValueError(
            "dedup_by must be one of None, 'id', 'seq', 'id_seq', 'id_or_seq'"
        )
    seen_ids: set[str] = set()
    seen_seqs: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()

    has_len_column = LEN_COLUMN in df.columns
    has_label_column = LABEL_COLUMN in df.columns
    has_split_column = SPLIT_COLUMN in df.columns

    for idx, row in df.iterrows():
        rid = _normalize_id_value(row.get(SEQ_ID_COLUMN))
        if rid is None:
            skipped.append({"row_index": int(idx), "id": None, "reason": "missing_id"})
            continue

        seq_value = row.get(SEQ_COLUMN)
        if pd.isna(seq_value):
            skipped.append(
                {"row_index": int(idx), "id": str(rid), "reason": "missing_seq"}
            )
            continue

        normalized = _normalize_sequence_or_none(seq_value, allow_n=allow_n)
        if normalized is None:
            skipped.append(
                {
                    "row_index": int(idx),
                    "id": str(rid),
                    "reason": "invalid_seq_chars",
                }
            )
            continue

        cleaned = row.to_dict()
        cleaned[SEQ_ID_COLUMN] = rid
        cleaned[SEQ_COLUMN] = normalized

        if dedup_by:
            if dedup_by in ("id", "id_or_seq") and rid in seen_ids:
                skipped.append(
                    {"row_index": int(idx), "id": str(rid), "reason": "duplicate_id"}
                )
                continue
            if dedup_by in ("seq", "id_or_seq") and normalized in seen_seqs:
                skipped.append(
                    {"row_index": int(idx), "id": str(rid), "reason": "duplicate_seq"}
                )
                continue
            if dedup_by == "id_seq":
                key = (rid, normalized)
                if key in seen_pairs:
                    skipped.append(
                        {
                            "row_index": int(idx),
                            "id": str(rid),
                            "reason": "duplicate_id_seq",
                        }
                    )
                    continue
                seen_pairs.add(key)
            if dedup_by in ("id", "id_or_seq"):
                seen_ids.add(rid)
            if dedup_by in ("seq", "id_or_seq"):
                seen_seqs.add(normalized)

        if has_len_column:
            len_value = cleaned.get(LEN_COLUMN)
            seq_len = len(normalized)
            if pd.isna(len_value):
                cleaned[LEN_COLUMN] = seq_len
            else:
                try:
                    reported_len = int(len_value)
                except (TypeError, ValueError):
                    skipped.append(
                        {
                            "row_index": int(idx),
                            "id": str(rid),
                            "reason": "invalid_len",
                        }
                    )
                    continue
                if reported_len != seq_len:
                    skipped.append(
                        {
                            "row_index": int(idx),
                            "id": str(rid),
                            "reason": "len_mismatch",
                            "details": {"expected": seq_len, "reported": reported_len},
                        }
                    )
                    continue

        if has_label_column:
            label_value = cleaned.get(LABEL_COLUMN)
            if pd.isna(label_value):
                skipped.append(
                    {"row_index": int(idx), "id": str(rid), "reason": "missing_label"}
                )
                continue
            try:
                label_int = int(label_value)
            except (TypeError, ValueError):
                skipped.append(
                    {"row_index": int(idx), "id": str(rid), "reason": "invalid_label"}
                )
                continue
            if label_int not in (0, 1):
                skipped.append(
                    {"row_index": int(idx), "id": str(rid), "reason": "invalid_label"}
                )
                continue
            cleaned[LABEL_COLUMN] = label_int

        if has_split_column:
            split_value = cleaned.get(SPLIT_COLUMN)
            if pd.isna(split_value):
                skipped.append(
                    {"row_index": int(idx), "id": str(rid), "reason": "missing_split"}
                )
                continue
            normalized_split = str(split_value).strip().lower()
            if normalized_split not in ALLOWED_SPLITS:
                skipped.append(
                    {"row_index": int(idx), "id": str(rid), "reason": "invalid_split"}
                )
                continue
            cleaned[SPLIT_COLUMN] = normalized_split

        cleaned_rows.append(cleaned)

    cleaned_df = pd.DataFrame.from_records(cleaned_rows, columns=df.columns)
    return cleaned_df, skipped
