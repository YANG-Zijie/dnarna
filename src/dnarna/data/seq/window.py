"""
# Sliding-window split for long sequences

Split sequences longer than window_size into overlapping windows.

## Usage

Input format: CSV (.csv) or Parquet (.parquet)
Default output file name: {input_stem}.windowed.parquet (parquet) /
{input_stem}.windowed.csv (csv)
Output format is controlled by --output_format.

Input requirements (CSV/Parquet):
- Required columns: id, seq
- seq allows only A/T/C/G (rows with N/other chars are skipped)
- Optional: len (validated if present; mismatches are skipped)
- Optional: label (validated if present; invalid rows are skipped)
- Optional: split (validated if present; invalid rows are skipped)

Output structure:
- All input columns are preserved
- Added columns: parent_id, window_index, window_start, window_end
- For split rows, id becomes {id}_win_{idx} and seq becomes the window segment
- If len exists, it is updated to the window length
Additional outputs:
- {output_name}.log contains summary counts (e.g. input.windowed.csv.log)
- {output_name}.report.json records parameters, skipped ids/reasons, and windowed ids

```bash
python -m dnarna.data.seq.window \
    --input_file path/to/input.parquet \
    --output_dir path/to/output_dir \
    --output_format parquet \
    --window_size 1024 \
    --stride 512
```
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from .validate import (
    LEN_COLUMN,
    SEQ_COLUMN,
    SEQ_ID_COLUMN,
    sanitize_sequence_dataframe,
)

__all__ = [
    "DEFAULT_STRIDE",
    "DEFAULT_WINDOW_SIZE",
    "MAX_WINDOW_SIZE",
    "split_long_sequences_dataframe",
    "window_sequence_dict",
    "parse_args",
    "main",
]

MAX_WINDOW_SIZE = 1000
DEFAULT_WINDOW_SIZE = MAX_WINDOW_SIZE
DEFAULT_STRIDE = 500
PARENT_ID_COLUMN = "parent_id"
WINDOW_INDEX_COLUMN = "window_index"
WINDOW_START_COLUMN = "window_start"
WINDOW_END_COLUMN = "window_end"

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    LOGGER.addHandler(logging.NullHandler())


def _configure_file_logging(log_path: Path) -> None:
    """Ensure logs are written to a file alongside the output data."""
    for handler in LOGGER.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_path:
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


def _log_and_print(message: str) -> None:
    LOGGER.info(message)
    print(message)


def _load_input(path: str | Path) -> pd.DataFrame:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"Input path must be a file: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(input_path)
    if suffix == ".csv":
        return pd.read_csv(input_path)
    raise ValueError(
        f"Unsupported input extension '{suffix}' for path '{input_path}'. "
        "Use CSV (.csv) or Parquet (.parquet)."
    )


def _default_output_name(input_path: Path, output_format: str) -> str:
    suffix = ".csv" if output_format == "csv" else ".parquet"
    return f"{input_path.stem}.windowed{suffix}"


def _resolve_output_path(
    output_dir: Path, output_name: str | None, output_format: str, default_name: str
) -> Path:
    name = output_name or default_name
    output_path = output_dir / name
    expected_suffix = ".csv" if output_format == "csv" else ".parquet"
    suffix = output_path.suffix.lower()
    if suffix != expected_suffix:
        if suffix:
            output_path = output_path.with_suffix(expected_suffix)
        else:
            output_path = output_path.with_name(output_path.name + expected_suffix)
    return output_path


def _write_output(df: pd.DataFrame, output_path: Path, output_format: str) -> None:
    if output_format == "csv":
        df.to_csv(output_path, index=False)
    else:
        df.to_parquet(output_path, index=False)



def _validate_window_params(window_size: int, stride: int) -> None:
    if window_size <= 0:
        raise ValueError(f"window_size must be > 0, got {window_size}")
    if stride <= 0:
        raise ValueError(f"stride must be > 0, got {stride}")


def _iter_windows(seq: str, window_size: int, stride: int) -> list[tuple[int, int, str]]:
    windows: list[tuple[int, int, str]] = []
    start = 0
    seq_len = len(seq)
    while start < seq_len:
        end = min(start + window_size, seq_len)
        windows.append((start, end, seq[start:end]))
        if end == seq_len:
            break
        start += stride
    return windows


def split_long_sequences_dataframe(
    df: pd.DataFrame,
    *,
    window_size: int,
    stride: int | None = None,
) -> pd.DataFrame:
    """
    Split sequences longer than window_size using sliding windows and append window metadata.

    - window_size: length of each window
    - stride: step size between windows (default: window_size // 2, at least 1)
    """
    if df.empty:
        raise ValueError("Input dataframe is empty; nothing to split.")

    resolved_window = int(window_size)
    resolved_stride = int(stride or max(1, resolved_window // 2))
    _validate_window_params(resolved_window, resolved_stride)

    has_len_column = LEN_COLUMN in df.columns
    records: list[dict] = []

    for _, row in df.iterrows():
        seq = str(row[SEQ_COLUMN])
        seq_len = len(seq)
        base = row.to_dict()
        base[PARENT_ID_COLUMN] = base.get(SEQ_ID_COLUMN)

        if seq_len <= resolved_window:
            base[WINDOW_INDEX_COLUMN] = 0
            base[WINDOW_START_COLUMN] = 0
            base[WINDOW_END_COLUMN] = seq_len
            if has_len_column:
                base[LEN_COLUMN] = seq_len
            records.append(base)
            continue

        for idx, (start, end, chunk) in enumerate(
            _iter_windows(seq, resolved_window, resolved_stride), start=1
        ):
            window_row = dict(base)
            window_row[SEQ_ID_COLUMN] = f"{base[SEQ_ID_COLUMN]}_win_{idx}"
            window_row[SEQ_COLUMN] = chunk
            window_row[WINDOW_INDEX_COLUMN] = idx
            window_row[WINDOW_START_COLUMN] = start
            window_row[WINDOW_END_COLUMN] = end
            if has_len_column:
                window_row[LEN_COLUMN] = len(chunk)
            records.append(window_row)

    return pd.DataFrame.from_records(records)


def window_sequence_dict(
    seqs: dict[str, str],
    *,
    window_size: int,
    stride: int | None = None,
    allow_n: bool = False,
) -> tuple[dict[str, str], dict]:
    """Window an in-memory id->seq mapping, returning windowed seqs and a report."""
    if not seqs:
        return {}, {
            "total_input_rows": 0,
            "total_valid_sequences": 0,
            "kept_as_is": 0,
            "split_sequences": 0,
            "total_output_rows": 0,
            "window_rows_from_long": 0,
            "skipped_count": 0,
            "skipped": [],
            "split_window_count": 0,
            "split_windows": [],
        }

    raw_df = pd.DataFrame(
        {SEQ_ID_COLUMN: list(seqs.keys()), SEQ_COLUMN: list(seqs.values())}
    )
    total_input = len(raw_df)
    clean_df, skipped = sanitize_sequence_dataframe(raw_df, allow_n=allow_n)
    total_valid = len(clean_df)

    lengths = clean_df[SEQ_COLUMN].map(lambda x: len(str(x)))
    over_mask = lengths > int(window_size)
    count_over = int(over_mask.sum())
    count_keep = total_valid - count_over

    split_df = split_long_sequences_dataframe(
        clean_df,
        window_size=window_size,
        stride=stride,
    )

    total_out = len(split_df)
    windows_from_long = total_out - count_keep

    split_windows: list[dict] = []
    if count_over:
        for parent_id, group in split_df[split_df[WINDOW_INDEX_COLUMN] > 0].groupby(
            PARENT_ID_COLUMN
        ):
            window_ids = group[SEQ_ID_COLUMN].tolist()
            split_windows.append({"parent_id": parent_id, "window_ids": window_ids})

    windowed = dict(zip(split_df[SEQ_ID_COLUMN], split_df[SEQ_COLUMN]))
    report = {
        "total_input_rows": total_input,
        "total_valid_sequences": total_valid,
        "kept_as_is": count_keep,
        "split_sequences": count_over,
        "total_output_rows": total_out,
        "window_rows_from_long": windows_from_long,
        "skipped_count": len(skipped),
        "skipped": skipped,
        "split_window_count": len(split_windows),
        "split_windows": split_windows,
    }
    return windowed, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split sequences longer than window_size using sliding windows."
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="Input CSV or Parquet file (must include id and seq columns).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory; data/logs are written inside.",
    )
    parser.add_argument(
        "--output_name",
        default=None,
        help=(
            "Output filename (default: {input_stem}.windowed.parquet or "
            "{input_stem}.windowed.csv)."
        ),
    )
    parser.add_argument(
        "--output_format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Output format (parquet or csv).",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        required=True,
        help="Window size; sequences longer than this will be split.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride between windows (default: window_size//2).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file)
    raw_df = _load_input(input_path)
    total_input = len(raw_df)
    df, skipped = sanitize_sequence_dataframe(raw_df, allow_n=False)
    split_windows: list[dict] = []

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(df)
    lengths = df[SEQ_COLUMN].map(lambda x: len(str(x)))
    over_mask = lengths > int(args.window_size)
    count_over = int(over_mask.sum())
    count_keep = total - count_over

    split_df = split_long_sequences_dataframe(
        df,
        window_size=args.window_size,
        stride=args.stride,
    )

    total_out = len(split_df)
    windows_from_long = total_out - count_keep

    if count_over:
        for parent_id, group in split_df[split_df[WINDOW_INDEX_COLUMN] > 0].groupby(
            PARENT_ID_COLUMN
        ):
            window_ids = group[SEQ_ID_COLUMN].tolist()
            split_windows.append({"parent_id": parent_id, "window_ids": window_ids})

    _log_and_print(f"Total input rows: {total_input}")
    _log_and_print(f"Total valid sequences: {total}")
    _log_and_print(f"Sequences kept as-is (<= window_size): {count_keep}")
    _log_and_print(f"Sequences split (> window_size): {count_over}")
    _log_and_print(f"Total output rows: {total_out}")
    _log_and_print(f"Window rows from long sequences: {windows_from_long}")
    _log_and_print(f"Skipped rows: {len(skipped)}")

    output_format = str(args.output_format).lower()
    default_name = _default_output_name(input_path, output_format)
    output_path = _resolve_output_path(
        output_dir, args.output_name, output_format, default_name
    )
    log_path = output_path.with_name(f"{output_path.name}.log")
    _configure_file_logging(log_path)
    _write_output(split_df, output_path, output_format)
    _log_and_print(f"Wrote windowed dataset to {output_path}")

    report_path = output_path.with_name(f"{output_path.name}.report.json")
    report_payload = {
        "input_file": str(args.input_file),
        "output_dir": str(output_dir),
        "output_name": output_path.name,
        "output_format": output_format,
        "window_size": int(args.window_size),
        "stride": int(args.stride or max(1, int(args.window_size) // 2)),
        "total_input_rows": total_input,
        "total_valid_sequences": total,
        "kept_as_is": count_keep,
        "split_sequences": count_over,
        "total_output_rows": total_out,
        "window_rows_from_long": windows_from_long,
        "skipped_count": len(skipped),
        "skipped": skipped,
        "split_window_count": len(split_windows),
        "split_windows": split_windows,
    }
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report_payload, fh, ensure_ascii=True, indent=2)
    _log_and_print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
