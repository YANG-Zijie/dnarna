"""
Window DNA/RNA sequences and expand pair datasets.
"""

from __future__ import annotations

import argparse
import json
import logging
import numbers
from pathlib import Path

import pandas as pd

from dnarna.data.seq.validate import (
    SEQ_COLUMN,
    SEQ_ID_COLUMN,
    sanitize_sequence_dataframe,
)
from dnarna.data.seq.window import DEFAULT_WINDOW_SIZE, split_long_sequences_dataframe

from .clean import (
    DNA_ID_COLUMN,
    DNA_SEQ_COLUMN,
    PAIR_ID_COLUMN,
    RNA_ID_COLUMN,
    RNA_SEQ_COLUMN,
)

__all__ = ["window_pair_dataset", "parse_args", "main"]

PAIR_PARENT_ID_COLUMN = "pair_parent_id"
DNA_PARENT_ID_COLUMN = "dna_parent_id"
RNA_PARENT_ID_COLUMN = "rna_parent_id"
DNA_WINDOW_INDEX_COLUMN = "dna_window_index"
DNA_WINDOW_START_COLUMN = "dna_window_start"
DNA_WINDOW_END_COLUMN = "dna_window_end"
RNA_WINDOW_INDEX_COLUMN = "rna_window_index"
RNA_WINDOW_START_COLUMN = "rna_window_start"
RNA_WINDOW_END_COLUMN = "rna_window_end"
PARENT_ID_COLUMN = "parent_id"
WINDOW_INDEX_COLUMN = "window_index"
WINDOW_START_COLUMN = "window_start"
WINDOW_END_COLUMN = "window_end"

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    LOGGER.addHandler(logging.NullHandler())


def _configure_file_logging(log_path: Path) -> None:
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
    if suffix == ".csv":
        return pd.read_csv(input_path)
    if suffix == ".parquet":
        return pd.read_parquet(input_path)
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


def _normalize_rna_seq(value: object) -> object:
    if value is None or pd.isna(value):
        return value
    return str(value).strip().upper().replace("U", "T")


def _ensure_columns(df: pd.DataFrame, required: list[str], *, source: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"{source} missing required columns: {', '.join(sorted(missing))}"
        )


def _validate_window_params(window_size: int, stride: int) -> None:
    if window_size <= 0:
        raise ValueError(f"window_size must be > 0, got {window_size}")
    if stride <= 0:
        raise ValueError(f"stride must be > 0, got {stride}")


def _normalize_pair_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    _ensure_columns(
        df, [PAIR_ID_COLUMN, DNA_ID_COLUMN, RNA_ID_COLUMN], source="pair file"
    )
    cleaned_rows: list[dict] = []
    skipped: list[dict] = []

    for idx, row in df.iterrows():
        pair_id = _normalize_id_value(row.get(PAIR_ID_COLUMN))
        dna_id = _normalize_id_value(row.get(DNA_ID_COLUMN))
        rna_id = _normalize_id_value(row.get(RNA_ID_COLUMN))
        if pair_id is None:
            skipped.append(
                {"row_index": int(idx), "pair_id": None, "reason": "missing_pair_id"}
            )
            continue
        if dna_id is None:
            skipped.append(
                {
                    "row_index": int(idx),
                    "pair_id": pair_id,
                    "reason": "missing_dna_id",
                }
            )
            continue
        if rna_id is None:
            skipped.append(
                {
                    "row_index": int(idx),
                    "pair_id": pair_id,
                    "reason": "missing_rna_id",
                }
            )
            continue
        cleaned = row.to_dict()
        cleaned[PAIR_ID_COLUMN] = pair_id
        cleaned[DNA_ID_COLUMN] = dna_id
        cleaned[RNA_ID_COLUMN] = rna_id
        cleaned_rows.append(cleaned)

    cleaned_df = pd.DataFrame.from_records(cleaned_rows, columns=df.columns)
    return cleaned_df, skipped


def _window_sequence_dataframe(
    df: pd.DataFrame,
    *,
    id_col: str,
    seq_col: str,
    prefix: str,
    window_size: int,
    stride: int,
    allow_u: bool,
) -> tuple[pd.DataFrame, dict]:
    _ensure_columns(df, [id_col, seq_col], source=f"{prefix} file")

    work = df.rename(columns={id_col: SEQ_ID_COLUMN, seq_col: SEQ_COLUMN}).copy()
    if allow_u:
        work[SEQ_COLUMN] = work[SEQ_COLUMN].map(_normalize_rna_seq)

    total_input = len(work)
    clean_df, skipped = sanitize_sequence_dataframe(work, allow_n=False)
    total_valid = len(clean_df)

    if clean_df.empty:
        empty = clean_df.copy()
        empty[PARENT_ID_COLUMN] = []
        empty[WINDOW_INDEX_COLUMN] = []
        empty[WINDOW_START_COLUMN] = []
        empty[WINDOW_END_COLUMN] = []
        split_df = empty
        count_keep = 0
        count_over = 0
        total_out = 0
        windows_from_long = 0
    else:
        lengths = clean_df[SEQ_COLUMN].map(lambda x: len(str(x)))
        count_over = int((lengths > window_size).sum())
        count_keep = total_valid - count_over
        split_df = split_long_sequences_dataframe(
            clean_df,
            window_size=window_size,
            stride=stride,
        )
        total_out = len(split_df)
        windows_from_long = total_out - count_keep

    renamed = split_df.rename(
        columns={
            SEQ_ID_COLUMN: id_col,
            SEQ_COLUMN: seq_col,
            PARENT_ID_COLUMN: f"{prefix}_parent_id",
            WINDOW_INDEX_COLUMN: f"{prefix}_window_index",
            WINDOW_START_COLUMN: f"{prefix}_window_start",
            WINDOW_END_COLUMN: f"{prefix}_window_end",
        }
    )

    report = {
        "input_rows": total_input,
        "valid_rows": total_valid,
        "skipped_count": len(skipped),
        "skipped": skipped,
        "kept_as_is": count_keep,
        "split_sequences": count_over,
        "total_output_rows": total_out,
        "window_rows_from_long": windows_from_long,
    }
    return renamed, report


def _build_pair_ids(
    df: pd.DataFrame,
    *,
    parent_col: str,
    dna_index_col: str,
    rna_index_col: str,
) -> tuple[pd.Series, int]:
    base = (
        df[parent_col].astype(str)
        + "__d"
        + df[dna_index_col].astype(str)
        + "__r"
        + df[rna_index_col].astype(str)
    )
    dup = base.groupby(base).cumcount()
    collision_count = int((dup > 0).sum())
    pair_ids = base + dup.map(lambda n: f"__dup{n}" if n else "")
    return pair_ids, collision_count


def window_pair_dataset(
    pair_df: pd.DataFrame,
    dna_df: pd.DataFrame,
    rna_df: pd.DataFrame,
    *,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int | None = None,
    dna_window_size: int | None = None,
    dna_stride: int | None = None,
    rna_window_size: int | None = None,
    rna_stride: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    resolved_window = int(window_size)
    resolved_stride = (
        int(stride) if stride is not None else max(1, resolved_window // 2)
    )
    _validate_window_params(resolved_window, resolved_stride)

    resolved_dna_window = (
        int(dna_window_size) if dna_window_size is not None else resolved_window
    )
    if dna_stride is not None:
        resolved_dna_stride = int(dna_stride)
    elif dna_window_size is not None and stride is None:
        resolved_dna_stride = max(1, resolved_dna_window // 2)
    else:
        resolved_dna_stride = resolved_stride
    _validate_window_params(resolved_dna_window, resolved_dna_stride)

    resolved_rna_window = (
        int(rna_window_size) if rna_window_size is not None else resolved_window
    )
    if rna_stride is not None:
        resolved_rna_stride = int(rna_stride)
    elif rna_window_size is not None and stride is None:
        resolved_rna_stride = max(1, resolved_rna_window // 2)
    else:
        resolved_rna_stride = resolved_stride
    _validate_window_params(resolved_rna_window, resolved_rna_stride)

    normalized_pairs, skipped_pairs = _normalize_pair_dataframe(pair_df)

    dna_windowed, dna_report = _window_sequence_dataframe(
        dna_df,
        id_col=DNA_ID_COLUMN,
        seq_col=DNA_SEQ_COLUMN,
        prefix="dna",
        window_size=resolved_dna_window,
        stride=resolved_dna_stride,
        allow_u=False,
    )
    rna_windowed, rna_report = _window_sequence_dataframe(
        rna_df,
        id_col=RNA_ID_COLUMN,
        seq_col=RNA_SEQ_COLUMN,
        prefix="rna",
        window_size=resolved_rna_window,
        stride=resolved_rna_stride,
        allow_u=True,
    )

    pairs = normalized_pairs.rename(
        columns={
            PAIR_ID_COLUMN: PAIR_PARENT_ID_COLUMN,
            DNA_ID_COLUMN: DNA_PARENT_ID_COLUMN,
            RNA_ID_COLUMN: RNA_PARENT_ID_COLUMN,
        }
    )

    dna_parent_ids = set(dna_windowed.get(DNA_PARENT_ID_COLUMN, pd.Series()).unique())
    rna_parent_ids = set(rna_windowed.get(RNA_PARENT_ID_COLUMN, pd.Series()).unique())
    missing_dna = pairs[~pairs[DNA_PARENT_ID_COLUMN].isin(dna_parent_ids)]
    missing_rna = pairs[~pairs[RNA_PARENT_ID_COLUMN].isin(rna_parent_ids)]

    dna_cols = [
        DNA_ID_COLUMN,
        DNA_PARENT_ID_COLUMN,
        DNA_WINDOW_INDEX_COLUMN,
        DNA_WINDOW_START_COLUMN,
        DNA_WINDOW_END_COLUMN,
    ]
    rna_cols = [
        RNA_ID_COLUMN,
        RNA_PARENT_ID_COLUMN,
        RNA_WINDOW_INDEX_COLUMN,
        RNA_WINDOW_START_COLUMN,
        RNA_WINDOW_END_COLUMN,
    ]

    pairs_dna = pairs.merge(
        dna_windowed[dna_cols] if dna_cols[0] in dna_windowed.columns else dna_windowed,
        on=DNA_PARENT_ID_COLUMN,
        how="inner",
    )
    pairs_full = pairs_dna.merge(
        rna_windowed[rna_cols] if rna_cols[0] in rna_windowed.columns else rna_windowed,
        on=RNA_PARENT_ID_COLUMN,
        how="inner",
    )

    if pairs_full.empty:
        pair_ids = pd.Series([], dtype=str)
        collision_count = 0
    else:
        pair_ids, collision_count = _build_pair_ids(
            pairs_full,
            parent_col=PAIR_PARENT_ID_COLUMN,
            dna_index_col=DNA_WINDOW_INDEX_COLUMN,
            rna_index_col=RNA_WINDOW_INDEX_COLUMN,
        )
    pairs_full[PAIR_ID_COLUMN] = pair_ids

    ordered = [
        PAIR_ID_COLUMN,
        PAIR_PARENT_ID_COLUMN,
        DNA_ID_COLUMN,
        RNA_ID_COLUMN,
        DNA_PARENT_ID_COLUMN,
        RNA_PARENT_ID_COLUMN,
        DNA_WINDOW_INDEX_COLUMN,
        RNA_WINDOW_INDEX_COLUMN,
        DNA_WINDOW_START_COLUMN,
        DNA_WINDOW_END_COLUMN,
        RNA_WINDOW_START_COLUMN,
        RNA_WINDOW_END_COLUMN,
    ]
    ordered = [col for col in ordered if col in pairs_full.columns]
    remaining = [col for col in pairs_full.columns if col not in ordered]
    pairs_full = pairs_full[ordered + remaining]

    report = {
        "parameters": {
            "window_size": resolved_window,
            "stride": resolved_stride,
            "dna_window_size": resolved_dna_window,
            "dna_stride": resolved_dna_stride,
            "rna_window_size": resolved_rna_window,
            "rna_stride": resolved_rna_stride,
        },
        "pairs": {
            "input_rows": len(pair_df),
            "valid_rows": len(normalized_pairs),
            "skipped_count": len(skipped_pairs),
            "skipped": skipped_pairs,
            "missing_dna_refs": int(len(missing_dna)),
            "missing_rna_refs": int(len(missing_rna)),
            "output_rows": len(pairs_full),
            "pair_id_collision_count": collision_count,
        },
        "dna": dna_report,
        "rna": rna_report,
    }
    return pairs_full, dna_windowed, rna_windowed, report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Window DNA/RNA sequences and expand DNA-RNA pairs."
    )
    parser.add_argument(
        "--pair_file",
        required=True,
        help="Path to the pair CSV/Parquet file (pair_id, dna_id, rna_id).",
    )
    parser.add_argument(
        "--dna_file",
        required=True,
        help="Path to the DNA CSV/Parquet file (dna_id, dna_seq).",
    )
    parser.add_argument(
        "--rna_file",
        required=True,
        help="Path to the RNA CSV/Parquet file (rna_id, rna_seq).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write outputs (data + log + report).",
    )
    parser.add_argument(
        "--output_format",
        choices=["csv", "parquet"],
        default="csv",
        help="Output format for windowed data files.",
    )
    parser.add_argument(
        "--pair_output",
        default=None,
        help="Optional output filename for the windowed pair file.",
    )
    parser.add_argument(
        "--dna_output",
        default=None,
        help="Optional output filename for the windowed DNA file.",
    )
    parser.add_argument(
        "--rna_output",
        default=None,
        help="Optional output filename for the windowed RNA file.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Window size; sequences longer than this will be split.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride between windows (default: window_size//2).",
    )
    parser.add_argument(
        "--dna_window_size",
        type=int,
        default=None,
        help="Override window size for DNA (defaults to window_size).",
    )
    parser.add_argument(
        "--dna_stride",
        type=int,
        default=None,
        help="Override stride for DNA (defaults to stride or dna_window_size//2).",
    )
    parser.add_argument(
        "--rna_window_size",
        type=int,
        default=None,
        help="Override window size for RNA (defaults to window_size).",
    )
    parser.add_argument(
        "--rna_stride",
        type=int,
        default=None,
        help="Override stride for RNA (defaults to stride or rna_window_size//2).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_format = str(args.output_format).lower()

    pair_path = Path(args.pair_file)
    dna_path = Path(args.dna_file)
    rna_path = Path(args.rna_file)

    pair_output_path = _resolve_output_path(
        output_dir,
        args.pair_output,
        output_format,
        _default_output_name(pair_path, output_format),
    )
    dna_output_path = _resolve_output_path(
        output_dir,
        args.dna_output,
        output_format,
        _default_output_name(dna_path, output_format),
    )
    rna_output_path = _resolve_output_path(
        output_dir,
        args.rna_output,
        output_format,
        _default_output_name(rna_path, output_format),
    )

    log_path = pair_output_path.with_name(f"{pair_output_path.name}.log")
    _configure_file_logging(log_path)

    _log_and_print("Starting DNA-RNA pair windowing job.")
    _log_and_print(f"Loading input files: {pair_path}, {dna_path}, {rna_path}")

    pair_df = _load_input(pair_path)
    dna_df = _load_input(dna_path)
    rna_df = _load_input(rna_path)

    pairs_windowed, dna_windowed, rna_windowed, report = window_pair_dataset(
        pair_df,
        dna_df,
        rna_df,
        window_size=int(args.window_size),
        stride=int(args.stride) if args.stride is not None else None,
        dna_window_size=int(args.dna_window_size)
        if args.dna_window_size is not None
        else None,
        dna_stride=int(args.dna_stride) if args.dna_stride is not None else None,
        rna_window_size=int(args.rna_window_size)
        if args.rna_window_size is not None
        else None,
        rna_stride=int(args.rna_stride) if args.rna_stride is not None else None,
    )

    _log_and_print(
        "Pairs: input {input_rows}, valid {valid_rows}, output {output_rows}".format(
            **report["pairs"]
        )
    )
    if report["pairs"]["skipped_count"]:
        _log_and_print(
            f"Pairs: skipped {report['pairs']['skipped_count']} rows for missing ids."
        )
    if report["pairs"]["missing_dna_refs"] or report["pairs"]["missing_rna_refs"]:
        _log_and_print(
            "Pairs: dropped references missing DNA/RNA "
            f"(dna={report['pairs']['missing_dna_refs']}, "
            f"rna={report['pairs']['missing_rna_refs']})."
        )
    if report["pairs"]["pair_id_collision_count"]:
        _log_and_print(
            "Pairs: resolved {count} pair_id collisions with __dup suffix.".format(
                count=report["pairs"]["pair_id_collision_count"]
            )
        )

    _log_and_print(
        "DNA: input {input_rows}, valid {valid_rows}, output {total_output_rows}".format(
            **report["dna"]
        )
    )
    _log_and_print(
        "RNA: input {input_rows}, valid {valid_rows}, output {total_output_rows}".format(
            **report["rna"]
        )
    )

    _write_output(pairs_windowed, pair_output_path, output_format)
    _write_output(dna_windowed, dna_output_path, output_format)
    _write_output(rna_windowed, rna_output_path, output_format)
    _log_and_print(f"Wrote windowed pairs to {pair_output_path}")
    _log_and_print(f"Wrote windowed DNA to {dna_output_path}")
    _log_and_print(f"Wrote windowed RNA to {rna_output_path}")

    report_path = pair_output_path.with_name(f"{pair_output_path.name}.report.json")
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=True, indent=2)
    _log_and_print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
