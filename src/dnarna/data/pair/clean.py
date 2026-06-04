"""
Clean DNA-RNA pair datasets with consistent validation and logging.

Input files:
- pairs: pair_id, dna_id, rna_id
   - duplicates are removed based on (dna_id, rna_id)
   - pairs referencing missing DNA/RNA ids are dropped
   - pairs referencing duplicate-sequence ids are remapped to the kept id
- DNA sequences: dna_id, dna_seq
- RNA sequences: rna_id, rna_seq (U is converted to T)

Outputs:
- cleaned pair/DNA/RNA files
- a JSON report with summary stats and removed-id details
- a log file with detailed removal lists

Usage:
```bash
python -m dnarna.data.pair.clean \
    --pair_file path/to/pairs.csv \
    --dna_file path/to/dna.csv \
    --rna_file path/to/rna.csv \
    --output_dir path/to/output_dir \
    --output_format csv
```
"""

from __future__ import annotations

import argparse
import json
import logging
import numbers
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "PAIR_ID_COLUMN",
    "DNA_ID_COLUMN",
    "DNA_SEQ_COLUMN",
    "RNA_ID_COLUMN",
    "RNA_SEQ_COLUMN",
    "clean_dnarna_dataset",
    "clean_pair_dataframe",
    "clean_sequence_dataframe",
    "parse_args",
    "main",
]

PAIR_ID_COLUMN = "pair_id"
DNA_ID_COLUMN = "dna_id"
DNA_SEQ_COLUMN = "dna_seq"
RNA_ID_COLUMN = "rna_id"
RNA_SEQ_COLUMN = "rna_seq"
VALID_BASES: set[str] = {"A", "C", "G", "T"}

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    LOGGER.addHandler(logging.NullHandler())


def _configure_file_logging(log_path: Path) -> None:
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


def _ensure_columns(df: pd.DataFrame, required: list[str], *, source: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"{source} missing required columns: {', '.join(sorted(missing))}"
        )


def _normalize_row_index(idx: Any) -> int | str:
    if isinstance(idx, numbers.Integral):
        return int(idx)
    return str(idx)


def _normalize_id_value(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        if float(value).is_integer():
            return str(int(value))
    text = str(value).strip()
    return text or None


def _normalize_sequence(value: Any, *, allow_u: bool) -> tuple[str | None, str | None]:
    if value is None or pd.isna(value):
        return None, "empty_seq"
    seq = str(value).strip()
    if not seq:
        return None, "empty_seq"
    seq = seq.upper()
    if allow_u:
        seq = seq.replace("U", "T")
    invalid = sorted({ch for ch in seq if ch not in VALID_BASES})
    if invalid:
        return None, "invalid_seq_chars"
    return seq, None


def _summarize_skipped(skipped: list[dict], *, id_key: str) -> tuple[dict, dict]:
    reason_counts: dict[str, int] = {}
    reason_ids: dict[str, list[str | None]] = {}
    for entry in skipped:
        reason = str(entry.get("reason"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        reason_ids.setdefault(reason, []).append(entry.get(id_key))
    return reason_counts, reason_ids


def clean_sequence_dataframe(
    df: pd.DataFrame,
    *,
    id_col: str,
    seq_col: str,
    allow_u: bool,
    source_label: str,
) -> tuple[pd.DataFrame, dict]:
    _ensure_columns(df, [id_col, seq_col], source=source_label)

    seen_ids: set[str] = set()
    seen_seqs: dict[str, str] = {}
    sequence_aliases: dict[str, str] = {}
    cleaned_rows: list[dict] = []
    skipped: list[dict] = []

    for idx, row in df.iterrows():
        row_index = _normalize_row_index(idx)
        raw_id = _normalize_id_value(row.get(id_col))
        if raw_id is None:
            skipped.append(
                {"row_index": row_index, "id": None, "reason": "missing_id"}
            )
            continue

        normalized_seq, reason = _normalize_sequence(row.get(seq_col), allow_u=allow_u)
        if reason:
            skipped.append(
                {"row_index": row_index, "id": raw_id, "reason": reason}
            )
            continue

        if raw_id in seen_ids:
            skipped.append(
                {"row_index": row_index, "id": raw_id, "reason": "duplicate_id"}
            )
            continue

        existing_id = seen_seqs.get(normalized_seq)
        if existing_id is not None and existing_id != raw_id:
            sequence_aliases[raw_id] = existing_id
            skipped.append(
                {
                    "row_index": row_index,
                    "id": raw_id,
                    "reason": "duplicate_seq",
                    "details": {"kept_id": existing_id},
                }
            )
            continue

        seen_ids.add(raw_id)
        seen_seqs[normalized_seq] = raw_id
        cleaned = row.to_dict()
        cleaned[id_col] = raw_id
        cleaned[seq_col] = normalized_seq
        cleaned_rows.append(cleaned)

    cleaned_df = pd.DataFrame.from_records(cleaned_rows, columns=df.columns)
    reason_counts, reason_ids = _summarize_skipped(skipped, id_key="id")
    report = {
        "input_rows": len(df),
        "kept_rows": len(cleaned_df),
        "skipped_count": len(skipped),
        "reason_counts": reason_counts,
        "skipped_ids": reason_ids,
        "skipped": skipped,
        "sequence_alias_count": len(sequence_aliases),
        "sequence_aliases": sequence_aliases,
    }
    return cleaned_df, report


def clean_pair_dataframe(
    df: pd.DataFrame,
    *,
    pair_id_col: str = PAIR_ID_COLUMN,
    dna_id_col: str = DNA_ID_COLUMN,
    rna_id_col: str = RNA_ID_COLUMN,
    valid_dna_ids: set[str],
    valid_rna_ids: set[str],
    dna_aliases: dict[str, str] | None = None,
    rna_aliases: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    _ensure_columns(df, [pair_id_col, dna_id_col, rna_id_col], source="pair file")

    cleaned_rows: list[dict] = []
    skipped: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()
    remapped_dna_refs = 0
    remapped_rna_refs = 0
    dna_aliases = dna_aliases or {}
    rna_aliases = rna_aliases or {}

    for idx, row in df.iterrows():
        row_index = _normalize_row_index(idx)
        pair_id = _normalize_id_value(row.get(pair_id_col))
        if pair_id is None:
            skipped.append(
                {"row_index": row_index, "pair_id": None, "reason": "missing_pair_id"}
            )
            continue

        dna_id = _normalize_id_value(row.get(dna_id_col))
        rna_id = _normalize_id_value(row.get(rna_id_col))
        if dna_id is None:
            skipped.append(
                {
                    "row_index": row_index,
                    "pair_id": pair_id,
                    "reason": "missing_dna_id",
                }
            )
            continue
        if rna_id is None:
            skipped.append(
                {
                    "row_index": row_index,
                    "pair_id": pair_id,
                    "reason": "missing_rna_id",
                }
            )
            continue

        if dna_id in dna_aliases:
            dna_id = dna_aliases[dna_id]
            remapped_dna_refs += 1
        if rna_id in rna_aliases:
            rna_id = rna_aliases[rna_id]
            remapped_rna_refs += 1

        missing_refs: list[str] = []
        if dna_id not in valid_dna_ids:
            missing_refs.append(dna_id_col)
        if rna_id not in valid_rna_ids:
            missing_refs.append(rna_id_col)
        if missing_refs:
            skipped.append(
                {
                    "row_index": row_index,
                    "pair_id": pair_id,
                    "reason": "missing_reference",
                    "details": {"missing": missing_refs},
                }
            )
            continue

        pair_key = (dna_id, rna_id)
        if pair_key in seen_pairs:
            skipped.append(
                {
                    "row_index": row_index,
                    "pair_id": pair_id,
                    "reason": "duplicate_pair",
                }
            )
            continue

        seen_pairs.add(pair_key)
        cleaned = row.to_dict()
        cleaned[pair_id_col] = pair_id
        cleaned[dna_id_col] = dna_id
        cleaned[rna_id_col] = rna_id
        cleaned_rows.append(cleaned)

    cleaned_df = pd.DataFrame.from_records(cleaned_rows, columns=df.columns)
    reason_counts, reason_ids = _summarize_skipped(skipped, id_key="pair_id")
    report = {
        "input_rows": len(df),
        "kept_rows": len(cleaned_df),
        "skipped_count": len(skipped),
        "reason_counts": reason_counts,
        "skipped_ids": reason_ids,
        "skipped": skipped,
        "remapped_references": {
            dna_id_col: remapped_dna_refs,
            rna_id_col: remapped_rna_refs,
        },
    }
    return cleaned_df, report


def clean_dnarna_dataset(
    pair_df: pd.DataFrame,
    dna_df: pd.DataFrame,
    rna_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    cleaned_dna, dna_report = clean_sequence_dataframe(
        dna_df,
        id_col=DNA_ID_COLUMN,
        seq_col=DNA_SEQ_COLUMN,
        allow_u=False,
        source_label="DNA sequence file",
    )
    cleaned_rna, rna_report = clean_sequence_dataframe(
        rna_df,
        id_col=RNA_ID_COLUMN,
        seq_col=RNA_SEQ_COLUMN,
        allow_u=True,
        source_label="RNA sequence file",
    )

    valid_dna_ids = set(cleaned_dna[DNA_ID_COLUMN].tolist())
    valid_rna_ids = set(cleaned_rna[RNA_ID_COLUMN].tolist())

    cleaned_pairs, pair_report = clean_pair_dataframe(
        pair_df,
        valid_dna_ids=valid_dna_ids,
        valid_rna_ids=valid_rna_ids,
        dna_aliases=dna_report.get("sequence_aliases", {}),
        rna_aliases=rna_report.get("sequence_aliases", {}),
    )

    report = {
        "dna": dna_report,
        "rna": rna_report,
        "pairs": pair_report,
    }
    return cleaned_pairs, cleaned_dna, cleaned_rna, report


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
    return f"{input_path.stem}{suffix}"


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


def _log_dataset_report(label: str, report: dict, *, id_label: str) -> None:
    _log_and_print(
        f"{label}: input {report['input_rows']} rows, kept {report['kept_rows']} rows, "
        f"skipped {report['skipped_count']} rows"
    )
    reason_counts = report.get("reason_counts", {})
    for reason, count in reason_counts.items():
        _log_and_print(f"{label}: removed {count} rows for {reason}")
    _log_and_print(
        f"{label}: removed {id_label} lists are saved in the JSON report only."
    )
    remapped = report.get("remapped_references")
    if remapped:
        _log_and_print(
            f"{label}: remapped references ("
            f"{', '.join(f'{key}={value}' for key, value in remapped.items())})"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean DNA-RNA pair datasets (pairs + DNA/RNA sequences)."
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
        help="Output format for cleaned data files.",
    )
    parser.add_argument(
        "--pair_output",
        default=None,
        help="Optional output filename for the cleaned pair file.",
    )
    parser.add_argument(
        "--dna_output",
        default=None,
        help="Optional output filename for the cleaned DNA file.",
    )
    parser.add_argument(
        "--rna_output",
        default=None,
        help="Optional output filename for the cleaned RNA file.",
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
    _log_and_print("Starting DNA-RNA pair cleaning job.")
    _log_and_print(f"Loading input files: {pair_path}, {dna_path}, {rna_path}")

    pair_df = _load_input(pair_path)
    dna_df = _load_input(dna_path)
    rna_df = _load_input(rna_path)
    _log_and_print("Finished loading input files.")

    _log_and_print("Running data cleaning steps...")
    cleaned_pairs, cleaned_dna, cleaned_rna, report = clean_dnarna_dataset(
        pair_df, dna_df, rna_df
    )
    _log_and_print("Finished data cleaning steps.")

    _log_dataset_report("DNA sequences", report["dna"], id_label="id")
    _log_dataset_report("RNA sequences", report["rna"], id_label="id")
    _log_dataset_report("Pairs", report["pairs"], id_label="pair_id")

    _log_and_print("Writing cleaned outputs...")
    _write_output(cleaned_pairs, pair_output_path, output_format)
    _write_output(cleaned_dna, dna_output_path, output_format)
    _write_output(cleaned_rna, rna_output_path, output_format)

    _log_and_print(f"Wrote cleaned pairs to {pair_output_path}")
    _log_and_print(f"Wrote cleaned DNA to {dna_output_path}")
    _log_and_print(f"Wrote cleaned RNA to {rna_output_path}")
    _log_and_print(f"Detailed log saved to {log_path}")

    report_path = pair_output_path.with_name(f"{pair_output_path.name}.report.json")
    report_payload = {
        "input_files": {
            "pairs": str(pair_path),
            "dna": str(dna_path),
            "rna": str(rna_path),
        },
        "output_files": {
            "pairs": str(pair_output_path),
            "dna": str(dna_output_path),
            "rna": str(rna_output_path),
        },
        "report": report,
    }
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report_payload, fh, ensure_ascii=True, indent=2)
    _log_and_print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
