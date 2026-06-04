"""Helpers to load sequences from FASTA or id/seq CSV with base validation (U→T)."""

__all__ = ["read_seq_dict", "read_fasta", "read_csv"]

import csv
import warnings
from pathlib import Path

BASES = ["A", "C", "G", "T", "N"]
ALLOWED = set(BASES)


def _normalize_and_validate(seq: str, record: str, path: str) -> str:
    seq = seq.upper().replace("U", "T")
    invalid = set(seq) - ALLOWED
    if invalid:
        inv = "".join(sorted(invalid))
        raise ValueError(
            f"Record '{record}' contains invalid bases '{inv}' in file '{path}'. "
            f"Allowed characters: {', '.join(BASES)}."
        )
    return seq


def _ensure_csv(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    if not path.is_file():
        raise ValueError(f"CSV path must refer to a file: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv file, got: {path}")


def read_fasta(path: str) -> dict[str, str]:
    """
    Load a multi-FASTA file and return an ordered mapping of record name to sequence.

    Rules:
    - Header: uses the token immediately after ">" up to the first whitespace as the record name.
    - Sequence: concatenates subsequent lines until the next header; accepts lower/upper case,
      converts U to T to normalize RNA input.
    - Validation: only A/C/G/T/N are allowed; raises ValueError naming the offending record/file otherwise.

    Args:
        path: Path to a FASTA file.

    Returns:
        Dictionary mapping record names to uppercase sequences (with U replaced by T).
    """
    seqs: dict[str, str] = {}
    name = None
    chunks: list[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                # flush previous
                if name is not None:
                    seqs[name] = _normalize_and_validate("".join(chunks), name, path)
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
        # flush last
        if name is not None:
            seqs[name] = _normalize_and_validate("".join(chunks), name, path)
    return seqs


def read_csv(path: str) -> dict[str, str]:
    """
    Load a CSV with columns `id` and `seq`, return mapping of id->sequence.

    - Header search is case-insensitive (id/ID/Id, seq/SEQ/...).
    - Sequences are normalized to upper case with U converted to T.
    - Validates that only A/C/G/T/N appear; if duplicates occur, keeps the first and warns.
    """
    csv_path = Path(path)
    _ensure_csv(csv_path)

    seqs: dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")

        def _find_col(target: str) -> str | None:
            for col in reader.fieldnames or []:
                if col is not None and col.strip().lower() == target:
                    return col
            return None

        id_col = _find_col("id")
        seq_col = _find_col("seq")
        if not id_col or not seq_col:
            cols = ", ".join(reader.fieldnames)
            raise ValueError(
                f"CSV must contain 'id' and 'seq' columns (got: {cols}) in file '{path}'."
            )

        for i, row in enumerate(reader, start=2):  # data starts on line 2
            rid_raw = (row.get(id_col) or "").strip()
            if not rid_raw:
                raise ValueError(f"Missing id in row {i} of '{path}'.")
            seq_raw = row.get(seq_col) or ""
            seq_norm = _normalize_and_validate(str(seq_raw), rid_raw, path)
            if rid_raw in seqs:
                warnings.warn(
                    f"Duplicate id '{rid_raw}' found in '{path}', keeping the first occurrence.",
                    RuntimeWarning,
                )
                continue
            seqs[rid_raw] = seq_norm

    return seqs


def read_seq_dict(path: str) -> dict[str, str]:
    """
    Load sequences from FASTA or CSV (id, seq columns). Dispatches by suffix.
    """
    suffix = Path(path).suffix.lower()
    if suffix in {".fa", ".fasta"}:
        return read_fasta(path)
    if suffix == ".csv":
        return read_csv(path)
    raise ValueError(
        f"Unsupported sequence file extension '{suffix}' for path '{path}'. "
        "Use FASTA (.fa/.fasta) or CSV with columns 'id' and 'seq'."
    )
