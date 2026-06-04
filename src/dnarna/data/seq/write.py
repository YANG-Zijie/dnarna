"""Sequence writers for FASTA and id/seq CSV."""

from __future__ import annotations

import csv
from pathlib import Path


def write_fasta(seqs: dict[str, str], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for rid, seq in seqs.items():
            fh.write(f">{rid}\n{seq}\n")


def write_id_seq_csv(seqs: dict[str, str], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "seq"])
        for rid, seq in seqs.items():
            writer.writerow([rid, seq])


__all__ = ["write_fasta", "write_id_seq_csv"]
