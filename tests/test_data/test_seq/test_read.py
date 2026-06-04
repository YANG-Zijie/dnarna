from pathlib import Path

import pytest

from dnarna.data.seq.read import read_csv, read_fasta, read_seq_dict


def test_read_fasta_parses_multiple_records(tmp_path: Path):
    fa = tmp_path / "sample.fa"
    fa.write_text(
        ">rna1 desc\nacg\nu\n\n>rna2 other\nNNnn\n",
        encoding="utf-8",
    )

    seqs = read_fasta(str(fa))

    assert seqs == {"rna1": "ACGT", "rna2": "NNNN"}


def test_read_fasta_raises_on_invalid_base(tmp_path: Path):
    fa = tmp_path / "bad.fa"
    fa.write_text(">bad\nACGX\n", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        read_fasta(str(fa))

    assert "bad" in str(excinfo.value)
    assert "invalid bases 'X'" in str(excinfo.value)


def test_read_csv_parses_and_normalizes(tmp_path: Path):
    csv_path = tmp_path / "samples.csv"
    csv_path.write_text("id,seq\nr1,acgu\nr2,NNnn\n", encoding="utf-8")

    seqs = read_csv(str(csv_path))

    assert seqs == {"r1": "ACGT", "r2": "NNNN"}


def test_read_csv_raises_on_invalid_base(tmp_path: Path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("id,seq\nr1,ACGX\n", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        read_csv(str(csv_path))

    assert "invalid bases 'X'" in str(excinfo.value)


def test_read_csv_numeric_ids_remain_strings(tmp_path: Path):
    csv_path = tmp_path / "numeric.csv"
    csv_path.write_text("id,seq\n1,AAA\n2,TTT\n", encoding="utf-8")

    seqs = read_csv(str(csv_path))

    assert seqs == {"1": "AAA", "2": "TTT"}
    assert all(isinstance(k, str) for k in seqs)


def test_read_seq_dict_dispatches_fasta_and_csv(tmp_path: Path):
    fasta_path = tmp_path / "sample.fa"
    fasta_path.write_text(">rna1\nACGT\n", encoding="utf-8")
    fasta_long = tmp_path / "sample.fasta"
    fasta_long.write_text(">rna_long\nTTTT\n", encoding="utf-8")
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("id,seq\nr_csv,TTT\n", encoding="utf-8")

    assert read_seq_dict(str(fasta_path)) == {"rna1": "ACGT"}
    assert read_seq_dict(str(fasta_long)) == {"rna_long": "TTTT"}
    assert read_seq_dict(str(csv_path)) == {"r_csv": "TTT"}

    with pytest.raises(ValueError):
        read_seq_dict(str(tmp_path / "unknown.txt"))


def test_read_csv_duplicates_keep_first_and_warn(tmp_path: Path):
    csv_path = tmp_path / "dup.csv"
    csv_path.write_text("id,seq\nr1,AAA\nr1,TTT\n", encoding="utf-8")

    with pytest.warns(RuntimeWarning):
        seqs = read_csv(str(csv_path))

    assert seqs == {"r1": "AAA"}
