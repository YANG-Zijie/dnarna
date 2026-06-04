from pathlib import Path

import pandas as pd
import pytest

from dnarna.data.seq import validate as seq_validate


def _call_validate(df: pd.DataFrame, tmp_path, monkeypatch) -> pd.DataFrame:
    """Helper that fakes parquet loading without requiring pyarrow."""
    parquet_path = tmp_path / "data.parquet"
    parquet_path.write_text("")  # ensure the path exists

    def _fake_read_parquet(path):
        if Path(path) != parquet_path:
            raise AssertionError("Unexpected parquet path")
        return df.copy()

    monkeypatch.setattr(seq_validate.pd, "read_parquet", _fake_read_parquet)
    return seq_validate.validate_sequence_file(str(parquet_path))


def test_validate_sequence_file_fills_optional_len_column(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {
            "id": ["s1", "s2"],
            "seq": ["ATCG", "ATCGA"],
            "label": [1, 0],
            "len": [4, pd.NA],
        }
    )
    validated = _call_validate(df, tmp_path, monkeypatch)
    assert list(validated["len"]) == [4, 5]


def test_validate_sequence_file_rejects_mismatched_len(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {
            "id": ["s1"],
            "seq": ["ATCG"],
            "label": [1],
            "len": [999],
        }
    )
    with pytest.raises(ValueError, match="len.*expected 4, got 999"):
        _call_validate(df, tmp_path, monkeypatch)


def test_validate_sequence_dataframe_rejects_duplicate_id():
    df = pd.DataFrame(
        {
            "id": ["s1", "s1"],
            "seq": ["ATCG", "GGGG"],
            "label": [1, 0],
        }
    )
    with pytest.raises(ValueError, match="duplicate id"):
        seq_validate.validate_sequence_dataframe(df)


def test_validate_sequence_dataframe_rejects_duplicate_seq():
    df = pd.DataFrame(
        {
            "id": ["s1", "s2"],
            "seq": ["ATCG", "ATCG"],
            "label": [1, 0],
        }
    )
    with pytest.raises(ValueError, match="already assigned to id"):
        seq_validate.validate_sequence_dataframe(df)


def test_sanitize_sequence_dataframe_default_dedup_id_or_seq():
    df = pd.DataFrame(
        {
            "id": ["a", "a", "b", "c"],
            "seq": ["ATCG", "GGGG", "ATCG", "TTTT"],
        }
    )
    cleaned, skipped = seq_validate.sanitize_sequence_dataframe(df)
    assert cleaned["id"].tolist() == ["a", "c"]
    assert cleaned["seq"].tolist() == ["ATCG", "TTTT"]
    assert [row["reason"] for row in skipped] == ["duplicate_id", "duplicate_seq"]


def test_sanitize_sequence_dataframe_seq_dup_after_id_dedup():
    df = pd.DataFrame(
        {
            "id": ["a", "a", "b", "c"],
            "seq": ["ATCG", "GGGG", "GGGG", "GGGG"],
        }
    )
    cleaned, skipped = seq_validate.sanitize_sequence_dataframe(df)
    assert cleaned["id"].tolist() == ["a", "b"]
    assert cleaned["seq"].tolist() == ["ATCG", "GGGG"]
    assert [row["reason"] for row in skipped] == ["duplicate_id", "duplicate_seq"]
