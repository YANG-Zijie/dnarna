from pathlib import Path

import pandas as pd
import pytest

from dnarna.data.seq.merge_pos_neg import merge_positive_negative


def _write_csv(df: pd.DataFrame, path: Path) -> str:
    df.to_csv(path, index=False)
    return str(path)


def test_merge_generates_ids_and_drops_cross_label_duplicates(tmp_path):
    pos_df = pd.DataFrame({"seq": ["ATCG", "ATCG", "GGGG", "CCCC"]})
    neg_df = pd.DataFrame(
        [
            {"id": "neg-1", "seq": "TTTT"},
            {"id": "", "seq": "GGGG"},
            {"id": "neg-3", "seq": "AAAA"},
        ]
    )
    pos_path = _write_csv(pos_df, tmp_path / "pos.csv")
    neg_path = _write_csv(neg_df, tmp_path / "neg.csv")

    merged = merge_positive_negative(pos_path, neg_path)

    assert sorted(merged["seq"].unique()) == ["AAAA", "ATCG", "CCCC", "TTTT"]
    assert "GGGG" not in merged["seq"].values  # removed due to cross-label clash

    # IDs are non-empty and consistent for duplicated sequences.
    assert merged["id"].map(lambda v: bool(str(v).strip())).all()
    atcg_ids = merged.loc[merged["seq"] == "ATCG", "id"].unique()
    assert len(atcg_ids) == 1

    # Labels preserved
    assert merged.loc[merged["seq"] == "ATCG", "label"].eq(1).all()
    assert merged.loc[merged["seq"] == "AAAA", "label"].eq(0).all()


def test_merge_raises_on_conflicting_sequence_ids(tmp_path):
    pos_df = pd.DataFrame({"id": ["shared"], "seq": ["ATCG"]})
    neg_df = pd.DataFrame({"id": ["shared"], "seq": ["GGGG"]})
    pos_path = _write_csv(pos_df, tmp_path / "pos.csv")
    neg_path = _write_csv(neg_df, tmp_path / "neg.csv")

    with pytest.raises(ValueError, match="already used"):
        merge_positive_negative(pos_path, neg_path)
