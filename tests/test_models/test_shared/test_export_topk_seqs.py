from __future__ import annotations

from pathlib import Path

import pandas as pd

from dnarna.models.shared.predict.export_topk_seqs import export_topk_sequences


def test_export_topk_sequences_csv(tmp_path: Path) -> None:
    seq_csv = tmp_path / "dna.csv"
    pd.DataFrame(
        [
            {"id": "a", "seq": "ACGT"},
            {"id": "b", "seq": "AAAA"},
            {"id": "c", "seq": "TTTT"},
        ]
    ).to_csv(seq_csv, index=False)

    preds_csv = tmp_path / "preds.csv"
    pd.DataFrame(
        [
            {"id": "b", "prob": 0.2, "pred": 0},
            {"id": "a", "prob": 0.9, "pred": 1},
            {"id": "c", "prob": 0.8, "pred": 1},
        ]
    ).to_csv(preds_csv, index=False)

    out_csv = tmp_path / "top2.csv"
    df = export_topk_sequences(
        predictions_path=preds_csv,
        seq_file=seq_csv,
        output=out_csv,
        top_n=2,
    )
    assert df["id"].tolist() == ["a", "c"]

    out = pd.read_csv(out_csv)
    assert out["id"].tolist() == ["a", "c"]
    assert out["seq"].tolist() == ["ACGT", "TTTT"]
    assert out["prob"].tolist() == [0.9, 0.8]
    assert out["pred"].tolist() == [1, 1]


def test_export_topk_sequences_missing_id_errors(tmp_path: Path) -> None:
    seq_csv = tmp_path / "dna.csv"
    pd.DataFrame([{"id": "a", "seq": "ACGT"}]).to_csv(seq_csv, index=False)

    preds_csv = tmp_path / "preds.csv"
    pd.DataFrame([{"id": "missing", "prob": 0.99, "pred": 1}]).to_csv(
        preds_csv, index=False
    )

    out_csv = tmp_path / "top.csv"
    try:
        export_topk_sequences(
            predictions_path=preds_csv,
            seq_file=seq_csv,
            output=out_csv,
        )
        assert False, "expected KeyError for missing sequence ID"
    except KeyError:
        pass


def test_export_topk_sequences_preserves_sequence_metadata(tmp_path: Path) -> None:
    seq_csv = tmp_path / "dna.windowed.csv"
    pd.DataFrame(
        [
            {"id": "d1_win_1", "seq": "AAAA", "parent_id": "d1", "window_index": 1},
            {"id": "d1_win_2", "seq": "CCCC", "parent_id": "d1", "window_index": 2},
            {"id": "d2", "seq": "GGGG", "parent_id": "d2", "window_index": 0},
        ]
    ).to_csv(seq_csv, index=False)

    preds_csv = tmp_path / "preds.csv"
    pd.DataFrame(
        [
            {"id": "d1_win_2", "prob": 0.91, "pred": 1},
            {"id": "d2", "prob": 0.84, "pred": 1},
            {"id": "d1_win_1", "prob": 0.32, "pred": 0},
        ]
    ).to_csv(preds_csv, index=False)

    out_csv = tmp_path / "top2.csv"
    df = export_topk_sequences(
        predictions_path=preds_csv,
        seq_file=seq_csv,
        output=out_csv,
        top_n=2,
    )

    assert df["id"].tolist() == ["d1_win_2", "d2"]
    assert "parent_id" in df.columns
    assert "window_index" in df.columns

    out = pd.read_csv(out_csv)
    assert out["parent_id"].tolist() == ["d1", "d2"]
    assert out["window_index"].tolist() == [2, 0]
