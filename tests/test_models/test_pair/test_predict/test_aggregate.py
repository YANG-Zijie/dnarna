from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dnarna.models.pair.predict.aggregate import (
    COMBINED_SCORE_COLUMN,
    DNA_PARENT_ID_COLUMN,
    POSITIVE_WINDOW_PAIR_COUNT_COLUMN,
    POSITIVE_WINDOW_PAIR_FRACTION_COLUMN,
    PROB_MAX_COLUMN,
    PROB_MEAN_COLUMN,
    RNA_PARENT_ID_COLUMN,
    THRESHOLD_COLUMN,
    WINDOW_PAIR_COUNT_COLUMN,
    aggregate_pair_predictions,
    attach_parent_metadata,
    load_sequence_window_metadata,
    summarize_pair_predictions,
    summarize_pair_predictions_with_thresholds,
)


def test_aggregate_pair_predictions_from_pair_parent() -> None:
    df = pd.DataFrame(
        [
            {
                "pair_id": "p1__d1__r1",
                "pair_parent_id": "p1",
                "dna_id": "d1_win_1",
                "rna_id": "r1_win_1",
                "dna_parent_id": "d1",
                "rna_parent_id": "r1",
                "dna_window_index": 1,
                "rna_window_index": 1,
                "prob": 0.8,
                "pred": 1,
            },
            {
                "pair_id": "p1__d1__r2",
                "pair_parent_id": "p1",
                "dna_id": "d1_win_1",
                "rna_id": "r1_win_2",
                "dna_parent_id": "d1",
                "rna_parent_id": "r1",
                "dna_window_index": 1,
                "rna_window_index": 2,
                "prob": 0.2,
                "pred": 0,
            },
            {
                "pair_id": "p1__d2__r1",
                "pair_parent_id": "p1",
                "dna_id": "d1_win_2",
                "rna_id": "r1_win_1",
                "dna_parent_id": "d1",
                "rna_parent_id": "r1",
                "dna_window_index": 2,
                "rna_window_index": 1,
                "prob": 0.4,
                "pred": 0,
            },
            {
                "pair_id": "p1__d2__r2",
                "pair_parent_id": "p1",
                "dna_id": "d1_win_2",
                "rna_id": "r1_win_2",
                "dna_parent_id": "d1",
                "rna_parent_id": "r1",
                "dna_window_index": 2,
                "rna_window_index": 2,
                "prob": 0.6,
                "pred": 1,
            },
        ]
    )

    summary = aggregate_pair_predictions(
        df,
        pair_id_col="pair_id",
        dna_id_col="dna_id",
        rna_id_col="rna_id",
        threshold=0.5,
    )

    assert summary is not None
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["pair_parent_id"] == "p1"
    assert row[DNA_PARENT_ID_COLUMN] == "d1"
    assert row[RNA_PARENT_ID_COLUMN] == "r1"
    assert row["dna_window_count"] == 2
    assert row["rna_window_count"] == 2
    assert row[WINDOW_PAIR_COUNT_COLUMN] == 4
    assert row[POSITIVE_WINDOW_PAIR_COUNT_COLUMN] == 2
    assert np.isclose(row[PROB_MEAN_COLUMN], 0.5)
    assert np.isclose(row[PROB_MAX_COLUMN], 0.8)
    assert np.isclose(row[COMBINED_SCORE_COLUMN], 1.0 - (0.2 * 0.8 * 0.6 * 0.4))
    assert "prob_noisy_or" not in summary.columns
    assert row["best_window_pair_id"] == "p1__d1__r1"
    assert row["best_dna_id"] == "d1_win_1"
    assert row["best_rna_id"] == "r1_win_1"


def test_attach_parent_metadata_enables_grouping_for_all_pairs(tmp_path: Path) -> None:
    dna_seq = tmp_path / "dna.windowed.csv"
    pd.DataFrame(
        [
            {"id": "d1_win_1", "seq": "AAAA", "parent_id": "d1", "window_index": 1},
            {"id": "d1_win_2", "seq": "CCCC", "parent_id": "d1", "window_index": 2},
        ]
    ).to_csv(dna_seq, index=False)
    rna_seq = tmp_path / "rna.windowed.csv"
    pd.DataFrame(
        [
            {"id": "r1_win_1", "seq": "TTTT", "parent_id": "r1", "window_index": 1},
            {"id": "r1_win_2", "seq": "GGGG", "parent_id": "r1", "window_index": 2},
        ]
    ).to_csv(rna_seq, index=False)

    pred_df = pd.DataFrame(
        [
            {"pair_id": "d1_win_1__r1_win_1", "dna_id": "d1_win_1", "rna_id": "r1_win_1", "prob": 0.7, "pred": 1},
            {"pair_id": "d1_win_1__r1_win_2", "dna_id": "d1_win_1", "rna_id": "r1_win_2", "prob": 0.1, "pred": 0},
            {"pair_id": "d1_win_2__r1_win_1", "dna_id": "d1_win_2", "rna_id": "r1_win_1", "prob": 0.2, "pred": 0},
            {"pair_id": "d1_win_2__r1_win_2", "dna_id": "d1_win_2", "rna_id": "r1_win_2", "prob": 0.9, "pred": 1},
        ]
    )

    dna_meta = load_sequence_window_metadata(
        dna_seq,
        merge_id_col="dna_id",
        prefix="dna",
    )
    rna_meta = load_sequence_window_metadata(
        rna_seq,
        merge_id_col="rna_id",
        prefix="rna",
    )
    enriched = attach_parent_metadata(
        pred_df,
        dna_meta=dna_meta,
        rna_meta=rna_meta,
        dna_id_col="dna_id",
        rna_id_col="rna_id",
    )
    summary = aggregate_pair_predictions(
        enriched,
        pair_id_col="pair_id",
        dna_id_col="dna_id",
        rna_id_col="rna_id",
        threshold=0.5,
    )

    assert summary is not None
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["pair_group_id"] == "d1__r1"
    assert row["dna_window_count"] == 2
    assert row["rna_window_count"] == 2
    assert row["best_window_pair_id"] == "d1_win_2__r1_win_2"
    assert np.isclose(row[PROB_MAX_COLUMN], 0.9)


def test_summarize_pair_predictions_from_existing_predictions(tmp_path: Path) -> None:
    predictions_csv = tmp_path / "pair_predictions.csv"
    pd.DataFrame(
        [
            {"pair_id": "d1_win_1__r1_win_1", "dna_id": "d1_win_1", "rna_id": "r1_win_1", "prob": 0.7, "pred": 1},
            {"pair_id": "d1_win_1__r1_win_2", "dna_id": "d1_win_1", "rna_id": "r1_win_2", "prob": 0.1, "pred": 0},
            {"pair_id": "d1_win_2__r1_win_1", "dna_id": "d1_win_2", "rna_id": "r1_win_1", "prob": 0.2, "pred": 0},
            {"pair_id": "d1_win_2__r1_win_2", "dna_id": "d1_win_2", "rna_id": "r1_win_2", "prob": 0.9, "pred": 1},
        ]
    ).to_csv(predictions_csv, index=False)

    dna_seq = tmp_path / "dna.windowed.csv"
    pd.DataFrame(
        [
            {"id": "d1_win_1", "seq": "AAAA", "parent_id": "d1", "window_index": 1},
            {"id": "d1_win_2", "seq": "CCCC", "parent_id": "d1", "window_index": 2},
        ]
    ).to_csv(dna_seq, index=False)
    rna_seq = tmp_path / "rna.windowed.csv"
    pd.DataFrame(
        [
            {"id": "r1_win_1", "seq": "TTTT", "parent_id": "r1", "window_index": 1},
            {"id": "r1_win_2", "seq": "GGGG", "parent_id": "r1", "window_index": 2},
        ]
    ).to_csv(rna_seq, index=False)

    summary_df, summary_path = summarize_pair_predictions(
        predictions_path=predictions_csv,
        dna_seq_file=dna_seq,
        rna_seq_file=rna_seq,
        threshold=0.5,
        chunksize=2,
    )

    assert summary_path == tmp_path / "pair_predictions.summary.csv"
    assert summary_path.exists()
    assert len(summary_df) == 1
    row = summary_df.iloc[0]
    assert row["pair_group_id"] == "d1__r1"
    assert row[WINDOW_PAIR_COUNT_COLUMN] == 4
    assert row[POSITIVE_WINDOW_PAIR_COUNT_COLUMN] == 2
    assert np.isclose(row[POSITIVE_WINDOW_PAIR_FRACTION_COLUMN], 0.5)


def test_summarize_pair_predictions_from_pairs_file_metadata(tmp_path: Path) -> None:
    predictions_csv = tmp_path / "pair_predictions.csv"
    pd.DataFrame(
        [
            {"pair_id": "p1__d1__r1", "dna_id": "d1_win_1", "rna_id": "r1_win_1", "prob": 0.4, "pred": 0},
            {"pair_id": "p1__d1__r2", "dna_id": "d1_win_1", "rna_id": "r1_win_2", "prob": 0.6, "pred": 1},
        ]
    ).to_csv(predictions_csv, index=False)

    pairs_csv = tmp_path / "pairs.windowed.csv"
    pd.DataFrame(
        [
            {
                "pair_id": "p1__d1__r1",
                "pair_parent_id": "p1",
                "dna_parent_id": "d1",
                "rna_parent_id": "r1",
                "dna_window_index": 1,
                "rna_window_index": 1,
            },
            {
                "pair_id": "p1__d1__r2",
                "pair_parent_id": "p1",
                "dna_parent_id": "d1",
                "rna_parent_id": "r1",
                "dna_window_index": 1,
                "rna_window_index": 2,
            },
        ]
    ).to_csv(pairs_csv, index=False)

    summary_df, summary_path = summarize_pair_predictions(
        predictions_path=predictions_csv,
        pairs_file=pairs_csv,
        output_path=tmp_path / "custom_summary.csv",
        threshold=0.5,
        ignore_pred_col=True,
        chunksize=1,
    )

    assert summary_path == tmp_path / "custom_summary.csv"
    assert summary_path.exists()
    assert len(summary_df) == 1
    row = summary_df.iloc[0]
    assert row["pair_parent_id"] == "p1"
    assert row[POSITIVE_WINDOW_PAIR_COUNT_COLUMN] == 1
    assert np.isclose(row[PROB_MEAN_COLUMN], 0.5)


def test_summarize_pair_predictions_streaming_across_multiple_chunks(tmp_path: Path) -> None:
    predictions_csv = tmp_path / "pair_predictions.csv"
    pd.DataFrame(
        [
            {"pair_id": "d1_win_1__r1_win_1", "dna_id": "d1_win_1", "rna_id": "r1_win_1", "prob": 0.9, "pred": 1},
            {"pair_id": "d1_win_1__r1_win_2", "dna_id": "d1_win_1", "rna_id": "r1_win_2", "prob": 0.2, "pred": 0},
            {"pair_id": "d1_win_2__r1_win_1", "dna_id": "d1_win_2", "rna_id": "r1_win_1", "prob": 0.3, "pred": 0},
            {"pair_id": "d2_win_1__r2_win_1", "dna_id": "d2_win_1", "rna_id": "r2_win_1", "prob": 0.7, "pred": 1},
            {"pair_id": "d2_win_1__r2_win_2", "dna_id": "d2_win_1", "rna_id": "r2_win_2", "prob": 0.8, "pred": 1},
            {"pair_id": "d1_win_2__r1_win_2", "dna_id": "d1_win_2", "rna_id": "r1_win_2", "prob": 0.4, "pred": 0},
        ]
    ).to_csv(predictions_csv, index=False)

    dna_seq = tmp_path / "dna.windowed.csv"
    pd.DataFrame(
        [
            {"id": "d1_win_1", "seq": "AAAA", "parent_id": "d1", "window_index": 1},
            {"id": "d1_win_2", "seq": "CCCC", "parent_id": "d1", "window_index": 2},
            {"id": "d2_win_1", "seq": "GGGG", "parent_id": "d2", "window_index": 1},
        ]
    ).to_csv(dna_seq, index=False)
    rna_seq = tmp_path / "rna.windowed.csv"
    pd.DataFrame(
        [
            {"id": "r1_win_1", "seq": "TTTT", "parent_id": "r1", "window_index": 1},
            {"id": "r1_win_2", "seq": "GGGG", "parent_id": "r1", "window_index": 2},
            {"id": "r2_win_1", "seq": "ATAT", "parent_id": "r2", "window_index": 1},
            {"id": "r2_win_2", "seq": "CGCG", "parent_id": "r2", "window_index": 2},
        ]
    ).to_csv(rna_seq, index=False)

    summary_df, _ = summarize_pair_predictions(
        predictions_path=predictions_csv,
        dna_seq_file=dna_seq,
        rna_seq_file=rna_seq,
        threshold=0.5,
        chunksize=2,
    )

    assert len(summary_df) == 2
    top = summary_df.iloc[0]
    second = summary_df.iloc[1]
    assert {top["pair_group_id"], second["pair_group_id"]} == {"d1__r1", "d2__r2"}
    d1_row = summary_df.loc[summary_df["pair_group_id"] == "d1__r1"].iloc[0]
    assert d1_row[WINDOW_PAIR_COUNT_COLUMN] == 4
    assert d1_row[POSITIVE_WINDOW_PAIR_COUNT_COLUMN] == 1
    assert np.isclose(d1_row[PROB_MEAN_COLUMN], (0.9 + 0.2 + 0.3 + 0.4) / 4)


def test_summarize_pair_predictions_infers_parent_ids_from_window_suffix(tmp_path: Path) -> None:
    predictions_csv = tmp_path / "pair_predictions.csv"
    pd.DataFrame(
        [
            {"pair_id": "d1_win_1__r1_win_1", "dna_id": "d1_win_1", "rna_id": "r1_win_1", "prob": 0.9, "pred": 1},
            {"pair_id": "d1_win_2__r1_win_1", "dna_id": "d1_win_2", "rna_id": "r1_win_1", "prob": 0.1, "pred": 0},
            {"pair_id": "d1_win_1__r1_win_2", "dna_id": "d1_win_1", "rna_id": "r1_win_2", "prob": 0.2, "pred": 0},
            {"pair_id": "d1_win_2__r1_win_2", "dna_id": "d1_win_2", "rna_id": "r1_win_2", "prob": 0.8, "pred": 1},
        ]
    ).to_csv(predictions_csv, index=False)

    summary_df, _ = summarize_pair_predictions(
        predictions_path=predictions_csv,
        threshold=0.5,
        chunksize=2,
    )

    assert len(summary_df) == 1
    row = summary_df.iloc[0]
    assert row[DNA_PARENT_ID_COLUMN] == "d1"
    assert row[RNA_PARENT_ID_COLUMN] == "r1"
    assert row["pair_group_id"] == "d1__r1"
    assert row[WINDOW_PAIR_COUNT_COLUMN] == 4
    assert row[POSITIVE_WINDOW_PAIR_COUNT_COLUMN] == 2


def test_summarize_pair_predictions_writes_threshold_summary(tmp_path: Path) -> None:
    predictions_csv = tmp_path / "pair_predictions.csv"
    pd.DataFrame(
        [
            {"pair_id": "d1_win_1__r1_win_1", "dna_id": "d1_win_1", "rna_id": "r1_win_1", "prob": 0.95, "pred": 1},
            {"pair_id": "d1_win_2__r1_win_1", "dna_id": "d1_win_2", "rna_id": "r1_win_1", "prob": 0.72, "pred": 1},
            {"pair_id": "d1_win_1__r1_win_2", "dna_id": "d1_win_1", "rna_id": "r1_win_2", "prob": 0.45, "pred": 0},
            {"pair_id": "d1_win_2__r1_win_2", "dna_id": "d1_win_2", "rna_id": "r1_win_2", "prob": 0.12, "pred": 0},
        ]
    ).to_csv(predictions_csv, index=False)

    summary_df, summary_path, threshold_df, threshold_path = (
        summarize_pair_predictions_with_thresholds(
            predictions_path=predictions_csv,
            threshold=0.5,
            thresholds=[0.5, 0.8],
            chunksize=2,
        )
    )

    assert summary_path == tmp_path / "pair_predictions.summary.csv"
    assert threshold_path == tmp_path / "pair_predictions.summary.by_threshold.csv"
    assert summary_df is not None
    assert threshold_df is not None
    assert threshold_path.exists()
    assert len(threshold_df) == 2

    threshold_05 = threshold_df.loc[threshold_df[THRESHOLD_COLUMN] == 0.5].iloc[0]
    threshold_08 = threshold_df.loc[threshold_df[THRESHOLD_COLUMN] == 0.8].iloc[0]

    assert threshold_05["pair_group_id"] == "d1__r1"
    assert threshold_05[WINDOW_PAIR_COUNT_COLUMN] == 4
    assert threshold_05[POSITIVE_WINDOW_PAIR_COUNT_COLUMN] == 2
    assert np.isclose(threshold_05[POSITIVE_WINDOW_PAIR_FRACTION_COLUMN], 0.5)
    assert "combined_score" not in threshold_df.columns
    assert "combined_pred" not in threshold_df.columns

    assert threshold_08[POSITIVE_WINDOW_PAIR_COUNT_COLUMN] == 1
    assert np.isclose(threshold_08[POSITIVE_WINDOW_PAIR_FRACTION_COLUMN], 0.25)
