import pandas as pd

from dnarna.data.pair.window import window_pair_dataset


def test_window_pair_dataset_generates_windowed_ids_and_pairs():
    pair_df = pd.DataFrame({"pair_id": ["p1"], "dna_id": ["d1"], "rna_id": ["r1"]})
    dna_df = pd.DataFrame({"dna_id": ["d1"], "dna_seq": ["A" * 6]})
    rna_df = pd.DataFrame({"rna_id": ["r1"], "rna_seq": ["T" * 5]})

    pairs_w, dna_w, rna_w, report = window_pair_dataset(
        pair_df,
        dna_df,
        rna_df,
        dna_window_size=4,
        dna_stride=2,
        rna_window_size=4,
        rna_stride=2,
    )

    assert set(dna_w["dna_id"]) == {"d1_win_1", "d1_win_2"}
    assert set(rna_w["rna_id"]) == {"r1_win_1", "r1_win_2"}
    assert set(pairs_w["pair_id"]) == {
        "p1__d1__r1",
        "p1__d1__r2",
        "p1__d2__r1",
        "p1__d2__r2",
    }
    assert pairs_w["pair_id"].is_unique
    assert (pairs_w["pair_parent_id"] == "p1").all()
    assert report["pairs"]["output_rows"] == 4


def test_window_pair_dataset_supports_different_window_sizes():
    pair_df = pd.DataFrame({"pair_id": ["p1"], "dna_id": ["d1"], "rna_id": ["r1"]})
    dna_df = pd.DataFrame({"dna_id": ["d1"], "dna_seq": ["A" * 6]})
    rna_df = pd.DataFrame({"rna_id": ["r1"], "rna_seq": ["T" * 5]})

    pairs_w, dna_w, rna_w, _ = window_pair_dataset(
        pair_df,
        dna_df,
        rna_df,
        dna_window_size=4,
        dna_stride=2,
        rna_window_size=6,
    )

    assert set(dna_w["dna_id"]) == {"d1_win_1", "d1_win_2"}
    assert set(rna_w["rna_id"]) == {"r1"}
    assert set(pairs_w["pair_id"]) == {"p1__d1__r0", "p1__d2__r0"}
