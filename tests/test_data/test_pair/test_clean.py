import pandas as pd

from dnarna.data.pair.clean import (
    DNA_ID_COLUMN,
    DNA_SEQ_COLUMN,
    PAIR_ID_COLUMN,
    RNA_ID_COLUMN,
    RNA_SEQ_COLUMN,
    clean_dnarna_dataset,
    clean_pair_dataframe,
    clean_sequence_dataframe,
)


def test_clean_sequence_dataframe_normalizes_rna_and_dedup():
    df = pd.DataFrame(
        {
            RNA_ID_COLUMN: [1, 2, 3, 1, 4, None],
            RNA_SEQ_COLUMN: ["auu", "", "ACGX", "TTTT", "att", "ACGT"],
        }
    )

    cleaned, report = clean_sequence_dataframe(
        df,
        id_col=RNA_ID_COLUMN,
        seq_col=RNA_SEQ_COLUMN,
        allow_u=True,
        source_label="RNA",
    )

    assert cleaned[RNA_ID_COLUMN].tolist() == ["1"]
    assert cleaned[RNA_SEQ_COLUMN].tolist() == ["ATT"]
    assert report["skipped_count"] == 5
    reason_counts = report["reason_counts"]
    assert reason_counts["empty_seq"] == 1
    assert reason_counts["invalid_seq_chars"] == 1
    assert reason_counts["duplicate_id"] == 1
    assert reason_counts["duplicate_seq"] == 1
    assert reason_counts["missing_id"] == 1
    duplicate_seq_entries = [
        entry for entry in report["skipped"] if entry["reason"] == "duplicate_seq"
    ]
    assert duplicate_seq_entries[0]["details"]["kept_id"] == "1"


def test_clean_pair_dataframe_filters_missing_and_duplicate_pairs():
    df = pd.DataFrame(
        {
            PAIR_ID_COLUMN: ["p1", "p2", "p3", "p4", None, "p5", "p6"],
            DNA_ID_COLUMN: ["d1", "d1", "d2", "d1", "d1", None, "d1"],
            RNA_ID_COLUMN: ["r1", "r1", "r1", "r9", "r1", "r1", None],
        }
    )

    cleaned, report = clean_pair_dataframe(
        df,
        valid_dna_ids={"d1"},
        valid_rna_ids={"r1", "r2"},
    )

    assert cleaned[PAIR_ID_COLUMN].tolist() == ["p1"]
    assert cleaned[DNA_ID_COLUMN].tolist() == ["d1"]
    assert cleaned[RNA_ID_COLUMN].tolist() == ["r1"]
    assert report["skipped_count"] == 6
    reason_counts = report["reason_counts"]
    assert reason_counts["missing_pair_id"] == 1
    assert reason_counts["missing_dna_id"] == 1
    assert reason_counts["missing_rna_id"] == 1
    assert reason_counts["missing_reference"] == 2
    assert reason_counts["duplicate_pair"] == 1


def test_clean_dnarna_dataset_drops_invalid_and_unreferenced_pairs():
    dna_df = pd.DataFrame(
        {
            DNA_ID_COLUMN: ["d1", "d2", "d3"],
            DNA_SEQ_COLUMN: ["acgt", "ACGT", ""],
        }
    )
    rna_df = pd.DataFrame(
        {
            RNA_ID_COLUMN: ["r1", "r2", "r3"],
            RNA_SEQ_COLUMN: ["auu", "AUGX", "AUU"],
        }
    )
    pair_df = pd.DataFrame(
        {
            PAIR_ID_COLUMN: ["p1", "p2", "p3", "p4", "p5"],
            DNA_ID_COLUMN: ["d1", "d2", "d1", "d1", "d3"],
            RNA_ID_COLUMN: ["r1", "r1", "r2", "r1", "r1"],
        }
    )

    cleaned_pairs, cleaned_dna, cleaned_rna, report = clean_dnarna_dataset(
        pair_df, dna_df, rna_df
    )

    assert cleaned_dna[DNA_ID_COLUMN].tolist() == ["d1"]
    assert cleaned_dna[DNA_SEQ_COLUMN].tolist() == ["ACGT"]
    assert cleaned_rna[RNA_ID_COLUMN].tolist() == ["r1"]
    assert cleaned_rna[RNA_SEQ_COLUMN].tolist() == ["ATT"]
    assert cleaned_pairs[PAIR_ID_COLUMN].tolist() == ["p1"]
    assert report["pairs"]["skipped_count"] == 4


def test_clean_dnarna_dataset_remaps_duplicate_sequence_ids():
    dna_df = pd.DataFrame(
        {
            DNA_ID_COLUMN: ["d1", "d2"],
            DNA_SEQ_COLUMN: ["ACGT", "ACGT"],
        }
    )
    rna_df = pd.DataFrame(
        {
            RNA_ID_COLUMN: ["r1"],
            RNA_SEQ_COLUMN: ["AUU"],
        }
    )
    pair_df = pd.DataFrame(
        {
            PAIR_ID_COLUMN: ["p1"],
            DNA_ID_COLUMN: ["d2"],
            RNA_ID_COLUMN: ["r1"],
        }
    )

    cleaned_pairs, cleaned_dna, cleaned_rna, report = clean_dnarna_dataset(
        pair_df, dna_df, rna_df
    )

    assert cleaned_dna[DNA_ID_COLUMN].tolist() == ["d1"]
    assert cleaned_pairs[DNA_ID_COLUMN].tolist() == ["d1"]
    assert report["pairs"]["remapped_references"][DNA_ID_COLUMN] == 1


def test_clean_dnarna_dataset_remaps_multiple_duplicate_sequences():
    dna_df = pd.DataFrame(
        {
            DNA_ID_COLUMN: ["d1", "d2", "d3", "d4", "d5"],
            DNA_SEQ_COLUMN: ["ACGT", "ACGT", "TTTT", "TTTT", "GGGG"],
        }
    )
    rna_df = pd.DataFrame(
        {
            RNA_ID_COLUMN: ["r1", "r2", "r3", "r4", "r5"],
            RNA_SEQ_COLUMN: ["AUU", "AUU", "CCCC", "CCCC", "GAAA"],
        }
    )
    pair_df = pd.DataFrame(
        {
            PAIR_ID_COLUMN: ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9"],
            DNA_ID_COLUMN: ["d2", "d2", "d4", "d4", "d5", "d1", "d3", "d2", "d5"],
            RNA_ID_COLUMN: ["r2", "r1", "r4", "r3", "r5", "r2", "r1", "r5", "r4"],
        }
    )

    cleaned_pairs, cleaned_dna, cleaned_rna, report = clean_dnarna_dataset(
        pair_df, dna_df, rna_df
    )

    assert cleaned_dna[DNA_ID_COLUMN].tolist() == ["d1", "d3", "d5"]
    assert cleaned_rna[RNA_ID_COLUMN].tolist() == ["r1", "r3", "r5"]

    assert report["dna"]["sequence_aliases"] == {"d2": "d1", "d4": "d3"}
    assert report["rna"]["sequence_aliases"] == {"r2": "r1", "r4": "r3"}

    pairs = list(zip(cleaned_pairs[DNA_ID_COLUMN], cleaned_pairs[RNA_ID_COLUMN]))
    assert pairs == [
        ("d1", "r1"),
        ("d3", "r3"),
        ("d5", "r5"),
        ("d3", "r1"),
        ("d1", "r5"),
        ("d5", "r3"),
    ]
    assert report["pairs"]["remapped_references"][DNA_ID_COLUMN] == 5
    assert report["pairs"]["remapped_references"][RNA_ID_COLUMN] == 4
