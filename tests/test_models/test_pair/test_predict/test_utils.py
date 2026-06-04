import numpy as np
import pytest

from dnarna.models.pair.predict.utils import build_pair_features, sample_negative_pairs


def test_build_pair_features_concat():
    dna_emb = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    rna_emb = np.array([[10.0, 11.0], [12.0, 13.0]], dtype=np.float32)
    dna_idx = np.array([0, 1], dtype=int)
    rna_idx = np.array([1, 0], dtype=int)

    features = build_pair_features(
        dna_emb,
        rna_emb,
        dna_idx,
        rna_idx,
        mode="concat",
        chunk_size=1,
    )

    expected = np.array(
        [[1.0, 2.0, 12.0, 13.0], [3.0, 4.0, 10.0, 11.0]],
        dtype=np.float32,
    )
    assert features.shape == expected.shape
    assert np.allclose(features, expected)


def test_build_pair_features_all_requires_matching_dims():
    dna_emb = np.zeros((2, 3), dtype=np.float32)
    rna_emb = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        build_pair_features(
            dna_emb,
            rna_emb,
            np.array([0, 1], dtype=int),
            np.array([0, 1], dtype=int),
            mode="all",
        )


def test_sample_negative_pairs_excludes_positives():
    pos_pairs = {("d1", "r1"), ("d2", "r2")}
    dna_ids = ["d1", "d2", "d3"]
    rna_ids = ["r1", "r2", "r3"]

    negatives = sample_negative_pairs(
        pos_pairs=pos_pairs,
        dna_ids=dna_ids,
        rna_ids=rna_ids,
        n_samples=3,
        seed=123,
    )

    assert len(negatives) > 0
    assert all(pair not in pos_pairs for pair in negatives)
