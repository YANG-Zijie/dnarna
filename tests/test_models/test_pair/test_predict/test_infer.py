from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import dnarna.models.pair.predict.infer as pair_infer


def _make_cfg(
    tmp_path: Path,
    *,
    dna_seq_file: str,
    rna_seq_file: str,
    summary_only: bool,
) -> pair_infer.InferConfig:
    return pair_infer.InferConfig(
        pairs_file="",
        dna_embeddings="",
        rna_embeddings="",
        checkpoint="",
        output=str(tmp_path / "pair_predictions.csv"),
        output_dir=str(tmp_path),
        dna_seq_file=dna_seq_file,
        rna_seq_file=rna_seq_file,
        pair_id_col="pair_id",
        dna_id_col="dna_id",
        rna_id_col="rna_id",
        feature_mode="concat",
        chunk_size=16,
        num_workers=1,
        max_pairs=0,
        all_pairs=True,
        dna_block_size=2,
        rna_block_size=1,
        max_dna=0,
        max_rna=0,
        batch_size=8,
        device="cpu",
        threshold=0.5,
        progress=False,
        summary_only=summary_only,
    )


def test_infer_all_pairs_summary_only_writes_summary_without_raw_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dna_seq = tmp_path / "dna.windowed.csv"
    pd.DataFrame(
        [
            {"id": "d1_win_0", "parent_id": "d1", "window_index": 0},
            {"id": "d1_win_1", "parent_id": "d1", "window_index": 1},
        ]
    ).to_csv(dna_seq, index=False)
    rna_seq = tmp_path / "rna.windowed.csv"
    pd.DataFrame(
        [
            {"id": "r1_win_0", "parent_id": "r1", "window_index": 0},
        ]
    ).to_csv(rna_seq, index=False)

    cfg = _make_cfg(
        tmp_path,
        dna_seq_file=str(dna_seq),
        rna_seq_file=str(rna_seq),
        summary_only=True,
    )
    output_path = Path(cfg.output)

    monkeypatch.setattr(
        pair_infer,
        "build_pair_features",
        lambda *args, **kwargs: np.zeros((len(args[2]), 1), dtype=np.float32),
    )
    monkeypatch.setattr(
        pair_infer,
        "predict_probabilities",
        lambda model, loader, device, show_progress=False: np.array(
            [0.9, 0.2], dtype=np.float32
        ),
    )

    stats = pair_infer._infer_all_pairs(
        cfg,
        dna_embeddings=np.zeros((2, 1), dtype=np.float32),
        dna_ids=["d1_win_0", "d1_win_1"],
        rna_embeddings=np.zeros((1, 1), dtype=np.float32),
        rna_ids=["r1_win_0"],
        model=torch.nn.Linear(1, 1),
        mean=np.zeros(1, dtype=np.float32),
        std=np.ones(1, dtype=np.float32),
        device=torch.device("cpu"),
    )

    summary_path = output_path.with_name(f"{output_path.stem}.summary{output_path.suffix}")
    assert not output_path.exists()
    assert summary_path.exists()
    assert stats["summary_only"] is True
    assert stats["summary_output"] == str(summary_path)

    summary_df = pd.read_csv(summary_path)
    assert len(summary_df) == 1
    row = summary_df.iloc[0]
    assert row["pair_group_id"] == "d1__r1"
    assert int(row["window_pair_count"]) == 2
    assert int(row["positive_window_pair_count"]) == 1


def test_infer_all_pairs_summary_only_requires_sequence_metadata(tmp_path: Path) -> None:
    cfg = _make_cfg(
        tmp_path,
        dna_seq_file="",
        rna_seq_file="",
        summary_only=True,
    )

    with pytest.raises(ValueError, match="requires --dna_seq_file and --rna_seq_file"):
        pair_infer._infer_all_pairs(
            cfg,
            dna_embeddings=np.zeros((1, 1), dtype=np.float32),
            dna_ids=["d1_win_0"],
            rna_embeddings=np.zeros((1, 1), dtype=np.float32),
            rna_ids=["r1_win_0"],
            model=torch.nn.Linear(1, 1),
            mean=np.zeros(1, dtype=np.float32),
            std=np.ones(1, dtype=np.float32),
            device=torch.device("cpu"),
        )


def test_infer_all_pairs_summary_only_streams_multiple_parent_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dna_seq = tmp_path / "dna.windowed.csv"
    pd.DataFrame(
        [
            {"id": "d1_win_0", "parent_id": "d1", "window_index": 0},
            {"id": "d2_win_0", "parent_id": "d2", "window_index": 0},
        ]
    ).to_csv(dna_seq, index=False)
    rna_seq = tmp_path / "rna.windowed.csv"
    pd.DataFrame(
        [
            {"id": "r1_win_0", "parent_id": "r1", "window_index": 0},
            {"id": "r2_win_0", "parent_id": "r2", "window_index": 0},
        ]
    ).to_csv(rna_seq, index=False)

    cfg = _make_cfg(
        tmp_path,
        dna_seq_file=str(dna_seq),
        rna_seq_file=str(rna_seq),
        summary_only=True,
    )
    cfg.dna_block_size = 1
    cfg.rna_block_size = 1
    output_path = Path(cfg.output)

    monkeypatch.setattr(
        pair_infer,
        "build_pair_features",
        lambda *args, **kwargs: np.zeros((len(args[2]), 1), dtype=np.float32),
    )
    monkeypatch.setattr(
        pair_infer,
        "predict_probabilities",
        lambda model, loader, device, show_progress=False: np.full(
            len(loader.dataset), 0.6, dtype=np.float32
        ),
    )

    stats = pair_infer._infer_all_pairs(
        cfg,
        dna_embeddings=np.zeros((2, 1), dtype=np.float32),
        dna_ids=["d1_win_0", "d2_win_0"],
        rna_embeddings=np.zeros((2, 1), dtype=np.float32),
        rna_ids=["r1_win_0", "r2_win_0"],
        model=torch.nn.Linear(1, 1),
        mean=np.zeros(1, dtype=np.float32),
        std=np.ones(1, dtype=np.float32),
        device=torch.device("cpu"),
    )

    summary_path = output_path.with_name(f"{output_path.stem}.summary{output_path.suffix}")
    summary_df = pd.read_csv(summary_path)
    assert not output_path.exists()
    assert len(summary_df) == 4
    assert set(summary_df["pair_group_id"]) == {
        "d1__r1",
        "d1__r2",
        "d2__r1",
        "d2__r2",
    }
    assert stats["processed_pairs"] == 4
    assert stats["summary"]["group_count"] == 4
