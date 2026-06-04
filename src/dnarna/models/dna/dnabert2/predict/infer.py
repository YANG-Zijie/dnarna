"""
Run inference with a trained DNABERT-2 classifier on precomputed embeddings.

Example:
    python -m dnarna.models.dna.dnabert2.predict.infer \
        --embeddings_npz data/dna_embeddings.npz \
        --checkpoint outputs/dnabert2_clf/model.pt \
        --output outputs/dnabert2_clf/predictions.csv \
        --device cuda:0

Full example (overriding optional flags):
    python -m dnarna.models.dna.dnabert2.predict.infer \
        --embeddings_npz data/dna_embeddings.npz \
        --checkpoint outputs/dnabert2_clf/model.pt \
        --output outputs/dnabert2_clf/predictions.csv \
        --batch_size 512 \
        --threshold 0.5 \
        --device cuda:0 \
        --no_progress

Inputs:
    - embeddings_npz: .npz created by ``dnarna.models.dna.dnabert2.embed`` (expects
      ``embeddings`` + ``ids`` keys).
    - checkpoint: model.pt produced by ``dnarna.models.dna.dnabert2.predict.train``.

Outputs:
    - CSV/TSV/Parquet/JSON with per-ID probabilities and binary predictions.
"""

from __future__ import annotations

import argparse
import logging

from dnarna.models.shared.predict.infer import (
    EmbeddingInferConfig,
    run_embedding_classifier_inference,
)
from dnarna.models.shared.predict.train import setup_logging

LOGGER = logging.getLogger("dnabert2_infer")
InferConfig = EmbeddingInferConfig


def _parse_args() -> InferConfig:
    parser = argparse.ArgumentParser(
        description="Run DNABERT-2 classifier inference on saved embeddings."
    )
    parser.add_argument(
        "--embeddings_npz",
        required=True,
        help="Combined embeddings .npz with embeddings and ids.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="model.pt produced by the training script.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination for predictions (csv/tsv/parquet/json).",
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", default=None, help="Torch device (e.g., cuda:0).")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability cutoff for positive predictions.",
    )
    parser.add_argument(
        "--plot",
        default=None,
        help="Optional path to save a score distribution plot (pdf/png).",
    )
    parser.add_argument(
        "--plot_bins",
        type=int,
        default=50,
        help="Number of histogram bins for score distribution plot.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable info-level logging; otherwise warnings only.",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable tqdm progress bars for inference.",
    )
    args = parser.parse_args()
    setup_logging(
        log_file=None,
        level=logging.INFO if args.verbose else logging.WARNING,
        quiet_tqdm=False,
    )
    return InferConfig(
        embeddings_npz=args.embeddings_npz,
        checkpoint=args.checkpoint,
        output=args.output,
        batch_size=max(1, args.batch_size),
        device=args.device,
        threshold=float(args.threshold),
        progress=not args.no_progress,
        verbose=args.verbose,
        plot_path=args.plot,
        plot_bins=max(1, int(args.plot_bins)),
    )


def _run(cfg: InferConfig) -> None:
    run_embedding_classifier_inference(cfg, logger=LOGGER)


def main() -> None:
    cfg = _parse_args()
    _run(cfg)


if __name__ == "__main__":
    main()


__all__ = ["InferConfig", "main", "_run"]
