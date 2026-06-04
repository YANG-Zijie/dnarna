"""
Run an end-to-end job: embeddings + optional top-K filtering + pair prediction.

This CLI keeps the logic decoupled:
  - model scoring/top-K selection happens in `dnarna.app.pipeline`
  - pair predictions are produced by the trained pair classifier
  - when `--output_dir` is provided, artifacts are persisted under:
      - inputs/ (raw CSV inputs)
      - processed/ (windowed inputs + embeddings)
      - predictions/ (per-sequence scores)
      - topk/ (filtered inputs)
      - pair_predictions/ (pair inference outputs)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dnarna.app.pipeline import AppJobConfig, run_app_job
from dnarna.data.seq.read import read_seq_dict

LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="One-click pipeline: embeddings + optional top-K + pair prediction."
    )
    p.add_argument("--dna", required=True, help="DNA input file (.fa/.fasta/.csv)")
    p.add_argument("--rna", required=True, help="RNA input file (.fa/.fasta/.csv)")

    p.add_argument(
        "--output_dir",
        default=None,
        help="If set, write windowed inputs, scores, and top-K outputs to this directory.",
    )

    p.add_argument("--dnabert_checkpoint", default=None)
    p.add_argument("--rnafm_checkpoint", default=None)
    p.add_argument("--pair_checkpoint", required=True)
    p.add_argument("--top_k_dna", type=int, default=0)
    p.add_argument("--top_k_rna", type=int, default=0)
    p.add_argument("--dnabert_backbone", default="zhihan1996/DNABERT-2-117M")
    p.add_argument("--rnafm_variant", default="rna")
    p.add_argument(
        "--dna_device",
        default=None,
        help="Optional device for DNABERT-2 embedding/scoring (e.g. cuda:0).",
    )
    p.add_argument(
        "--rna_device",
        default=None,
        help="Optional device for RNA-FM embedding/scoring (e.g. cuda:1).",
    )
    p.add_argument("--dna_embed_batch_size", type=int, default=1024)
    p.add_argument("--rna_embed_batch_size", type=int, default=256)
    p.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for per-sequence classifier inference (top-K only).",
    )
    p.add_argument("--score_threshold", type=float, default=0.5)
    p.add_argument("--pair_feature_mode", default="concat")
    p.add_argument("--pair_dna_block_size", type=int, default=64)
    p.add_argument("--pair_rna_block_size", type=int, default=64)
    p.add_argument("--pair_batch_size", type=int, default=4096)
    p.add_argument("--pair_chunk_size", type=int, default=4096)
    p.add_argument("--pair_num_workers", type=int, default=1)
    p.add_argument("--pair_threshold", type=float, default=0.5)
    p.add_argument("--pair_device", default=None)
    p.add_argument("--pair_max_dna", type=int, default=0)
    p.add_argument("--pair_max_rna", type=int, default=0)
    p.add_argument(
        "--no_save_artifacts",
        action="store_true",
        help="Do not save dna_scores/rna_scores and used inputs (even if output_dir is set).",
    )

    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    dna = read_seq_dict(args.dna)
    rna = read_seq_dict(args.rna)

    cfg = AppJobConfig(
        output_dir=str(args.output_dir) if args.output_dir else None,
        dnabert_checkpoint=str(args.dnabert_checkpoint)
        if args.dnabert_checkpoint
        else None,
        rnafm_checkpoint=str(args.rnafm_checkpoint) if args.rnafm_checkpoint else None,
        pair_checkpoint=str(args.pair_checkpoint) if args.pair_checkpoint else None,
        top_k_dna=int(args.top_k_dna) if int(args.top_k_dna) > 0 else 0,
        top_k_rna=int(args.top_k_rna) if int(args.top_k_rna) > 0 else 0,
        dnabert_backbone=str(args.dnabert_backbone),
        rnafm_variant=str(args.rnafm_variant),
        dna_device=str(args.dna_device) if args.dna_device else None,
        rna_device=str(args.rna_device) if args.rna_device else None,
        dna_embed_batch_size=int(args.dna_embed_batch_size),
        rna_embed_batch_size=int(args.rna_embed_batch_size),
        score_batch_size=int(args.batch_size),
        score_threshold=float(args.score_threshold),
        pair_feature_mode=str(args.pair_feature_mode),
        pair_dna_block_size=int(args.pair_dna_block_size),
        pair_rna_block_size=int(args.pair_rna_block_size),
        pair_batch_size=int(args.pair_batch_size),
        pair_chunk_size=int(args.pair_chunk_size),
        pair_num_workers=int(args.pair_num_workers),
        pair_threshold=float(args.pair_threshold),
        pair_device=str(args.pair_device) if args.pair_device else None,
        pair_max_dna=int(args.pair_max_dna),
        pair_max_rna=int(args.pair_max_rna),
        save_artifacts=not bool(args.no_save_artifacts),
    )

    result = run_app_job(dna_seqs=dna, rna_seqs=rna, cfg=cfg)
    summary = {
        "mode": result.mode,
        "n_dna_used": int(result.used_dna_count),
        "n_rna_used": int(result.used_rna_count),
        "output_dir": str(cfg.output_dir) if cfg.output_dir else None,
        "dna_path": str(result.used_dna_path),
        "rna_path": str(result.used_rna_path),
        "dna_scores_path": str(result.dna_scores_path)
        if result.dna_scores_path
        else None,
        "rna_scores_path": str(result.rna_scores_path)
        if result.rna_scores_path
        else None,
        "pair_predictions_path": str(result.pair_predictions_path)
        if result.pair_predictions_path
        else None,
        "pair_summary_path": str(result.pair_summary_path)
        if result.pair_summary_path
        else None,
        "pair_meta_path": str(result.pair_meta_path)
        if result.pair_meta_path
        else None,
    }

    if cfg.output_dir:
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    LOGGER.warning(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
