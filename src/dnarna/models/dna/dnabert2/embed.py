"""
Encode DNA sequences from a Parquet or CSV file with a DNABERT-2 backbone.

Input file requirements:
    Must contain columns ``id`` and ``seq`` (other columns are ignored).
以前的版本使用 ``dna_id`` / ``dna_seq``，现在仅保留通用列名以减少维护成本。

Example:
    python -m dnarna.models.dnabert2.embed \
        --input_file data/sample_dna.parquet \
        --output_dir outputs/dnabert2_embeddings \
        --batch_size 64 \
        --device cuda:0

Outputs:
    <output_dir>/<input_stem>.embeddings.npz
        - ids: str array of retained sequence IDs (length N; skipped IDs listed separately)
        - embeddings: float16/32 array shaped [N, D] with DNABERT-2 embeddings
    <output_dir>/<input_stem>.embeddings.npz.meta.json
        - JSON metadata containing model/backbone/device/dtype info and skipped IDs
    <output_dir>/<input_stem>.embeddings.npz.skipped.txt (only if any were skipped)
        - Plain-text list of sequence IDs exceeding the tokenizer length limit
    <output_dir>/<input_stem>.embeddings.npz.log
        - Run log
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from dnarna.models.dna.dnabert2.encoder import DNABERT2Encoder
from dnarna.models.shared.embed import (
    load_input_dataframe,
    save_embeddings_npz,
    write_meta_json,
    write_skipped_ids,
)

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    LOGGER.addHandler(logging.NullHandler())
DNA_ALPHABET = set("ACGTN")


def _setup_logging(*, log_path: Path, verbose: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s")

    stream_level = logging.INFO if verbose else logging.WARNING
    if not any(isinstance(handler, logging.StreamHandler) for handler in root.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(stream_level)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)
    else:
        for handler in root.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(stream_level)

    if not any(
        isinstance(handler, logging.FileHandler)
        and Path(getattr(handler, "baseFilename", "")) == log_path
        for handler in root.handlers
    ):
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def _standardize_sequence(seq: str) -> str:
    seq = str(seq).upper()
    cleaned = "".join(c for c in seq if c in DNA_ALPHABET)
    return cleaned or "A"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode DNA sequences with DNABERT-2 and save embeddings."
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help=(
            "Input Parquet/CSV file with identifier + sequence columns. "
            "Defaults expect dna_id + dna_seq; id + seq are also accepted."
        ),
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write outputs (.npz/.meta.json/.skipped.txt/.log).",
    )
    # 固定使用 id/seq，仍保留参数以防未来扩展，但默认即为 id/seq
    parser.add_argument(
        "--id_col", default="id", help="Identifier column name (default: id)."
    )
    parser.add_argument(
        "--seq_col", default="seq", help="Sequence column name (default: seq)."
    )
    parser.add_argument(
        "--backbone",
        default="zhihan1996/DNABERT-2-117M",
        help="DNABERT-2 checkpoint identifier.",
    )
    parser.add_argument(
        "--dtype",
        choices=["fp32", "fp16"],
        default="fp16",
        help="Output tensor dtype.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for encoding."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Tokenizer max_length (also used to filter over-long sequences).",
    )
    parser.add_argument("--device", default=None, help="Torch device identifier.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable info-level logging (default warns only).",
    )
    return parser.parse_args()


def _resolve_output_npz(*, input_path: Path, output_dir: str | Path) -> Path:
    output_dir_path = Path(output_dir).expanduser()
    filename = f"{input_path.stem}.embeddings.npz"
    return output_dir_path / filename


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input_file).expanduser()
    output_npz = _resolve_output_npz(input_path=input_path, output_dir=args.output_dir)
    log_path = output_npz.with_suffix(output_npz.suffix + ".log")
    _setup_logging(log_path=log_path, verbose=args.verbose)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = load_input_dataframe(input_path)
    if args.id_col not in df.columns or args.seq_col not in df.columns:
        raise ValueError(
            (
                "Input file must contain columns '{id_col}' and '{seq_col}'. Available columns: {available}"
            ).format(
                id_col=args.id_col, seq_col=args.seq_col, available=df.columns.tolist()
            )
        )

    encoder = DNABERT2Encoder(
        backbone=args.backbone,
        batch_size=args.batch_size,
        max_length=args.max_length,
        dtype=(np.float32 if args.dtype == "fp32" else np.float16),
        device=args.device,
    )
    max_len = getattr(encoder, "_effective_max_len", None) or encoder.max_length

    ids: list[str] = []
    seqs: list[str] = []
    skipped: list[str] = []

    for did, seq in zip(df[args.id_col].astype(str), df[args.seq_col]):
        standardized = _standardize_sequence(seq)
        if max_len and len(standardized) > max_len:
            skipped.append(did)
            continue
        ids.append(did)
        seqs.append(standardized)

    if not ids:
        raise RuntimeError("No sequences available for encoding after filtering.")

    LOGGER.info(
        "Loaded %d sequences (skipped %d over-length). Encoding with batch_size=%d on %s.",
        len(ids),
        len(skipped),
        encoder.batch_size,
        encoder.device,
    )

    embeddings = encoder.encode_many(
        seqs,
        l2norm=False,
        show_progress=True,
        desc="DNABERT-2",
    )
    embeddings = embeddings.astype(np.float16 if args.dtype == "fp16" else np.float32)

    save_embeddings_npz(output_npz, ids, embeddings)
    LOGGER.info("Saved embeddings with shape %s to %s", embeddings.shape, output_npz)

    meta = {
        "model": "dnabert2",
        "backbone": args.backbone,
        "dtype": args.dtype,
        "max_length": int(args.max_length),
        "n_sequences": int(len(ids)),
        "embedding_dim": int(embeddings.shape[1]),
        "device": str(encoder.device),
        "skipped_ids": skipped,
    }
    meta_path = write_meta_json(output_npz, meta)
    LOGGER.info("Metadata written to %s", meta_path)

    skipped_path = write_skipped_ids(output_npz, skipped)
    if skipped_path:
        LOGGER.info("Skipped sequence IDs written to %s", skipped_path)


if __name__ == "__main__":
    main()
