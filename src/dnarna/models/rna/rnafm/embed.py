"""
Encode RNA sequences from a Parquet or CSV file with the official RNA-FM model.

Input file requirements:
    Must contain columns ``id`` and ``seq`` (other columns are ignored).

Example:
    python -m dnarna.models.rna.rnafm.embed \
        --input_file data/sample_rna.parquet \
        --output_dir outputs/rnafm_embeddings \
        --batch_size 64 \
        --device cuda:0

Outputs:
    <output_dir>/<input_stem>.embeddings.npz
        - ids: str array of retained sequence IDs (length N; skipped IDs listed separately)
        - embeddings: float16/32 array shaped [N, D] with RNA-FM embeddings. D: embedding dimension for the selected RNA-FM variant.
    <output_dir>/<input_stem>.embeddings.npz.meta.json
        - JSON metadata containing model/device/dtype info and skipped IDs
    <output_dir>/<input_stem>.embeddings.npz.skipped.txt (only if any were skipped)
        - Plain-text list of sequence IDs exceeding the model length limit
    <output_dir>/<input_stem>.embeddings.npz.log
        - Run log
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from dnarna.models.rna.rnafm.encoder import RNAFMEncoder
from dnarna.models.shared.embed import (
    load_input_dataframe,
    save_embeddings_npz,
    write_meta_json,
    write_skipped_ids,
)

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    LOGGER.addHandler(logging.NullHandler())


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


def _standardize_sequence(seq: str, replace_t_with_u: bool = True) -> str:
    seq = str(seq).upper()
    if replace_t_with_u:
        seq = seq.replace("T", "U")
    cleaned = "".join(c for c in seq if c in "AUCGN-")
    return cleaned or "A"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode RNA sequences with RNA-FM and save embeddings."
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="Input Parquet/CSV file with identifier + sequence columns.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write outputs (.npz/.meta.json/.skipped.txt/.log).",
    )
    parser.add_argument(
        "--id_col", default="id", help="Identifier column name (default: id)."
    )
    parser.add_argument(
        "--seq_col", default="seq", help="Sequence column name (default: seq)."
    )
    parser.add_argument(
        "--variant",
        choices=["rna", "mrna"],
        default="rna",
        help="RNA-FM checkpoint variant.",
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
        default=0,
        help="Optional max sequence length (<= model limit). 0 uses model default.",
    )
    parser.add_argument("--device", default=None, help="Torch device identifier.")
    parser.add_argument(
        "--no_t_to_u",
        action="store_true",
        help="Disable automatic T→U substitution during cleaning.",
    )
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
            f"Input file must contain columns '{args.id_col}' and '{args.seq_col}'. "
            f"Available columns: {df.columns.tolist()}"
        )

    encoder = RNAFMEncoder(
        variant=args.variant,
        batch_size=args.batch_size,
        max_length=args.max_length,
        dtype=(np.float32 if args.dtype == "fp32" else np.float16),
        device=args.device,
    )
    max_len = getattr(encoder, "_effective_max_len", None) or getattr(
        encoder, "_model_max_seq_len", None
    )

    ids: list[str] = []
    seqs: list[str] = []
    skipped: list[str] = []

    for rid, seq in zip(df[args.id_col].astype(str), df[args.seq_col]):
        standardized = _standardize_sequence(seq, replace_t_with_u=not args.no_t_to_u)
        if max_len and len(standardized) > max_len:
            skipped.append(rid)
            continue
        ids.append(rid)
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
        desc=f"RNA-FM ({args.variant})",
    )
    embeddings = embeddings.astype(np.float16 if args.dtype == "fp16" else np.float32)

    save_embeddings_npz(output_npz, ids, embeddings)
    LOGGER.info("Saved embeddings with shape %s to %s", embeddings.shape, output_npz)

    meta = {
        "model": "rnafm",
        "variant": args.variant,
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
