from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("shared_embed")


def save_embeddings_npz(
    output_npz: Path, ids: list[str], embeddings: np.ndarray
) -> None:
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        ids=np.array(ids, dtype=str),
        embeddings=embeddings,
    )


def write_meta_json(output_npz: Path, meta: dict) -> Path:
    import json

    meta_path = output_npz.with_suffix(output_npz.suffix + ".meta.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta_path


def write_skipped_ids(output_npz: Path, skipped_ids: list[str]) -> Path | None:
    if not skipped_ids:
        return None
    skipped_path = output_npz.with_suffix(output_npz.suffix + ".skipped.txt")
    with skipped_path.open("w", encoding="utf-8") as f:
        for seq_id in skipped_ids:
            f.write(f"{seq_id}\n")
    return skipped_path


PARQUET_EXTENSIONS = {".parquet", ".parq", ".pq"}
CSV_EXTENSIONS = {".csv", ".tsv"}


def load_input_dataframe(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix in PARQUET_EXTENSIONS:
        return pd.read_parquet(file_path)
    if suffix in CSV_EXTENSIONS:
        sep = "," if suffix == ".csv" else "\t"
        return pd.read_csv(file_path, sep=sep)
    supported = ", ".join(sorted(PARQUET_EXTENSIONS | CSV_EXTENSIONS))
    raise ValueError(
        f"Unsupported input file extension '{suffix}'. "
        f"Supported extensions: {supported}"
    )


def load_embeddings(
    path: Path, *, require_ids: bool = True
) -> tuple[np.ndarray, list[str]]:
    """Load embeddings from an .npz with standard keys: ``embeddings`` + ``ids``."""
    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")
    data = np.load(path, allow_pickle=False)
    if "embeddings" not in data:
        keys = list(data.keys())
        raise KeyError(
            f"{path} missing required key 'embeddings'. Keys present: {keys}. "
            "If this file was produced by an older embed script, regenerate embeddings "
            "or load via load_embeddings_any(...) to accept dna/rna-prefixed keys."
        )
    emb = data["embeddings"]
    if "ids" not in data:
        if require_ids:
            keys = list(data.keys())
            raise KeyError(
                f"{path} missing required key 'ids'. Keys present: {keys}. "
                "Regenerate embeddings to include ids, or call load_embeddings(..., require_ids=False) "
                "to generate positional IDs."
            )
        LOGGER.warning("Sequence IDs not found in %s; generating positional IDs.", path)
        ids = [f"idx_{i}" for i in range(len(emb))]
        return emb, ids
    ids = data["ids"].astype(str).tolist()
    return emb, ids
