"""
Shared utilities for embedding-based binary classification inference.

Used by DNABERT-2 and RNA-FM inference scripts to avoid duplicate logic.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from dnarna.models.shared.embed import load_embeddings
from dnarna.models.shared.predict.train import LogisticMLP

LOGGER = logging.getLogger("predict_utils")
PARQUET_EXTENSIONS = {".parquet", ".parq", ".pq"}
CSV_EXTENSIONS = {".csv", ".tsv", ".csv.gz", ".tsv.gz"}


@dataclass
class EmbeddingInferConfig:
    embeddings_npz: str
    checkpoint: str
    output: str
    batch_size: int
    device: str | None
    threshold: float
    progress: bool
    verbose: bool
    plot_path: str | None = None
    plot_bins: int = 50


def infer_extension(path: Path) -> str:
    lower = path.name.lower()
    for ext in PARQUET_EXTENSIONS | CSV_EXTENSIONS | {".json", ".jsonl"}:
        if lower.endswith(ext):
            return ext
    return path.suffix.lower()


def normalize_embeddings(
    embeddings: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected embeddings to be 2D [N, D], got shape {embeddings.shape}"
        )
    if embeddings.shape[1] != mean.shape[0]:
        raise ValueError(
            f"Embedding dim mismatch: embeddings have {embeddings.shape[1]}, "
            f"but checkpoint stats are {mean.shape[0]}"
        )
    safe_std = np.where(np.abs(std) < 1e-6, 1.0, std)
    return (embeddings - mean) / safe_std


def predict_probabilities(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    show_progress: bool,
) -> np.ndarray:
    model.eval()
    probs: list[np.ndarray] = []
    iterator: Iterable = (
        tqdm(loader, desc="inference", leave=False) if show_progress else loader
    )
    with torch.no_grad():
        for (xb,) in iterator:
            xb = xb.to(device)
            logits = model(xb)
            batch_probs = torch.sigmoid(logits).cpu().numpy()
            probs.append(batch_probs)
    if show_progress and hasattr(iterator, "close"):
        iterator.close()
    if not probs:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(probs, axis=0)


def save_predictions(
    ids: list[str],
    probs: np.ndarray,
    output: Path,
    *,
    threshold: float,
    id_column: str = "id",
) -> None:
    if len(ids) != len(probs):
        raise ValueError(
            f"IDs ({len(ids)}) and probabilities ({len(probs)}) length mismatch"
        )
    preds = (probs >= threshold).astype(int)
    df = pd.DataFrame(
        {
            id_column: ids,
            "prob": probs.astype(np.float32),
            "pred": preds,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    ext = infer_extension(output)
    if ext in PARQUET_EXTENSIONS:
        df.to_parquet(output, index=False)
    elif ext in {".json", ".jsonl"}:
        df.to_json(output, orient="records", lines=ext == ".jsonl")
    elif ext in CSV_EXTENSIONS or not ext:
        sep = "\t" if ext.startswith(".tsv") else ","
        df.to_csv(output, index=False, sep=sep)
    else:
        raise ValueError(
            f"Unsupported output extension '{ext}'. "
            f"Supported: {sorted(PARQUET_EXTENSIONS | CSV_EXTENSIONS | {'.json', '.jsonl'})}"
        )
    LOGGER.info(
        "Wrote predictions to %s (pos=%d neg=%d threshold=%.3f)",
        output,
        int((preds == 1).sum()),
        int((preds == 0).sum()),
        threshold,
    )


def load_classifier_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device)
    required_keys = {"model_state_dict", "feature_mean", "feature_std", "input_dim"}
    missing = required_keys - set(ckpt.keys())
    if missing:
        raise KeyError(f"Checkpoint missing keys: {sorted(missing)}")
    return ckpt


def run_embedding_classifier_inference(
    cfg: EmbeddingInferConfig, *, logger: logging.Logger | None = None
) -> None:
    log = logger or LOGGER
    log.info("[stage] loading embeddings from %s", cfg.embeddings_npz)
    embeddings, ids = load_embeddings(Path(cfg.embeddings_npz).expanduser())
    log.info(
        "[ok] embeddings loaded: shape=%s ids=%d", tuple(embeddings.shape), len(ids)
    )

    device = torch.device(
        cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    log.info("[stage] loading checkpoint %s on %s", cfg.checkpoint, device.type)
    ckpt = load_classifier_checkpoint(Path(cfg.checkpoint).expanduser(), device)

    input_dim = int(ckpt["input_dim"])
    training_cfg = ckpt.get("config", {}) or {}
    hidden_dims_raw = training_cfg.get("hidden_dims")
    if not hidden_dims_raw:
        raise KeyError(
            "Checkpoint config missing required 'hidden_dims' "
            "(linear/legacy models not supported)."
        )
    hidden_dims = [int(d) for d in hidden_dims_raw if int(d) > 0]
    if not hidden_dims:
        raise ValueError(
            "Checkpoint config 'hidden_dims' must contain positive integers."
        )
    model = LogisticMLP(input_dim, hidden_dims).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.info("[ok] model restored: input_dim=%d hidden_dims=%s", input_dim, hidden_dims)

    mean = ckpt["feature_mean"].cpu().numpy().astype(np.float32)
    std = ckpt["feature_std"].cpu().numpy().astype(np.float32)
    X = normalize_embeddings(embeddings.astype(np.float32), mean, std)

    ds = TensorDataset(torch.from_numpy(X).float())
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    probs = predict_probabilities(model, loader, device, show_progress=cfg.progress)

    output_path = Path(cfg.output).expanduser()
    save_predictions(ids, probs, output_path, threshold=cfg.threshold)

    plot_target = (
        Path(cfg.plot_path).expanduser()
        if cfg.plot_path
        else output_path.with_name(f"{output_path.stem}.score_distribution.pdf")
    )
    from dnarna.plot.score_distribution import plot_score_distribution

    plot_path = plot_score_distribution(
        probs,
        output_path=plot_target,
        threshold=cfg.threshold,
        bins=cfg.plot_bins,
        xlim=(0.0, 1.0),
    )
    if plot_path is not None:
        log.info("Score distribution plot saved to %s", plot_path)

    meta = {
        "n_sequences": int(len(ids)),
        "threshold": float(cfg.threshold),
        "checkpoint": str(Path(cfg.checkpoint).expanduser()),
        "device": str(device),
        "mean_probability": float(np.mean(probs)) if len(probs) else None,
        "score_distribution_plot": str(plot_path) if plot_path is not None else None,
        "score_distribution_bins": int(cfg.plot_bins),
    }
    meta_path = (
        Path(cfg.output)
        .expanduser()
        .with_suffix(Path(cfg.output).suffix + ".meta.json")
    )
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log.info("Metadata written to %s", meta_path)
