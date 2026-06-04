"""
Shared utilities for training shallow binary classifiers on embeddings.

Used by DNABERT-2 / RNA-FM training scripts to avoid duplicate logic.
"""

from __future__ import annotations

import csv
import logging
import random
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

__all__ = [
    "LogisticMLP",
    "setup_logging",
    "set_seed",
    "standardize",
    "build_loaders",
    "run_epoch",
    "collect_predictions",
    "export_val_predictions",
    "infer_metadata_extension",
    "load_metadata",
    "prepare_datasets",
    "train_binary_classifier",
    "DEFAULT_SAVE_BEST",
    "DEFAULT_MONITOR",
    "DEFAULT_EARLY_STOP_PATIENCE",
    "DEFAULT_EPOCHS",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_LR",
    "DEFAULT_WEIGHT_DECAY",
    "DEFAULT_VAL_FRACTION",
    "DEFAULT_SEED",
    "DEFAULT_NUM_WORKERS",
    "DEFAULT_HEARTBEAT_SECONDS",
    "DEFAULT_ID_COL",
    "DEFAULT_LABEL_COL",
    "DEFAULT_SPLIT_COL",
    "DEFAULT_TRAIN_SPLITS",
    "DEFAULT_VAL_SPLITS",
]

LOGGER = logging.getLogger("train_utils")
PARQUET_EXTENSIONS = {".parquet", ".parq", ".pq"}
CSV_EXTENSIONS = {".csv", ".tsv", ".csv.gz", ".tsv.gz"}
_QUIET_WITH_TQDM_ATTR = "quiet_with_tqdm"

DEFAULT_SAVE_BEST = True
DEFAULT_MONITOR = "val_loss"
DEFAULT_EARLY_STOP_PATIENCE = 0

# CLI defaults shared across model-specific train entrypoints.
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_VAL_FRACTION = 0.1
DEFAULT_SEED = 42
DEFAULT_NUM_WORKERS = 0
DEFAULT_HEARTBEAT_SECONDS = 60
DEFAULT_ID_COL = "id"
DEFAULT_LABEL_COL = "label"
DEFAULT_SPLIT_COL = "split"
DEFAULT_TRAIN_SPLITS = "train"
DEFAULT_VAL_SPLITS = "val"


class _QuietTqdmFilter(logging.Filter):
    def __init__(self, *, enabled: bool) -> None:
        super().__init__()
        self.enabled = enabled

    def filter(self, record: logging.LogRecord) -> bool:
        if not self.enabled:
            return True
        return not bool(getattr(record, _QUIET_WITH_TQDM_ATTR, False))


def _build_console_handler(*, quiet_tqdm: bool) -> logging.Handler:
    handler: logging.Handler = logging.StreamHandler()
    handler.addFilter(_QuietTqdmFilter(enabled=quiet_tqdm))
    return handler


def setup_logging(
    log_file: Path | None = None,
    *,
    level: int = logging.INFO,
    quiet_tqdm: bool = False,
) -> None:
    """Configure logging for training scripts (console + optional file)."""
    handlers: list[logging.Handler] = [_build_console_handler(quiet_tqdm=quiet_tqdm)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def _emit_status(
    message: str,
    *,
    logger: logging.Logger | None = None,
    level: int = logging.INFO,
    quiet_with_tqdm: bool = False,
) -> None:
    """Emit a status line via logging if enabled, otherwise print with flush."""
    if logger is not None and logger.isEnabledFor(level):
        logger.log(level, message, extra={_QUIET_WITH_TQDM_ATTR: quiet_with_tqdm})
    else:
        print(message, flush=True)


class LogisticMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int] | None = None) -> None:
        super().__init__()
        hidden_dims = [int(d) for d in (hidden_dims or []) if int(d) > 0]
        if not hidden_dims:
            raise ValueError(
                "hidden_dims must be non-empty (linear model is not supported)."
            )

        layers: list[nn.Module] = []
        prev_dim = in_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def standardize(
    x_train: np.ndarray, x_eval: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    x_train_norm = (x_train - mean) / std
    x_eval_norm = None
    if x_eval is not None:
        x_eval_norm = (x_eval - mean) / std
    return x_train_norm, x_eval_norm, mean, std


def build_loaders(
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    num_workers: int,
    val_fraction: float,
    val_data: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[DataLoader, DataLoader | None, np.ndarray, np.ndarray]:
    N = X.shape[0]
    indices = np.random.permutation(N)
    X = X[indices]
    y = y[indices]

    if val_data is not None:
        X_train, y_train = X, y
        X_val, y_val = val_data
    else:
        if val_fraction > 0 and N > 1:
            tentative = int(N * val_fraction)
            n_val = max(1, tentative)
            if n_val >= N:
                n_val = N - 1
        else:
            n_val = 0

        if n_val > 0:
            X_val = X[:n_val]
            y_val = y[:n_val]
            X_train = X[n_val:]
            y_train = y[n_val:]
        else:
            X_train, y_train = X, y
            X_val = np.empty((0, X.shape[1]), dtype=X.dtype)
            y_val = np.empty((0,), dtype=y.dtype)

    X_train, X_val_norm, mean, std = standardize(
        X_train, X_val if val_data is not None or (X_val.size > 0) else None
    )
    has_val = (val_data is not None) or (X_val_norm is not None and X_val.size > 0)
    if has_val and X_val_norm is not None:
        X_val = X_val_norm

    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )

    if has_val:
        val_ds = TensorDataset(
            torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
    else:
        val_loader = None

    return train_loader, val_loader, mean, std


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    desc: str,
    show_progress: bool,
    logger: logging.Logger | None = None,
    heartbeat_seconds: float | None = None,
    heartbeat_batches: int | None = None,
) -> dict[str, float | int]:
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    is_train = optimizer is not None
    model.train(is_train)
    progress_bar = tqdm(loader, desc=desc, leave=False) if show_progress else None
    iterator: Iterable = progress_bar if progress_bar is not None else loader

    n_batches: int | None
    try:
        n_batches = len(loader)
    except TypeError:
        n_batches = None

    start_time = time.monotonic()
    last_heartbeat_time = start_time
    last_heartbeat_batch = 0
    if heartbeat_batches is not None and heartbeat_batches <= 0:
        heartbeat_batches = None
    if heartbeat_seconds is not None and heartbeat_seconds <= 0:
        heartbeat_seconds = None

    _emit_status(
        f"[stage] {desc}: begin ({'train' if is_train else 'eval'})"
        + (f" batches={n_batches}" if n_batches is not None else ""),
        logger=logger,
        quiet_with_tqdm=True,
    )

    for batch_idx, (xb, yb) in enumerate(iterator, start=1):
        xb = xb.to(device)
        yb = yb.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        if is_train:
            loss.backward()
            optimizer.step()
        probs = torch.sigmoid(logits)
        preds = probs >= 0.5
        labels = yb >= 0.5

        total_correct += (preds == labels).sum().item()
        total_loss += loss.item() * xb.size(0)
        total_examples += xb.size(0)

        tp += (preds & labels).sum().item()
        tn += ((~preds) & (~labels)).sum().item()
        fp += (preds & (~labels)).sum().item()
        fn += ((~preds) & labels).sum().item()

        if heartbeat_seconds is not None or heartbeat_batches is not None:
            now = time.monotonic()
            due_by_time = (
                heartbeat_seconds is not None
                and (now - last_heartbeat_time) >= heartbeat_seconds
            )
            due_by_batch = (
                heartbeat_batches is not None
                and (batch_idx - last_heartbeat_batch) >= heartbeat_batches
            )
            if due_by_time or due_by_batch:
                avg_loss_so_far = total_loss / max(1, total_examples)
                acc_so_far = total_correct / max(1, total_examples)
                elapsed = max(1e-9, now - start_time)
                throughput = total_examples / elapsed
                batch_total = (
                    f"{batch_idx}/{n_batches}"
                    if n_batches is not None
                    else f"{batch_idx}"
                )
                _emit_status(
                    f"[heartbeat] {desc}: batch={batch_total} "
                    f"examples={total_examples} loss={avg_loss_so_far:.4f} "
                    f"acc={acc_so_far:.4f} ex/s={throughput:.1f}",
                    logger=logger,
                    quiet_with_tqdm=True,
                )
                last_heartbeat_time = now
                last_heartbeat_batch = batch_idx
    if progress_bar is not None:
        progress_bar.close()
    avg_loss = total_loss / max(1, total_examples)
    acc = total_correct / max(1, total_examples)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    elapsed = max(1e-9, time.monotonic() - start_time)
    throughput = total_examples / elapsed
    _emit_status(
        f"[stage] {desc}: end elapsed={elapsed:.1f}s "
        f"examples={total_examples} ex/s={throughput:.1f} "
        f"loss={avg_loss:.4f} acc={acc:.4f} f1={f1:.4f}",
        logger=logger,
        quiet_with_tqdm=True,
    )
    return {
        "loss": float(avg_loss),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def collect_predictions(
    model: nn.Module,
    loader: DataLoader | None,
    device: torch.device,
    *,
    show_progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect probabilities and labels for a given loader (used for ROC/PR plotting)."""
    if loader is None:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)
    model.eval()
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    iterator: Iterable = (
        tqdm(loader, desc="val predict", leave=False) if show_progress else loader
    )
    with torch.no_grad():
        for xb, yb in iterator:
            xb = xb.to(device)
            logits = model(xb)
            batch_probs = torch.sigmoid(logits).cpu().numpy()
            probs.append(batch_probs)
            labels.append(yb.cpu().numpy())
    if show_progress and hasattr(iterator, "close"):
        iterator.close()
    if not probs:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.concatenate(probs, axis=0), np.concatenate(labels, axis=0)


def export_val_predictions(
    model: nn.Module,
    val_loader: DataLoader | None,
    val_ids: list[str],
    device: torch.device,
    output_dir: Path,
    *,
    filename: str = "val_predictions.csv",
    threshold: float = 0.5,
    show_progress: bool = False,
) -> tuple[np.ndarray, np.ndarray, Path | None]:
    """
    Run validation inference, write a CSV with id/label/prob/pred, and return arrays.
    """
    probs, labels = collect_predictions(
        model, val_loader, device, show_progress=show_progress
    )
    if probs.size == 0 or labels.size == 0:
        return probs, labels, None

    if val_ids and len(val_ids) != len(probs):
        LOGGER.warning(
            "val_ids length (%d) does not match validation predictions (%d); writing without IDs.",
            len(val_ids),
            len(probs),
        )
        ids_for_write = [f"val_idx_{i}" for i in range(len(probs))]
    elif val_ids:
        ids_for_write = val_ids
    else:
        ids_for_write = [f"val_idx_{i}" for i in range(len(probs))]

    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / filename
    with pred_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "prob", "pred"])
        writer.writeheader()
        for seq_id, label, prob in zip(ids_for_write, labels, probs):
            writer.writerow(
                {
                    "id": seq_id,
                    "label": int(label),
                    "prob": float(prob),
                    "pred": int(prob >= threshold),
                }
            )
    LOGGER.info("Validation predictions written to %s", pred_path)
    return probs, labels, pred_path


def infer_metadata_extension(path: Path) -> str:
    lower_name = path.name.lower()
    for ext in PARQUET_EXTENSIONS | CSV_EXTENSIONS:
        if lower_name.endswith(ext):
            return ext
    return path.suffix.lower()


def load_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    ext = infer_metadata_extension(path)
    if ext in PARQUET_EXTENSIONS:
        return pd.read_parquet(path)
    if ext in CSV_EXTENSIONS:
        sep = "\t" if ext.startswith(".tsv") else ","
        return pd.read_csv(path, sep=sep)
    supported = ", ".join(sorted(PARQUET_EXTENSIONS | CSV_EXTENSIONS))
    raise ValueError(
        f"Unsupported metadata extension '{ext}' for {path}. Supported: {supported}"
    )


def prepare_datasets(
    embeddings: np.ndarray,
    ids: list[str],
    metadata: pd.DataFrame,
    *,
    id_col: str,
    label_col: str,
    split_col: str | None,
    train_splits: list[str],
    val_splits: list[str],
    logger: logging.Logger | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[str],
    np.ndarray | None,
    np.ndarray | None,
    list[str],
    list[str],
    np.ndarray,
]:
    log = logger or LOGGER
    for col in (id_col, label_col):
        if col not in metadata.columns:
            raise ValueError(
                f"Metadata file missing required column '{col}'. "
                f"Available columns: {metadata.columns.tolist()}"
            )
    split_col_local = split_col
    if split_col_local and split_col_local not in metadata.columns:
        log.warning(
            "Split column '%s' not found in metadata; falling back to random split.",
            split_col_local,
        )
        split_col_local = None

    meta = metadata.copy()
    meta[id_col] = meta[id_col].astype(str)
    id_to_idx = {seq_id: idx for idx, seq_id in enumerate(ids)}
    meta["_emb_idx"] = meta[id_col].map(id_to_idx)
    missing = meta["_emb_idx"].isna().sum()
    if missing:
        log.warning(
            "Dropped %d metadata rows without matching embeddings.", int(missing)
        )
    meta = meta.dropna(subset=["_emb_idx"])
    if meta.empty:
        raise ValueError("No overlapping IDs between embeddings and metadata.")

    meta["_emb_idx"] = meta["_emb_idx"].astype(int)
    label_values = pd.to_numeric(meta[label_col], errors="coerce")
    if label_values.isna().any():
        raise ValueError("Metadata label column contains non-numeric values.")
    unique_labels = set(label_values.unique().tolist())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            f"Label column must be binary (0/1). Observed values: {sorted(unique_labels)}"
        )
    labels = label_values.astype(np.float32).to_numpy()
    emb_idx = meta["_emb_idx"].to_numpy(dtype=int)
    X_all = embeddings[emb_idx].astype(np.float32)
    matched_ids = meta[id_col].astype(str).tolist()

    splits = (
        meta[split_col_local].astype(str).to_numpy()
        if split_col_local and split_col_local in meta.columns
        else None
    )

    val_mask = (
        np.isin(splits, val_splits)
        if splits is not None and val_splits
        else np.zeros(len(meta), dtype=bool)
    )
    if splits is not None:
        if train_splits:
            train_mask = np.isin(splits, train_splits)
        else:
            train_mask = ~val_mask
    else:
        train_mask = np.ones(len(meta), dtype=bool)
    train_mask = train_mask & ~val_mask
    usable_mask = train_mask | val_mask

    if not usable_mask.any():
        raise ValueError("No samples left after applying split filters.")

    X_all = X_all[usable_mask]
    labels = labels[usable_mask]
    matched_ids = [seq_id for seq_id, keep in zip(matched_ids, usable_mask) if keep]
    if splits is not None:
        splits = splits[usable_mask]
        train_mask = train_mask[usable_mask]
        val_mask = val_mask[usable_mask]
    else:
        train_mask = train_mask[usable_mask]
        val_mask = val_mask[usable_mask]

    X_train = X_all[train_mask]
    y_train = labels[train_mask]
    train_ids = [seq_id for seq_id, keep in zip(matched_ids, train_mask) if keep]
    if not len(X_train):
        raise ValueError("No training samples available after split filtering.")

    if val_mask.any():
        X_val = X_all[val_mask]
        y_val = labels[val_mask]
        val_ids = [seq_id for seq_id, keep in zip(matched_ids, val_mask) if keep]
    else:
        X_val = None
        y_val = None
        val_ids = []

    overall_labels = labels.copy()

    return (
        X_train,
        y_train,
        train_ids,
        X_val,
        y_val,
        val_ids,
        matched_ids,
        overall_labels,
    )


def train_binary_classifier(
    *,
    embeddings_npz: Path,
    load_embeddings: Callable[[Path], tuple[np.ndarray, list[str]]],
    metadata_file: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    hidden_dims: list[int],
    lr: float,
    weight_decay: float,
    val_fraction: float,
    seed: int,
    device: str | None,
    num_workers: int,
    progress: bool,
    heartbeat_seconds: float,
    id_col: str,
    label_col: str,
    split_col: str | None,
    train_splits: list[str],
    val_splits: list[str],
    save_best: bool = DEFAULT_SAVE_BEST,
    monitor: str = DEFAULT_MONITOR,
    early_stop_patience: int = DEFAULT_EARLY_STOP_PATIENCE,
    config: dict[str, object] | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """
    Shared training pipeline for shallow binary classifiers on embeddings.

    The embeddings loader is injected to avoid circular imports with infer utilities.
    """
    from dnarna.plot.binary_curves import plot_binary_curves
    from dnarna.plot.training_history import plot_training_history

    log = logger or LOGGER

    t_start = time.monotonic()
    _emit_status("[stage] start training module", logger=log)
    set_seed(seed)
    if not hidden_dims:
        raise ValueError(
            "hidden_dims must be non-empty (linear model is not supported)."
        )

    embeddings_npz = embeddings_npz.expanduser()
    metadata_file = metadata_file.expanduser()
    output_dir = output_dir.expanduser()

    _emit_status(f"[stage] loading embeddings from {embeddings_npz}", logger=log)
    t0 = time.monotonic()
    embeddings, emb_ids = load_embeddings(embeddings_npz)
    _emit_status(
        f"[ok] embeddings loaded: shape={tuple(embeddings.shape)} n_ids={len(emb_ids)} "
        f"elapsed={time.monotonic() - t0:.1f}s",
        logger=log,
    )

    _emit_status(f"[stage] loading metadata from {metadata_file}", logger=log)
    t0 = time.monotonic()
    metadata = load_metadata(metadata_file)
    _emit_status(
        f"[ok] metadata loaded: rows={len(metadata)} cols={len(metadata.columns)} "
        f"columns={metadata.columns.tolist()} "
        f"elapsed={time.monotonic() - t0:.1f}s",
        logger=log,
    )

    (
        X_train_raw,
        y_train_raw,
        _train_ids,
        X_val_raw,
        y_val_raw,
        val_ids,
        all_ids,
        label_vector,
    ) = prepare_datasets(
        embeddings,
        emb_ids,
        metadata,
        id_col=id_col,
        label_col=label_col,
        split_col=split_col,
        train_splits=train_splits,
        val_splits=val_splits,
        logger=log,
    )
    _emit_status(
        "[ok] dataset prepared: "
        f"train=({X_train_raw.shape[0]},{X_train_raw.shape[1]}) "
        + (
            f"val=({X_val_raw.shape[0]},{X_val_raw.shape[1]}) "
            if X_val_raw is not None
            else "val=(0,0) "
        )
        + f"total={len(label_vector)} "
        + f"pos={int((label_vector >= 0.5).sum())} "
        + f"neg={int((label_vector < 0.5).sum())}",
        logger=log,
    )

    if X_train_raw.ndim != 2:
        raise ValueError(
            f"Expected embeddings to be 2D [N, D], got shape {X_train_raw.shape}"
        )
    n_train = int(len(y_train_raw))
    n_val = int(len(y_val_raw)) if y_val_raw is not None else 0
    _emit_status(
        f"[stage] prepared samples: train={n_train} val={n_val} total={len(label_vector)}",
        logger=log,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _emit_status(f"[stage] output directory prepared: {output_dir}", logger=log)

    val_tuple = (
        (X_val_raw, y_val_raw)
        if X_val_raw is not None and y_val_raw is not None
        else None
    )
    _emit_status(
        f"[stage] building dataloaders (batch_size={batch_size} num_workers={num_workers})",
        logger=log,
    )
    t0 = time.monotonic()
    train_loader, val_loader, mean, std = build_loaders(
        X_train_raw,
        y_train_raw,
        batch_size=batch_size,
        num_workers=num_workers,
        val_fraction=val_fraction,
        val_data=val_tuple,
    )
    _emit_status(
        "[ok] dataloaders built: "
        f"train_batches={len(train_loader)} "
        + (
            f"val_batches={len(val_loader)} "
            if val_loader is not None
            else "val_batches=0 "
        )
        + f"batch_size={batch_size} "
        + f"elapsed={time.monotonic() - t0:.1f}s",
        logger=log,
    )

    input_dim = int(X_train_raw.shape[1])
    torch_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = LogisticMLP(input_dim, hidden_dims).to(torch_device)
    hidden_dims_str = "[" + ", ".join(str(int(d)) for d in hidden_dims) + "]"
    _emit_status(
        "[stage] model initialized: "
        f"input_dim={input_dim} "
        f"hidden_dims={hidden_dims_str} "
        f"device={torch_device} "
        f"params={sum(p.numel() for p in model.parameters())}",
        logger=log,
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if early_stop_patience < 0:
        raise ValueError("early_stop_patience must be >= 0.")
    if monitor not in {"val_loss", "val_f1"}:
        raise ValueError("monitor must be one of: val_loss, val_f1")

    def _metric_is_better(*, candidate: float, best: float) -> bool:
        if monitor.endswith("loss"):
            return candidate < best
        return candidate > best

    def _initial_best() -> float:
        if monitor.endswith("loss"):
            return float("inf")
        return float("-inf")

    best_metric = _initial_best()
    best_epoch: int | None = None
    best_model_path = output_dir / "best_model.pt"
    best_epoch_metrics: dict[str, float | int] | None = None
    best_state_dict: dict[str, torch.Tensor] | None = None

    metrics_csv_path = output_dir / "metrics.csv"
    fieldnames = [
        "epoch",
        "split",
        "loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "tn",
        "fn",
    ]
    history: list[dict[str, float | int]] = []

    feature_mean = torch.from_numpy(mean.astype(np.float32))
    feature_std = torch.from_numpy(std.astype(np.float32))

    if val_loader is None and save_best:
        log.warning(
            "save_best requested but no validation loader is available; disabling best checkpoint saving."
        )
        save_best = False
    if val_loader is None and early_stop_patience > 0:
        log.warning(
            "early_stop_patience set but no validation loader is available; disabling early stopping."
        )
        early_stop_patience = 0

    checkpoint_config: dict[str, object] = (
        dict(config)
        if config is not None
        else {
            "embeddings_npz": str(embeddings_npz),
            "metadata_file": str(metadata_file),
            "output_dir": str(output_dir),
            "epochs": epochs,
            "batch_size": batch_size,
            "hidden_dims": hidden_dims,
            "lr": lr,
            "weight_decay": weight_decay,
            "val_fraction": val_fraction,
            "seed": seed,
            "device": device,
            "num_workers": num_workers,
            "progress": progress,
            "heartbeat_seconds": heartbeat_seconds,
            "id_col": id_col,
            "label_col": label_col,
            "split_col": split_col,
            "train_splits": train_splits,
            "val_splits": val_splits,
            "save_best": save_best,
            "monitor": monitor,
            "early_stop_patience": early_stop_patience,
        }
    )

    with metrics_csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        t_train = time.monotonic()
        epoch_iter: Iterable[int]
        if progress:
            epoch_iter = tqdm(
                range(1, epochs + 1),
                desc="epochs",
                unit="epoch",
                leave=True,
                dynamic_ncols=True,
            )
        else:
            epoch_iter = range(1, epochs + 1)

        for epoch in epoch_iter:
            stop_training = False
            _emit_status(
                f"[epoch {epoch}/{epochs}] start", logger=log, quiet_with_tqdm=True
            )
            train_metrics = run_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                torch_device,
                desc=f"train {epoch}/{epochs}",
                show_progress=False,
                logger=log,
                heartbeat_seconds=heartbeat_seconds,
            )
            record: dict[str, float | int] = {
                "epoch": float(epoch),
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "train_precision": train_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "train_f1": train_metrics["f1"],
                "train_tp": train_metrics["tp"],
                "train_fp": train_metrics["fp"],
                "train_tn": train_metrics["tn"],
                "train_fn": train_metrics["fn"],
            }
            writer.writerow(
                {
                    "epoch": epoch,
                    "split": "train",
                    "loss": train_metrics["loss"],
                    "accuracy": train_metrics["accuracy"],
                    "precision": train_metrics["precision"],
                    "recall": train_metrics["recall"],
                    "f1": train_metrics["f1"],
                    "tp": train_metrics["tp"],
                    "fp": train_metrics["fp"],
                    "tn": train_metrics["tn"],
                    "fn": train_metrics["fn"],
                }
            )
            msg = (
                f"Epoch {epoch:03d}: "
                f"loss={train_metrics['loss']:.4f} "
                f"acc={train_metrics['accuracy']:.4f} "
                f"prec={train_metrics['precision']:.4f} "
                f"rec={train_metrics['recall']:.4f} "
                f"f1={train_metrics['f1']:.4f}"
            )
            if val_loader is not None:
                val_metrics = run_epoch(
                    model,
                    val_loader,
                    criterion,
                    optimizer=None,
                    device=torch_device,
                    desc=f"val {epoch}/{epochs}",
                    show_progress=False,
                    logger=log,
                    heartbeat_seconds=heartbeat_seconds,
                )
                record.update(
                    {
                        "val_loss": val_metrics["loss"],
                        "val_acc": val_metrics["accuracy"],
                        "val_precision": val_metrics["precision"],
                        "val_recall": val_metrics["recall"],
                        "val_f1": val_metrics["f1"],
                        "val_tp": val_metrics["tp"],
                        "val_fp": val_metrics["fp"],
                        "val_tn": val_metrics["tn"],
                        "val_fn": val_metrics["fn"],
                    }
                )
                writer.writerow(
                    {
                        "epoch": epoch,
                        "split": "val",
                        "loss": val_metrics["loss"],
                        "accuracy": val_metrics["accuracy"],
                        "precision": val_metrics["precision"],
                        "recall": val_metrics["recall"],
                        "f1": val_metrics["f1"],
                        "tp": val_metrics["tp"],
                        "fp": val_metrics["fp"],
                        "tn": val_metrics["tn"],
                        "fn": val_metrics["fn"],
                    }
                )
                msg += (
                    f" | val_loss={val_metrics['loss']:.4f} "
                    f"val_acc={val_metrics['accuracy']:.4f} "
                    f"val_prec={val_metrics['precision']:.4f} "
                    f"val_rec={val_metrics['recall']:.4f} "
                    f"val_f1={val_metrics['f1']:.4f}"
                )

                if save_best:
                    monitored_value = float(
                        val_metrics["loss"]
                        if monitor == "val_loss"
                        else val_metrics["f1"]
                    )
                    if not np.isfinite(monitored_value):
                        log.warning(
                            "Monitor metric '%s' is not finite; skipping best checkpoint update.",
                            monitor,
                        )
                    elif _metric_is_better(candidate=monitored_value, best=best_metric):
                        best_metric = monitored_value
                        best_epoch = epoch
                        best_epoch_metrics = {
                            "epoch": int(epoch),
                            "val_loss": float(val_metrics["loss"]),
                            "val_f1": float(val_metrics["f1"]),
                        }
                        best_state_dict = {
                            key: tensor.detach().cpu().clone()
                            for key, tensor in model.state_dict().items()
                        }
                        torch.save(
                            {
                                "model_state_dict": model.state_dict(),
                                "feature_mean": feature_mean,
                                "feature_std": feature_std,
                                "config": checkpoint_config,
                                "input_dim": input_dim,
                                "classes": {"positive": 1, "negative": 0},
                                "best": {
                                    "epoch": int(epoch),
                                    "monitor": monitor,
                                    "value": float(best_metric),
                                },
                            },
                            best_model_path,
                        )
                        _emit_status(
                            f"[ok] best checkpoint updated: {best_model_path} "
                            f"(epoch={epoch} {monitor}={best_metric:.6g})",
                            logger=log,
                            quiet_with_tqdm=True,
                        )
                    elif early_stop_patience > 0 and best_epoch is not None:
                        if (epoch - best_epoch) >= early_stop_patience:
                            _emit_status(
                                f"[stage] early stopping: no improvement in {early_stop_patience} epochs "
                                f"(best_epoch={best_epoch} {monitor}={best_metric:.6g})",
                                logger=log,
                                quiet_with_tqdm=True,
                            )
                            stop_training = True
            _emit_status(msg, logger=log, quiet_with_tqdm=True)
            _emit_status(
                f"[epoch {epoch}/{epochs}] done", logger=log, quiet_with_tqdm=True
            )
            history.append(record)
            if progress and hasattr(epoch_iter, "set_postfix"):
                epoch_iter.set_postfix(
                    {
                        "loss": f"{train_metrics['loss']:.4f}",
                        "acc": f"{train_metrics['accuracy']:.4f}",
                        "f1": f"{train_metrics['f1']:.4f}",
                    },
                    refresh=False,
                )
            if stop_training:
                break

        if progress and hasattr(epoch_iter, "close"):
            epoch_iter.close()

    _emit_status(
        f"[ok] per-epoch metrics written to {metrics_csv_path} "
        f"elapsed={time.monotonic() - t_train:.1f}s",
        logger=log,
    )

    model_path = output_dir / "model.pt"
    _emit_status("[stage] saving checkpoint", logger=log)
    t0 = time.monotonic()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "config": checkpoint_config,
            "input_dim": input_dim,
            "classes": {"positive": 1, "negative": 0},
            "best": (
                {
                    "epoch": int(best_epoch),
                    "monitor": monitor,
                    "value": float(best_metric),
                }
                if best_epoch is not None and np.isfinite(best_metric)
                else None
            ),
        },
        model_path,
    )
    _emit_status(
        f"[ok] model checkpoint saved to {model_path} elapsed={time.monotonic() - t0:.1f}s",
        logger=log,
    )

    _emit_status("[stage] plotting training history", logger=log)
    t0 = time.monotonic()
    plot_path = plot_training_history(
        history,
        output_dir=output_dir,
        filename="metrics.pdf",
    )
    if plot_path is not None:
        _emit_status(
            f"[ok] training history plot saved to {plot_path} elapsed={time.monotonic() - t0:.1f}s",
            logger=log,
        )

    val_probs_arr: np.ndarray | None = None
    val_labels_arr: np.ndarray | None = None
    val_pred_path: Path | None = None
    roc_pr_path: Path | None = None
    val_best_pred_path: Path | None = None
    roc_pr_best_path: Path | None = None
    if val_loader is not None:
        _emit_status("[stage] exporting validation predictions", logger=log)
        t0 = time.monotonic()
        val_probs_arr, val_labels_arr, val_pred_path = export_val_predictions(
            model,
            val_loader,
            val_ids,
            torch_device,
            output_dir,
            filename="val_predictions.csv",
            threshold=0.5,
            show_progress=False,
        )
        if val_pred_path is not None:
            _emit_status(
                f"[ok] validation predictions written: {val_pred_path} "
                f"elapsed={time.monotonic() - t0:.1f}s",
                logger=log,
            )
        if val_probs_arr.size and val_labels_arr.size:
            _emit_status("[stage] plotting ROC/PR curves", logger=log)
            t0 = time.monotonic()
            roc_pr_path = plot_binary_curves(
                val_labels_arr,
                val_probs_arr,
                output_path=output_dir / "roc_pr.pdf",
                split_label="val",
            )
            if roc_pr_path is not None:
                _emit_status(
                    f"[ok] ROC/PR plot saved to {roc_pr_path} elapsed={time.monotonic() - t0:.1f}s",
                    logger=log,
                )

        if save_best and best_state_dict is not None:
            _emit_status(
                "[stage] exporting validation predictions (best checkpoint)",
                logger=log,
            )
            t0 = time.monotonic()
            best_model = LogisticMLP(input_dim, hidden_dims)
            best_model.load_state_dict(best_state_dict)
            best_model.to(torch_device)
            best_probs_arr, best_labels_arr, val_best_pred_path = export_val_predictions(
                best_model,
                val_loader,
                val_ids,
                torch_device,
                output_dir,
                filename="val_predictions_best.csv",
                threshold=0.5,
                show_progress=False,
            )
            if val_best_pred_path is not None:
                _emit_status(
                    f"[ok] best validation predictions written: {val_best_pred_path} "
                    f"elapsed={time.monotonic() - t0:.1f}s",
                    logger=log,
                )
            if best_probs_arr.size and best_labels_arr.size:
                _emit_status(
                    "[stage] plotting ROC/PR curves (best checkpoint)", logger=log
                )
                t0 = time.monotonic()
                roc_pr_best_path = plot_binary_curves(
                    best_labels_arr,
                    best_probs_arr,
                    output_path=output_dir / "roc_pr_best.pdf",
                    split_label="val(best)",
                )
                if roc_pr_best_path is not None:
                    _emit_status(
                        f"[ok] best ROC/PR plot saved to {roc_pr_best_path} elapsed={time.monotonic() - t0:.1f}s",
                        logger=log,
                    )

    n_total = int(len(label_vector))
    n_pos = int((label_vector >= 0.5).sum())
    n_neg = int(n_total - n_pos)

    metrics: dict[str, object] = {
        "history": history,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_total": n_total,
        "embedding_dim": int(input_dim),
        "train_ids_sample": all_ids[: min(10, len(all_ids))],
        "metrics_csv": metrics_csv_path.name,
        "best": (
            {"epoch": int(best_epoch), "monitor": monitor, "value": float(best_metric)}
            if best_epoch is not None and np.isfinite(best_metric)
            else None
        ),
    }
    if best_epoch_metrics is not None:
        metrics["best_epoch_metrics"] = best_epoch_metrics
    if history:
        metrics["final_epoch"] = history[-1]
    if plot_path is not None:
        metrics["metrics_plot"] = (
            plot_path.name if plot_path.parent == output_dir else str(plot_path)
        )
    if val_probs_arr is not None and val_labels_arr is not None and val_probs_arr.size:
        metrics["val_labels"] = val_labels_arr.tolist()
        metrics["val_probs"] = val_probs_arr.tolist()
        if val_ids and len(val_ids) == len(val_probs_arr):
            metrics["val_ids"] = val_ids
        if val_pred_path is not None:
            metrics["val_predictions_csv"] = (
                val_pred_path.name
                if val_pred_path.parent == output_dir
                else str(val_pred_path)
            )
        if roc_pr_path is not None:
            metrics["roc_pr_plot"] = (
                roc_pr_path.name
                if roc_pr_path.parent == output_dir
                else str(roc_pr_path)
            )
        if val_best_pred_path is not None:
            metrics["val_predictions_best_csv"] = (
                val_best_pred_path.name
                if val_best_pred_path.parent == output_dir
                else str(val_best_pred_path)
            )
        if roc_pr_best_path is not None:
            metrics["roc_pr_best_plot"] = (
                roc_pr_best_path.name
                if roc_pr_best_path.parent == output_dir
                else str(roc_pr_best_path)
            )

    metrics_path = output_dir / "metrics.json"
    _emit_status("[stage] writing metrics.json", logger=log)
    t0 = time.monotonic()
    import json

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    _emit_status(
        f"[ok] training metrics written to {metrics_path} elapsed={time.monotonic() - t0:.1f}s",
        logger=log,
    )
    _emit_status(
        f"[stage] training complete total_elapsed={time.monotonic() - t_start:.1f}s",
        logger=log,
    )
