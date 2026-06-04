"""
High-level helpers to turn the library into an end-to-end app workflow.

This module provides in-memory parsing, embedding/top-K helpers, and
pair prediction orchestration so the UI can run one-click jobs without
duplicating model logic.
"""

from __future__ import annotations

import csv
import io
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from dnarna.data.seq.write import write_id_seq_csv
from dnarna.models.dna.dnabert2.encoder import DNABERT2Encoder
from dnarna.models.rna.rnafm.encoder import RNAFMEncoder
from dnarna.models.shared.embed import load_embeddings, save_embeddings_npz
from dnarna.models.shared.predict.infer import normalize_embeddings
from dnarna.models.shared.predict.train import LogisticMLP

LOGGER = logging.getLogger(__name__)


# ----------------------------- Parsing -----------------------------


def _normalize_seq(seq: str, *, replace_u_with_t: bool = True) -> str:
    seq_norm = seq.strip().upper()
    if replace_u_with_t:
        seq_norm = seq_norm.replace("U", "T")
    allowed = {"A", "C", "G", "T", "N"}
    invalid = set(seq_norm) - allowed
    if invalid:
        bad = "".join(sorted(invalid))
        raise ValueError(
            f"Sequence contains invalid bases '{bad}' (allowed: A/C/G/T/N)"
        )
    return seq_norm


def parse_fasta_text(text: str) -> dict[str, str]:
    """Parse FASTA content from a text area."""
    seqs: dict[str, str] = {}
    name: str | None = None
    chunks: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                seqs[name] = _normalize_seq("".join(chunks))
            name = line[1:].split()[0]
            chunks = []
        else:
            chunks.append(line)
    if name is not None:
        seqs[name] = _normalize_seq("".join(chunks))
    return seqs


def parse_id_seq_csv_text(text: str) -> dict[str, str]:
    """Parse CSV text with columns id,seq (case-insensitive)."""
    fh = io.StringIO(text)
    reader = csv.DictReader(fh)
    if not reader.fieldnames:
        raise ValueError("CSV input must include a header row with id,seq columns.")
    cols = [c.strip().lower() if c else "" for c in reader.fieldnames]
    try:
        id_idx = cols.index("id")
        seq_idx = cols.index("seq")
    except ValueError as exc:
        raise ValueError("CSV header must contain id and seq columns.") from exc
    seqs: dict[str, str] = {}
    for i, row in enumerate(reader, start=2):
        values = list(row.values())
        rid = str(values[id_idx]).strip()
        raw_seq = str(values[seq_idx])
        if not rid:
            raise ValueError(f"Missing id in CSV row {i}.")
        if rid in seqs:
            LOGGER.warning(
                "Duplicate id '%s' in CSV text; keeping first occurrence.", rid
            )
            continue
        seqs[rid] = _normalize_seq(raw_seq)
    return seqs


def parse_sequences(text: str) -> dict[str, str]:
    """Heuristically parse pasted content as FASTA or id,seq CSV."""
    cleaned = text.strip()
    if not cleaned:
        return {}
    if cleaned.lstrip().startswith(">"):
        return parse_fasta_text(cleaned)
    # fallback to CSV
    return parse_id_seq_csv_text(cleaned)


# ----------------------------- Model scoring -----------------------------


@dataclass
class ScoredSequences:
    df: pd.DataFrame
    top_k: dict[str, str]
    checkpoint: str


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device)
    required = {"model_state_dict", "feature_mean", "feature_std", "input_dim"}
    missing = required - set(ckpt.keys())
    if missing:
        raise KeyError(f"Checkpoint missing keys: {sorted(missing)}")
    return ckpt


def _score_with_encoder(
    seqs: dict[str, str],
    encoder_factory: Callable[[], object],
    *,
    checkpoint: str,
    batch_size: int,
    device_override: str | None,
    desc: str,
    top_k: int | None,
    threshold: float = 0.5,
) -> ScoredSequences:
    if not seqs:
        raise ValueError("No sequences provided for scoring.")

    device = torch.device(
        device_override or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    encoder = encoder_factory()
    ids = list(seqs.keys())
    raw_seqs = [seqs[i] for i in ids]

    LOGGER.info("[stage] encoding %d sequences via %s", len(ids), desc)
    embeddings = encoder.encode_many(
        raw_seqs, l2norm=False, show_progress=True, desc=desc
    )

    LOGGER.info("[stage] loading checkpoint %s", checkpoint)
    ckpt = _load_checkpoint(Path(checkpoint).expanduser(), device)
    model = LogisticMLP(
        int(ckpt["input_dim"]), ckpt.get("config", {}).get("hidden_dim")
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    mean = ckpt["feature_mean"].cpu().numpy().astype(np.float32)
    std = ckpt["feature_std"].cpu().numpy().astype(np.float32)
    X = normalize_embeddings(embeddings.astype(np.float32), mean, std)

    with torch.no_grad():
        probs: list[float] = []
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[start : start + batch_size]).float().to(device)
            logits = model(xb)
            batch_probs = torch.sigmoid(logits).cpu().numpy().ravel().tolist()
            probs.extend(batch_probs)

    df = pd.DataFrame(
        {
            "id": ids,
            "seq": raw_seqs,
            "prob": np.array(probs, dtype=np.float32),
        }
    )
    df["pred"] = (df["prob"] >= float(threshold)).astype(int)
    df = df.sort_values("prob", ascending=False)

    k = top_k or len(df)
    selected_ids = df.head(k)["id"].tolist()
    selected = {sid: seqs[sid] for sid in selected_ids}
    return ScoredSequences(df=df, top_k=selected, checkpoint=str(checkpoint))


def score_dna_with_dnabert2(
    seqs: dict[str, str],
    *,
    checkpoint: str,
    backbone: str = "zhihan1996/DNABERT-2-117M",
    batch_size: int = 64,
    device: str | None = None,
    top_k: int | None = None,
    threshold: float = 0.5,
) -> ScoredSequences:
    def _factory() -> DNABERT2Encoder:
        return DNABERT2Encoder(backbone=backbone, batch_size=batch_size, device=device)

    return _score_with_encoder(
        seqs,
        _factory,
        checkpoint=checkpoint,
        batch_size=batch_size,
        device_override=device,
        desc="DNABERT-2",
        top_k=top_k,
        threshold=threshold,
    )


def score_rna_with_rnafm(
    seqs: dict[str, str],
    *,
    checkpoint: str,
    variant: str = "rna",
    batch_size: int = 64,
    device: str | None = None,
    top_k: int | None = None,
    threshold: float = 0.5,
) -> ScoredSequences:
    def _factory() -> RNAFMEncoder:
        return RNAFMEncoder(variant=variant, batch_size=batch_size, device=device)

    return _score_with_encoder(
        seqs,
        _factory,
        checkpoint=checkpoint,
        batch_size=batch_size,
        device_override=device,
        desc=f"RNA-FM({variant})",
        top_k=top_k,
        threshold=threshold,
    )


# ----------------------------- Scenario orchestration -----------------------------


@dataclass
class AppJobConfig:
    output_dir: str | Path | None
    dnabert_checkpoint: str | None = None
    rnafm_checkpoint: str | None = None
    pair_checkpoint: str | None = None
    top_k_dna: int | None = None
    top_k_rna: int | None = None
    window_size: int = 1000
    window_stride: int = 500
    dnabert_backbone: str = "zhihan1996/DNABERT-2-117M"
    rnafm_variant: str = "rna"
    dna_device: str | None = None
    rna_device: str | None = None
    dna_embed_batch_size: int = 1024
    rna_embed_batch_size: int = 256
    score_batch_size: int = 1024
    score_threshold: float = 0.5
    pair_feature_mode: str = "concat"
    pair_dna_block_size: int = 64
    pair_rna_block_size: int = 64
    pair_batch_size: int = 4096
    pair_chunk_size: int = 4096
    pair_num_workers: int = 1
    pair_threshold: float = 0.5
    pair_device: str | None = None
    pair_max_dna: int = 0
    pair_max_rna: int = 0
    save_artifacts: bool = True


@dataclass
class AppJobResult:
    used_dna_path: Path
    used_rna_path: Path
    used_dna_count: int
    used_rna_count: int
    dna_scores_path: Path | None
    rna_scores_path: Path | None
    dna_predictions_path: Path | None
    rna_predictions_path: Path | None
    pair_predictions_path: Path | None
    pair_summary_path: Path | None
    pair_meta_path: Path | None
    mode: str


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        count = -1
        for count, _ in enumerate(handle):
            pass
    return max(0, count)


def _load_id_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Sequence file not found: {path}")
    df = pd.read_csv(path, usecols=["id"])
    if "id" not in df.columns:
        raise ValueError(f"Sequence file missing id column: {path}")
    return df["id"].astype(str).tolist()


def _filter_embeddings_by_ids(
    embeddings_path: Path, allowed_ids: list[str]
) -> tuple[np.ndarray, list[str], list[str]]:
    embeddings, emb_ids = load_embeddings(embeddings_path)
    emb_ids = [str(x) for x in emb_ids]
    allowed = set(str(x) for x in allowed_ids)
    indices = [idx for idx, emb_id in enumerate(emb_ids) if emb_id in allowed]
    filtered_ids = [emb_ids[idx] for idx in indices]
    missing = sorted(allowed - set(filtered_ids))
    if not indices:
        raise ValueError(
            f"No embeddings match selected IDs from {embeddings_path}."
        )
    return embeddings[indices], filtered_ids, missing


def _prepare_pair_embeddings(
    embeddings_path: Path, used_seq_path: Path, output_dir: Path
) -> Path:
    expected_name = f"{used_seq_path.stem}.embeddings.npz"
    if embeddings_path.name == expected_name:
        return embeddings_path

    ids = _load_id_list(used_seq_path)
    filtered_embeddings, filtered_ids, missing = _filter_embeddings_by_ids(
        embeddings_path, ids
    )
    if missing:
        LOGGER.warning(
            "Missing %d IDs in embeddings for %s (first 10: %s).",
            len(missing),
            used_seq_path,
            missing[:10],
        )
    output_path = output_dir / expected_name
    save_embeddings_npz(output_path, filtered_ids, filtered_embeddings)
    LOGGER.info(
        "Filtered embeddings written to %s (n=%d).",
        output_path,
        len(filtered_ids),
    )
    return output_path


def _run_cli_step(args: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        proc = subprocess.run(
            args,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if proc.returncode != 0:
        cmd = " ".join(args)
        raise RuntimeError(
            f"Command failed (exit={proc.returncode}): {cmd}. See {log_path}"
        )


def run_app_job(
    *,
    dna_seqs: dict[str, str],
    rna_seqs: dict[str, str],
    cfg: AppJobConfig,
) -> AppJobResult:
    if not cfg.output_dir:
        raise ValueError("output_dir 不能为空。")
    if not cfg.pair_checkpoint:
        raise ValueError("需要提供 DNA-RNA pair checkpoint 路径。")
    if not dna_seqs:
        raise ValueError("DNA 序列为空，请先提供 DNA 输入。")
    if not rna_seqs:
        raise ValueError("RNA 序列为空，请先提供 RNA 输入。")

    out_dir = Path(cfg.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    inputs_dir = out_dir / "inputs"
    processed_dir = out_dir / "processed"
    processed_dna_dir = processed_dir / "dna"
    processed_rna_dir = processed_dir / "rna"
    predictions_dir = out_dir / "predictions"
    topk_dir = out_dir / "topk"
    pair_predictions_dir = out_dir / "pair_predictions"

    inputs_dir.mkdir(parents=True, exist_ok=True)
    processed_dna_dir.mkdir(parents=True, exist_ok=True)
    processed_rna_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    topk_dir.mkdir(parents=True, exist_ok=True)
    pair_predictions_dir.mkdir(parents=True, exist_ok=True)

    dna_input_path = inputs_dir / "dna.csv"
    rna_input_path = inputs_dir / "rna.csv"
    write_id_seq_csv(dna_seqs, dna_input_path)
    write_id_seq_csv(rna_seqs, rna_input_path)

    window_args_common = [
        sys.executable,
        "-m",
        "dnarna.data.seq.window",
        "--output_format",
        "csv",
        "--window_size",
        str(int(cfg.window_size)),
        "--stride",
        str(int(cfg.window_stride)),
    ]
    _run_cli_step(
        window_args_common
        + ["--input_file", str(dna_input_path), "--output_dir", str(processed_dna_dir)],
        logs_dir / "window_dna.log",
    )
    _run_cli_step(
        window_args_common
        + ["--input_file", str(rna_input_path), "--output_dir", str(processed_rna_dir)],
        logs_dir / "window_rna.log",
    )

    dna_windowed = processed_dna_dir / "dna.windowed.csv"
    rna_windowed = processed_rna_dir / "rna.windowed.csv"
    dna_embeddings_path = processed_dna_dir / f"{dna_windowed.stem}.embeddings.npz"
    rna_embeddings_path = processed_rna_dir / f"{rna_windowed.stem}.embeddings.npz"

    dna_embed_args = [
        sys.executable,
        "-m",
        "dnarna.models.dna.dnabert2.embed",
        "--input_file",
        str(dna_windowed),
        "--output_dir",
        str(processed_dna_dir),
        "--backbone",
        str(cfg.dnabert_backbone),
        "--max_length",
        str(int(cfg.window_size)),
        "--batch_size",
        str(int(cfg.dna_embed_batch_size)),
    ]
    if cfg.dna_device:
        dna_embed_args += ["--device", str(cfg.dna_device)]
    _run_cli_step(dna_embed_args, logs_dir / "dnabert2_embed_dna.log")

    rna_embed_args = [
        sys.executable,
        "-m",
        "dnarna.models.rna.rnafm.embed",
        "--input_file",
        str(rna_windowed),
        "--output_dir",
        str(processed_rna_dir),
        "--max_length",
        str(int(cfg.window_size)),
        "--variant",
        str(cfg.rnafm_variant),
        "--batch_size",
        str(int(cfg.rna_embed_batch_size)),
    ]
    if cfg.rna_device:
        rna_embed_args += ["--device", str(cfg.rna_device)]
    _run_cli_step(rna_embed_args, logs_dir / "rnafm_embed_rna.log")

    dna_k = int(cfg.top_k_dna or 0)
    rna_k = int(cfg.top_k_rna or 0)
    needs_filter = dna_k > 0 or rna_k > 0
    mode = "topk_filter" if needs_filter else "all_pairs"

    dna_predictions_path = None
    rna_predictions_path = None
    dna_scores_path = None
    rna_scores_path = None
    used_dna_path = dna_windowed
    used_rna_path = rna_windowed

    if dna_k > 0:
        if not cfg.dnabert_checkpoint:
            raise ValueError("需要 top-K DNA 时必须提供 DNABERT-2 checkpoint 路径。")
        dna_predictions_path = predictions_dir / "dna_predictions.csv"
        infer_args = [
            sys.executable,
            "-m",
            "dnarna.models.dna.dnabert2.predict.infer",
            "--embeddings_npz",
            str(dna_embeddings_path),
            "--checkpoint",
            str(cfg.dnabert_checkpoint),
            "--output",
            str(dna_predictions_path),
            "--batch_size",
            str(int(cfg.score_batch_size)),
            "--threshold",
            str(float(cfg.score_threshold)),
        ]
        if cfg.dna_device:
            infer_args += ["--device", str(cfg.dna_device)]
        _run_cli_step(infer_args, logs_dir / "dnabert2_predict_dna.log")
        dna_scores_path = dna_predictions_path

        used_dna_path = topk_dir / "dna.topk.csv"
        export_args = [
            sys.executable,
            "-m",
            "dnarna.models.shared.predict.export_topk_seqs",
            "--predictions",
            str(dna_predictions_path),
            "--seq_file",
            str(dna_windowed),
            "--output",
            str(used_dna_path),
            "--top_n",
            str(dna_k),
        ]
        _run_cli_step(export_args, logs_dir / "dnabert2_topk_dna.log")

    if rna_k > 0:
        if not cfg.rnafm_checkpoint:
            raise ValueError("需要 top-K RNA 时必须提供 RNA-FM checkpoint 路径。")
        rna_predictions_path = predictions_dir / "rna_predictions.csv"
        infer_args = [
            sys.executable,
            "-m",
            "dnarna.models.rna.rnafm.predict.infer",
            "--embeddings_npz",
            str(rna_embeddings_path),
            "--checkpoint",
            str(cfg.rnafm_checkpoint),
            "--output",
            str(rna_predictions_path),
            "--batch_size",
            str(int(cfg.score_batch_size)),
            "--threshold",
            str(float(cfg.score_threshold)),
        ]
        if cfg.rna_device:
            infer_args += ["--device", str(cfg.rna_device)]
        _run_cli_step(infer_args, logs_dir / "rnafm_predict_rna.log")
        rna_scores_path = rna_predictions_path

        used_rna_path = topk_dir / "rna.topk.csv"
        export_args = [
            sys.executable,
            "-m",
            "dnarna.models.shared.predict.export_topk_seqs",
            "--predictions",
            str(rna_predictions_path),
            "--seq_file",
            str(rna_windowed),
            "--output",
            str(used_rna_path),
            "--top_n",
            str(rna_k),
        ]
        _run_cli_step(export_args, logs_dir / "rnafm_topk_rna.log")

    pair_dna_embeddings = _prepare_pair_embeddings(
        dna_embeddings_path, used_dna_path, processed_dna_dir
    )
    pair_rna_embeddings = _prepare_pair_embeddings(
        rna_embeddings_path, used_rna_path, processed_rna_dir
    )
    pair_output_name = "pair_predictions.csv"
    pair_predictions_path = pair_predictions_dir / pair_output_name
    pair_args = [
        sys.executable,
        "-m",
        "dnarna.models.pair.predict.infer",
        "--all_pairs",
        "--dna_embeddings",
        str(pair_dna_embeddings),
        "--rna_embeddings",
        str(pair_rna_embeddings),
        "--checkpoint",
        str(cfg.pair_checkpoint),
        "--dna_seq_file",
        str(used_dna_path),
        "--rna_seq_file",
        str(used_rna_path),
        "--output_dir",
        str(pair_predictions_dir),
        "--output_name",
        pair_output_name,
        "--feature_mode",
        str(cfg.pair_feature_mode),
        "--dna_block_size",
        str(int(cfg.pair_dna_block_size)),
        "--rna_block_size",
        str(int(cfg.pair_rna_block_size)),
        "--batch_size",
        str(int(cfg.pair_batch_size)),
        "--chunk_size",
        str(int(cfg.pair_chunk_size)),
        "--num_workers",
        str(int(cfg.pair_num_workers)),
        "--threshold",
        str(float(cfg.pair_threshold)),
    ]
    if cfg.pair_device:
        pair_args += ["--device", str(cfg.pair_device)]
    if int(cfg.pair_max_dna) > 0:
        pair_args += ["--max_dna", str(int(cfg.pair_max_dna))]
    if int(cfg.pair_max_rna) > 0:
        pair_args += ["--max_rna", str(int(cfg.pair_max_rna))]
    _run_cli_step(pair_args, logs_dir / "pair_infer.log")

    pair_meta_path = pair_predictions_path.with_suffix(
        pair_predictions_path.suffix + ".meta.json"
    )
    pair_summary_path = pair_predictions_path.with_name(
        f"{pair_predictions_path.stem}.summary{pair_predictions_path.suffix}"
    )
    if not pair_predictions_path.exists():
        pair_predictions_path = None
    if not pair_summary_path.exists():
        pair_summary_path = None
    if not pair_meta_path.exists():
        pair_meta_path = None

    result = AppJobResult(
        used_dna_path=used_dna_path,
        used_rna_path=used_rna_path,
        used_dna_count=_count_csv_rows(used_dna_path),
        used_rna_count=_count_csv_rows(used_rna_path),
        dna_scores_path=dna_scores_path,
        rna_scores_path=rna_scores_path,
        dna_predictions_path=dna_predictions_path,
        rna_predictions_path=rna_predictions_path,
        pair_predictions_path=pair_predictions_path,
        pair_summary_path=pair_summary_path,
        pair_meta_path=pair_meta_path,
        mode=mode,
    )
    if not cfg.save_artifacts:
        for path in [logs_dir, inputs_dir, processed_dir, predictions_dir, topk_dir]:
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
    return result
