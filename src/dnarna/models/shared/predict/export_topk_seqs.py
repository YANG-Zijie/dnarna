"""
Export top-ranked sequences (CSV) from prediction scores.

This is a glue utility for workflows like:
  1) Embed sequences -> infer probabilities (id, prob, pred)
  2) Select top-N (or prob>=threshold) IDs
  3) Write a filtered CSV with `id` and `seq`.

Notes:
  - CSV output keeps all columns from the original predictions file (after
    filtering/sorting/top-N) and appends the `seq` column.

Example:
  python -m dnarna.models.shared.predict.export_topk_seqs \
    --predictions outputs/dnabert2/predictions.csv \
    --seq_file data/dna.csv \
    --output outputs/dna.top200.csv \
    --top_n 200
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from dnarna.data.seq.read import read_seq_dict
from dnarna.models.shared.predict.infer import (
    CSV_EXTENSIONS,
    PARQUET_EXTENSIONS,
    infer_extension,
)

LOGGER = logging.getLogger(__name__)


def _read_predictions(path: Path) -> pd.DataFrame:
    ext = infer_extension(path)
    if ext in PARQUET_EXTENSIONS:
        return pd.read_parquet(path)
    if ext in {".json", ".jsonl"}:
        return pd.read_json(path, orient="records", lines=ext == ".jsonl")
    if ext in CSV_EXTENSIONS or not ext:
        sep = "\t" if ext.startswith(".tsv") else ","
        return pd.read_csv(path, sep=sep)
    raise ValueError(f"Unsupported predictions file extension '{ext}' for '{path}'.")


def _read_sequence_table(path: Path) -> pd.DataFrame | None:
    ext = infer_extension(path)
    if ext in PARQUET_EXTENSIONS:
        return pd.read_parquet(path)
    if ext in CSV_EXTENSIONS or not ext:
        sep = "\t" if ext.startswith(".tsv") else ","
        return pd.read_csv(path, sep=sep)
    return None


def export_topk_sequences(
    *,
    predictions_path: str | Path,
    seq_file: str | Path,
    output: str | Path,
    top_n: int | None = None,
    min_prob: float | None = None,
    only_pred_positive: bool = False,
    id_col: str = "id",
    prob_col: str = "prob",
    pred_col: str = "pred",
    ignore_missing_ids: bool = False,
) -> pd.DataFrame:
    """
    Export a subset of sequences selected by prediction scores.

    Returns:
        The filtered DataFrame including the `seq` column (sorted, after filtering & top-N cut).
    """
    predictions_path = Path(predictions_path).expanduser()
    seq_file = Path(seq_file).expanduser()
    output = Path(output).expanduser()

    df = _read_predictions(predictions_path)
    missing_cols = {id_col, prob_col} - set(df.columns)
    if missing_cols:
        raise KeyError(
            f"Predictions file missing columns: {sorted(missing_cols)}. "
            f"Available: {list(df.columns)}"
        )

    df = df.copy()

    def _ensure_alias(src: str, dst: str) -> None:
        if src == dst:
            return
        if dst in df.columns:
            a = df[dst].astype(str)
            b = df[src].astype(str)
            if not a.equals(b):
                raise ValueError(
                    f"Predictions already contain column '{dst}', but it differs from '{src}'. "
                    f"Please set --id_col/--prob_col/--pred_col accordingly."
                )
        df[dst] = df[src]

    _ensure_alias(id_col, "id")
    _ensure_alias(prob_col, "prob")
    if pred_col in df.columns:
        _ensure_alias(pred_col, "pred")

    df["id"] = df["id"].astype(str)
    df["prob"] = pd.to_numeric(df["prob"], errors="coerce")
    df = df.dropna(subset=["prob"])

    if only_pred_positive:
        if "pred" not in df.columns:
            raise KeyError(
                f"--only_pred_positive requires column '{pred_col}' in predictions."
            )
        df = df[df["pred"].astype(int) == 1]

    if min_prob is not None:
        df = df[df["prob"] >= float(min_prob)]

    df = df.sort_values("prob", ascending=False)
    if top_n is not None and int(top_n) > 0:
        df = df.head(int(top_n))

    if df.empty:
        raise ValueError("No sequences selected (filters removed all rows).")

    seq_df = _read_sequence_table(seq_file)
    if seq_df is None:
        seqs = read_seq_dict(str(seq_file))
        df["seq"] = df["id"].map(seqs)
    else:
        required_cols = {"id", "seq"} - set(seq_df.columns)
        if required_cols:
            raise KeyError(
                f"Sequence file missing columns: {sorted(required_cols)}. "
                f"Available: {list(seq_df.columns)}"
            )
        seq_df = seq_df.copy()
        seq_df["id"] = seq_df["id"].astype(str)
        if seq_df["id"].duplicated().any():
            dup_ids = seq_df.loc[seq_df["id"].duplicated(), "id"].tolist()
            raise ValueError(
                f"Sequence file contains duplicated ids: {dup_ids[:10]}"
            )
        merge_cols = ["id", "seq"] + [
            c for c in seq_df.columns if c not in {"id", "seq"} and c not in df.columns
        ]
        df = df.merge(seq_df[merge_cols], on="id", how="left")

    missing = df.loc[df["seq"].isna(), "id"].astype(str).tolist()

    if missing:
        msg = (
            f"{len(missing)} selected IDs not found in '{seq_file}' "
            f"(first 10: {missing[:10]})."
        )
        if ignore_missing_ids:
            LOGGER.warning(msg)
            df = df.dropna(subset=["seq"])
        else:
            raise KeyError(msg)

    ext = output.suffix.lower()
    if ext == ".csv":
        output.parent.mkdir(parents=True, exist_ok=True)
        cols = ["id", "seq"] + [c for c in df.columns if c not in {"id", "seq"}]
        df[cols].to_csv(output, index=False)
    else:
        raise ValueError(
            f"Unsupported output extension '{ext}' for '{output}'. Use .csv."
        )

    return df.reset_index(drop=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export top-ranked sequences from predictions."
    )
    p.add_argument(
        "--predictions",
        required=True,
        help="Predictions file (csv/tsv/parquet/json/jsonl).",
    )
    p.add_argument(
        "--seq_file",
        required=True,
        help="Original sequences file (FASTA or id,seq CSV).",
    )
    p.add_argument("--output", required=True, help="Output sequences file (.csv).")
    p.add_argument(
        "--top_n",
        type=int,
        default=None,
        help="Keep top-N by prob (default: keep all after filters).",
    )
    p.add_argument(
        "--min_prob", type=float, default=None, help="Keep rows with prob >= min_prob."
    )
    p.add_argument(
        "--only_pred_positive",
        action="store_true",
        help="Keep only rows where pred==1.",
    )
    p.add_argument(
        "--id_col", default="id", help="ID column name in predictions (default: id)."
    )
    p.add_argument(
        "--prob_col",
        default="prob",
        help="Probability column name in predictions (default: prob).",
    )
    p.add_argument(
        "--pred_col",
        default="pred",
        help="Binary prediction column name (default: pred).",
    )
    p.add_argument(
        "--ignore_missing_ids",
        action="store_true",
        help="Skip selected IDs that are missing from seq_file (otherwise error).",
    )
    p.add_argument("--verbose", action="store_true", help="Enable info logging.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    df = export_topk_sequences(
        predictions_path=args.predictions,
        seq_file=args.seq_file,
        output=args.output,
        top_n=args.top_n,
        min_prob=args.min_prob,
        only_pred_positive=args.only_pred_positive,
        id_col=args.id_col,
        prob_col=args.prob_col,
        pred_col=args.pred_col,
        ignore_missing_ids=args.ignore_missing_ids,
    )
    LOGGER.info("Selected %d sequences -> %s", len(df), args.output)


if __name__ == "__main__":
    main()
