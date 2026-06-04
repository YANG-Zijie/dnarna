"""
# 序列数据集拆分

本模块用于将 DNA/RNA 序列数据集拆分为训练集、验证集和测试集。

## 输入数据结构

输入数据为一个 Parquet 文件，至少包含以下列： 

- id: 序列的唯一标识符
- seq: 序列字符串。目前只支持 ATCG 4 种碱基字符
- label: 序列的标签，0 表示负样本，1 表示正样本

输入数据结构满足 `from .validate import validate_sequence_file` 函数的要求。

## 拆分方法

### 按比例拆分

默认train/val/test比例为8:1:1。可以通过命令行参数调整比例。拆分时会先按label进行
分层，在每个label内部打乱并按照比例切分，再将结果合并，确保各标签在train/val/test
中的占比与整体设置保持一致。

## 输出数据

输出数据为1个Parquet文件，包含以下列：

- id: 序列的唯一标识符
- seq: 序列字符串
- label: 序列的标签，0 表示负样本，1 表示正样本
- split: train/val/test划分标签

## 使用方法
```bash
python -m dnarna.data.seq.split \
    --input_file path/to/input.parquet \
    --output_dir path/to/output_dir \
    --output_format parquet \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1
```

注意，该模块会自动计算拆分比例，并确保总和为1.0。如果比例不正确，会抛出错误提示。

如果不想要test集，可以将test_ratio设置为0.0，并相应调整train_ratio和val_ratio，确保train_ratio + val_ratio = 1.0。
"""

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .validate import LABEL_COLUMN, validate_sequence_file

DEFAULT_OUTPUT_FILENAME = "splits.parquet"
LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    LOGGER.addHandler(logging.NullHandler())


def _validate_ratios(train: float, val: float, test: float) -> None:
    ratios = {"train": train, "val": val, "test": test}
    for name, ratio in ratios.items():
        if ratio < 0.0:
            raise ValueError(f"{name}_ratio must be non-negative, got {ratio}")

    total = train + val + test
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must equal 1.0, got {total}"
        )

    if test == 0.0 and val == 0.0:
        raise ValueError("At least one of val_ratio or test_ratio must be > 0.0")


def _compute_split_counts(
    total_rows: int, *, train: float, val: float, test: float
) -> dict[str, int]:
    ratios = {"train": train, "val": val, "test": test}
    counts = {
        name: int(math.floor(total_rows * ratio)) for name, ratio in ratios.items()
    }
    assigned = sum(counts.values())
    remainder = total_rows - assigned

    if remainder > 0:
        fractional_parts = sorted(
            ratios.items(),
            key=lambda item: (total_rows * item[1]) - counts[item[0]],
            reverse=True,
        )
        for name, _ratio in fractional_parts:
            if remainder == 0:
                break
            counts[name] += 1
            remainder -= 1

    return counts


def split_sequence_dataframe(
    df: pd.DataFrame,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int | None = None,
) -> dict[str, pd.DataFrame]:
    if df.empty:
        raise ValueError("Input dataframe is empty; nothing to split.")

    _validate_ratios(train_ratio, val_ratio, test_ratio)
    rng = np.random.default_rng(seed)
    max_int32 = np.iinfo(np.int32).max

    def _next_random_state() -> int:
        return int(rng.integers(0, max_int32))

    split_frames: dict[str, list[pd.DataFrame]] = {
        name: [] for name in ("train", "val", "test")
    }

    for _label, group in df.groupby(LABEL_COLUMN, sort=False):
        if group.empty:
            continue
        counts = _compute_split_counts(
            len(group), train=train_ratio, val=val_ratio, test=test_ratio
        )
        shuffled_group = group.sample(
            frac=1.0, random_state=_next_random_state()
        ).reset_index(drop=True)
        start = 0
        for split_name in ("train", "val", "test"):
            count = counts[split_name]
            end = start + count
            if count > 0:
                split_frames[split_name].append(
                    shuffled_group.iloc[start:end].reset_index(drop=True)
                )
            start = end

    empty_template = df.iloc[0:0].copy()
    splits: dict[str, pd.DataFrame] = {}
    for split_name, frames in split_frames.items():
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined = combined.sample(
                frac=1.0, random_state=_next_random_state()
            ).reset_index(drop=True)
            splits[split_name] = combined
        else:
            splits[split_name] = empty_template.copy()

    assert sum(len(frame) for frame in splits.values()) == len(df)
    return splits


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a validated nucleotide sequence dataset into train/val/test Parquet file."
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="Path to the validated sequence Parquet file.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write outputs (parquet + log).",
    )
    parser.add_argument(
        "--output_format",
        required=True,
        choices=["parquet"],
        help="Output file format (required). Currently only supports: parquet.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Proportion of data assigned to the train split. Default: 0.8.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Proportion of data assigned to the validation split. Default: 0.1.",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Proportion of data assigned to the test split. Default: 0.1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for deterministic shuffling.",
    )
    return parser.parse_args(argv)


def save_splits(
    splits: dict[str, pd.DataFrame],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_frames: list[pd.DataFrame] = []
    template_columns = list(next(iter(splits.values())).columns)

    for split_name, frame in splits.items():
        if frame.empty:
            continue
        annotated = frame.copy()
        annotated["split"] = split_name
        combined_frames.append(annotated)

    if combined_frames:
        combined = pd.concat(combined_frames, ignore_index=True)
    else:
        combined = pd.DataFrame(columns=template_columns + ["split"])

    combined.to_parquet(output_path, index=False)
    return output_path


def _configure_file_logging(log_path: Path) -> None:
    """Ensure split logs are written to a file alongside the parquet output."""
    for handler in LOGGER.handlers:
        if (
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename) == log_path
        ):
            return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s")
    )
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    LOGGER.addHandler(file_handler)


def _format_count_message(prefix: str, df: pd.DataFrame) -> str:
    total = len(df)
    label_counts = df[LABEL_COLUMN].value_counts().sort_index()
    if label_counts.empty:
        label_parts = "no rows"
    else:
        label_parts = ", ".join(
            f"label {label}: {count}" for label, count in label_counts.items()
        )
    return f"{prefix}: total {total} rows ({label_parts})"


def _log_and_print(message: str) -> None:
    LOGGER.info(message)
    print(message)


def _resolve_output_path(*, output_dir: str, output_format: str) -> Path:
    output_dir_path = Path(output_dir)
    fmt = output_format.strip().lower()
    if fmt != "parquet":
        raise ValueError(
            f"Unsupported --output_format: {output_format!r} (only 'parquet' is supported)"
        )
    return output_dir_path / DEFAULT_OUTPUT_FILENAME


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_path = _resolve_output_path(
        output_dir=args.output_dir,
        output_format=args.output_format,
    )
    log_path = output_path.with_suffix(output_path.suffix + ".log")
    _configure_file_logging(log_path)

    _log_and_print(
        "Requested split ratios: "
        f"train={args.train_ratio:.3f}, val={args.val_ratio:.3f}, test={args.test_ratio:.3f}"
    )

    df = validate_sequence_file(args.input_file)
    _log_and_print(_format_count_message("Input dataset", df))

    splits = split_sequence_dataframe(
        df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    for split_name in ("train", "val", "test"):
        frame = splits[split_name]
        _log_and_print(_format_count_message(f"{split_name} split", frame))

    total_after = sum(len(frame) for frame in splits.values())
    _log_and_print(f"Total rows across splits: {total_after}")

    final_path = save_splits(
        splits,
        output_path,
    )
    _log_and_print(f"Wrote split dataset to {final_path}")
    _log_and_print(f"Detailed log saved to {log_path}")


if __name__ == "__main__":
    main()

# Backward-compatible alias
split_rna_dataframe = split_sequence_dataframe
