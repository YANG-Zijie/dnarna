"""
# 序列数据增强

本模块用于对 DNA/RNA 序列进行增强处理，以提高模型的泛化能力。

## 输入数据结构

输入数据为一个 Parquet 文件，至少包含以下列：

- id: 序列的唯一标识符
- seq: 序列字符串。目前只支持 ATCG 4 种碱基字符
- label: 序列的标签，0 表示负样本，1 表示正样本
- split: train/val/test 划分标签

输入数据结构满足 `from .validate import validate_sequence_file` 函数的要求。

## 数据增强方法

注意，所有数据增强都只在 train 集上进行，val 和 test 集保持不变。

### 随机突变

#### 参数

- mutation_rate: 突变率，表示每个序列中有多少比例的碱基将被随机替换为其他碱基。取值范围为 0 到 1。
- mutations_per_sequence: 每个序列生成的增强样本数量。

#### 新 ID

增强后的序列 ID 格式为：`{original_id}-mut-{n}`，其中 `n` 为突变样本的编号，从 1 开始。

### 输出数据结构

输出数据为一个 Parquet 文件，包含以下列：

- id: 增强后序列的唯一标识符
- seq: 增强后的序列字符串
- label: 序列的标签，保持与输入数据一致
- split: train/val/test 划分标签，保持与输入数据一致
- mutation_rate: 应用的突变率。原始序列的该字段值为 0。

注意，为便于后续训练，原始序列也会包含在输出文件中。

## 使用方法

```bash
python -m dnarna.data.seq.seq_augment \
    --input_file path/to/input.parquet \
    --output_file path/to/output.parquet \
    --mutation_rate 0.05 \
    --mutations_per_sequence 3
```
"""

import argparse
import logging
import random
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from .validate import (
    LABEL_COLUMN,
    SEQ_COLUMN,
    SEQ_ID_COLUMN,
    SPLIT_COLUMN,
    VALID_BASES,
    validate_sequence_file,
)

__all__ = [
    "random_mutate_sequence",
    "augment_rna_dataframe",
    "parse_args",
    "main",
]

TRAIN_SPLIT = "train"
OUTPUT_COLUMNS: tuple[str, ...] = (
    SEQ_ID_COLUMN,
    SEQ_COLUMN,
    LABEL_COLUMN,
    SPLIT_COLUMN,
    "mutation_rate",
)

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    LOGGER.addHandler(logging.NullHandler())


def _configure_file_logging(log_path: Path) -> None:
    """Ensure augmentation logs are persisted alongside the output file."""
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
    if LABEL_COLUMN in df.columns:
        label_counts = df[LABEL_COLUMN].value_counts().sort_index()
        label_parts = (
            ", ".join(f"{label}: {count}" for label, count in label_counts.items())
            if not label_counts.empty
            else "none"
        )
    else:
        label_parts = "unknown"

    message = f"{prefix}: total {total} rows; labels [{label_parts}]"

    if SPLIT_COLUMN in df.columns:
        split_counts = df[SPLIT_COLUMN].value_counts()
        split_parts = (
            ", ".join(f"{split}: {count}" for split, count in split_counts.items())
            if not split_counts.empty
            else "none"
        )
        message += f"; splits [{split_parts}]"
    return message


def _log_and_print(message: str) -> None:
    LOGGER.info(message)
    print(message)


def _validate_parameters(mutation_rate: float, mutations_per_sequence: int) -> None:
    if not (0.0 <= mutation_rate <= 1.0):
        raise ValueError(f"mutation_rate must be in [0, 1], got {mutation_rate}")
    if mutations_per_sequence < 1:
        raise ValueError(
            f"mutations_per_sequence must be >= 1, got {mutations_per_sequence}"
        )


def _pick_other_base(rng: random.Random, current: str) -> str:
    choices = [base for base in VALID_BASES if base != current]
    # VALID_BASES 中的字符互不相等，因此 choices 至少有 3 个元素。
    return rng.choice(choices) if choices else current


def random_mutate_sequence(
    seq: str,
    *,
    mutation_rate: float,
    rng: random.Random,
    ensure_change_when_rate_positive: bool = True,
) -> str:
    """对序列执行逐位随机突变，返回新的字符串。"""
    if not seq or mutation_rate <= 0.0:
        return seq

    chars = list(seq)
    mutated = False

    for idx, base in enumerate(chars):
        if base not in VALID_BASES:
            continue
        if rng.random() < mutation_rate:
            chars[idx] = _pick_other_base(rng, base)
            mutated = True

    if (
        ensure_change_when_rate_positive
        and mutation_rate > 0.0
        and not mutated
        and any(ch in VALID_BASES for ch in seq)
    ):
        # 强制至少突变 1 位，避免率很小时完全不变。
        eligible = [i for i, ch in enumerate(seq) if ch in VALID_BASES]
        idx = rng.choice(eligible)
        chars[idx] = _pick_other_base(rng, seq[idx])

    return "".join(chars)


def augment_rna_dataframe(
    df: pd.DataFrame,
    *,
    mutation_rate: float,
    mutations_per_sequence: int,
    seed: int | None = None,
) -> pd.DataFrame:
    """基于输入 DataFrame 生成增强后的 DataFrame（包含原始与突变样本）。"""
    _validate_parameters(mutation_rate, mutations_per_sequence)

    rng = random.Random(seed)
    augmented_rows: list[dict[str, object]] = []

    has_split_column = SPLIT_COLUMN in df.columns
    train_count = (
        int(df[SPLIT_COLUMN].eq(TRAIN_SPLIT).sum()) if has_split_column else len(df)
    )

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Copying original sequences",
        unit="seq",
        disable=len(df) == 0,
    ):
        original_id = str(row[SEQ_ID_COLUMN])
        seq = str(row[SEQ_COLUMN])
        label = int(row[LABEL_COLUMN])
        split_value = str(row[SPLIT_COLUMN]) if has_split_column else TRAIN_SPLIT

        augmented_rows.append(
            {
                SEQ_ID_COLUMN: original_id,
                SEQ_COLUMN: seq,
                LABEL_COLUMN: label,
                SPLIT_COLUMN: split_value,
                "mutation_rate": 0.0,
            }
        )

    mutation_total = train_count * mutations_per_sequence
    train_row_iter = (
        row
        for _, row in df.iterrows()
        if not has_split_column or row[SPLIT_COLUMN] == TRAIN_SPLIT
    )
    with tqdm(
        total=mutation_total,
        desc="Generating mutations",
        unit="seq",
        disable=mutation_total == 0,
    ) as mutation_progress:
        for row in train_row_iter:
            original_id = str(row[SEQ_ID_COLUMN])
            seq = str(row[SEQ_COLUMN])
            label = int(row[LABEL_COLUMN])
            split_value = str(row[SPLIT_COLUMN]) if has_split_column else TRAIN_SPLIT

            for idx in range(1, mutations_per_sequence + 1):
                mutated_seq = random_mutate_sequence(
                    seq,
                    mutation_rate=mutation_rate,
                    rng=rng,
                )
                augmented_rows.append(
                    {
                        SEQ_ID_COLUMN: f"{original_id}-mut-{idx}",
                        SEQ_COLUMN: mutated_seq,
                        LABEL_COLUMN: label,
                        SPLIT_COLUMN: split_value,
                        "mutation_rate": mutation_rate,
                    }
                )
                mutation_progress.update(1)

    return pd.DataFrame(augmented_rows, columns=OUTPUT_COLUMNS)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Perform random mutation augmentation on nucleotide sequences."
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="Path to the validated sequence Parquet file.",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Path to write the augmented Parquet file.",
    )
    parser.add_argument(
        "--mutation_rate",
        type=float,
        required=True,
        help="Mutation rate applied to each base (0.0 ~ 1.0).",
    )
    parser.add_argument(
        "--mutations_per_sequence",
        type=int,
        default=1,
        help="Number of augmented samples to generate per training sequence.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible augmentation.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_path = Path(args.output_file)
    log_path = output_path.with_suffix(output_path.suffix + ".log")
    _configure_file_logging(log_path)

    seed_repr = args.seed if args.seed is not None else "None"
    _log_and_print(f"Input file: {args.input_file}")
    _log_and_print(f"Output file: {output_path}")
    _log_and_print(
        "Parameters: "
        f"mutation_rate={args.mutation_rate:.4f}, "
        f"mutations_per_sequence={args.mutations_per_sequence}, "
        f"seed={seed_repr}"
    )

    df = validate_sequence_file(args.input_file)
    _log_and_print(_format_count_message("Input dataset", df))

    if SPLIT_COLUMN in df.columns:
        train_df = df[df[SPLIT_COLUMN] == TRAIN_SPLIT]
    else:
        train_df = df
    _log_and_print(_format_count_message("Train subset for augmentation", train_df))

    augmented = augment_rna_dataframe(
        df,
        mutation_rate=args.mutation_rate,
        mutations_per_sequence=args.mutations_per_sequence,
        seed=args.seed,
    )

    generated_augmented = len(augmented) - len(df)
    expected_generated = len(train_df) * args.mutations_per_sequence
    _log_and_print(
        f"Generated augmented sequences: {generated_augmented} "
        f"(expected {expected_generated})"
    )
    _log_and_print(_format_count_message("Augmented dataset", augmented))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    augmented.to_parquet(output_path, index=False)
    _log_and_print(f"Wrote augmented dataset to {output_path}")
    _log_and_print(f"Detailed log saved to {log_path}")


if __name__ == "__main__":
    main()
