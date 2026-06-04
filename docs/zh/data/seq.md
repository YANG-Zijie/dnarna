# 简单序列输入格式（仅 ID 与序列，FASTA / CSV）

支持两种最简单的序列输入方式，用于快速加载 DNA/RNA 数据。

## 1 FASTA（`.fa`/`.fasta`）

- 记录名取自 `>` 后第一个空格前的 token，例如 `>rna1 desc` 的 ID 为 `rna1`。
- 序列可分多行、可包含空行；会自动拼接。
- 自动规范化：小写转大写，`U` 转 `T`；只允许 `A/C/G/T/N`，否则报错。

示例：

```
>rna1 desc
acg
u

>rna2 other
NNnn
```

解析结果：`{"rna1": "ACGT", "rna2": "NNNN"}`。

## 2 CSV（`.csv`，列名 `id`, `seq`）

- 表头大小写不敏感（`id`/`ID` 等）；`id` 会按字符串保留（如 `001` 不会被转成数字）。
- 自动规范化：小写转大写，`U` 转 `T`；只允许 `A/C/G/T/N`，否则报错。
- 如果出现重复 `id`，保留第一次并发出警告。

示例 `samples.csv`：

| id | seq  |
| - | - |
| r1 | acgu |
| 001 | NNnn |
| 123 | atn |

解析结果：`{"r1": "ACGT", "001": "NNNN", "123": "ATN"}`（纯数字 ID 也会以字符串形式保留，前导零不会丢失）。

## 调用方式

- FASTA: `from dnarna.data.seq.read import read_fasta` 然后 `read_fasta("path/to/sample.fa")`
- CSV: `from dnarna.data.seq.read import read_csv` 然后 `read_csv("path/to/samples.csv")`
- 自动按后缀分发：`from dnarna.data.seq.read import read_seq_dict` 然后 `read_seq_dict(path)` 会根据文件后缀选择 FASTA 或 CSV 读取。
