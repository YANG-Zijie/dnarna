# Simple Sequence Input Format (ID and Sequence Only, FASTA / CSV)

DnaRna supports two simple sequence input formats for quickly loading DNA/RNA data.

## 1 FASTA (`.fa`/`.fasta`)

- The record name is the token immediately after `>` and before the first whitespace. For example, `>rna1 desc` uses `rna1` as the ID.
- Sequences may span multiple lines and may contain blank lines; they are concatenated automatically.
- Automatic normalization: lowercase letters are converted to uppercase, `U` is converted to `T`, and only `A/C/G/T/N` are allowed. Other bases raise an error.

Example:

```text
>rna1 desc
acg
u

>rna2 other
NNnn
```

Parsed result: `{"rna1": "ACGT", "rna2": "NNNN"}`.

## 2 CSV (`.csv`, columns `id`, `seq`)

- Header names are case-insensitive (`id`, `ID`, etc.). `id` values are preserved as strings, so IDs such as `001` are not converted to numbers.
- Automatic normalization: lowercase letters are converted to uppercase, `U` is converted to `T`, and only `A/C/G/T/N` are allowed. Other bases raise an error.
- If duplicate `id` values are present, the first record is kept and a warning is emitted.

Example `samples.csv`:

| id | seq |
| - | - |
| r1 | acgu |
| 001 | NNnn |
| 123 | atn |

Parsed result: `{"r1": "ACGT", "001": "NNNN", "123": "ATN"}`. Numeric-looking IDs are preserved as strings, including leading zeros.

## Usage

- FASTA: `from dnarna.data.seq.read import read_fasta`, then call `read_fasta("path/to/sample.fa")`
- CSV: `from dnarna.data.seq.read import read_csv`, then call `read_csv("path/to/samples.csv")`
- Auto-dispatch by extension: `from dnarna.data.seq.read import read_seq_dict`, then call `read_seq_dict(path)` to choose FASTA or CSV parsing based on the file suffix.
