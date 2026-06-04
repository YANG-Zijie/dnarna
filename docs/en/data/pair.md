# DNA-RNA Pair Data

The dataset is composed of three files.

## File Structure

### File 1: DNA-RNA pair data

Contains three columns:

- `pair_id`: unique identifier for each DNA-RNA sequence pair
- `dna_id`: ID of the corresponding DNA sequence
- `rna_id`: ID of the corresponding RNA sequence

### File 2: DNA sequence data

Contains two columns:

- `dna_id`: DNA sequence ID
- `dna_seq`: DNA sequence string

### File 3: RNA sequence data

Contains two columns:

- `rna_id`: RNA sequence ID
- `rna_seq`: RNA sequence string

## Data Cleaning

The loader performs the following cleaning steps:

1. For files 2 and 3, remove records with empty sequences and log the removed IDs.
2. For files 2 and 3, convert sequences to uppercase and replace `U` with `T` in RNA sequences.
3. For files 2 and 3, remove sequences containing bases outside `A/C/G/T`, and log the removed IDs.
4. For files 2 and 3, remove duplicate IDs while keeping the first occurrence, and log the removed duplicate IDs.
5. For files 2 and 3, remove records with identical sequences but different IDs while keeping the first occurrence, and log the removed IDs.
   The corresponding `dna_id` / `rna_id` values in the pair file are automatically mapped to the retained IDs, avoiding accidental pair deletion caused by sequence deduplication.
6. For file 1, remove records that reference missing DNA or RNA IDs, and log the corresponding `pair_id` values.
7. Deduplicate file 1 by `dna_id` + `rna_id`, keeping only the first occurrence and logging duplicate `pair_id` values.

### Cleaning Script

The cleaning script is located at `src/dnarna/data/pair/clean.py`. It can process the three raw input files from the command line and write cleaned outputs:

```bash
python -m dnarna.data.pair.clean \
    --pair_file path/to/pairs.csv \
    --dna_file path/to/dna.csv \
    --rna_file path/to/rna.csv \
    --output_dir path/to/output_dir \
    --output_format csv
```

Arguments:

- `--pair_file`: file 1 (pair) path, supporting `.csv` / `.parquet`
- `--dna_file`: file 2 (DNA) path, supporting `.csv` / `.parquet`
- `--rna_file`: file 3 (RNA) path, supporting `.csv` / `.parquet`
- `--output_dir`: output directory
- `--output_format`: output format (`csv` or `parquet`)
- `--pair_output` / `--dna_output` / `--rna_output`: optional output file names

Output file names are generated from input file names by default, with the extension adjusted according to `--output_format`. The log and report file names follow the pair output file:

- `<pair_output>.log`: detailed cleaning log
- `<pair_output>.report.json`: cleaning statistics and removed-record summary

### Cleaned Outputs

After cleaning, the script still outputs the three main data files (files 1/2/3), plus:

- `*.json`: cleaning statistics and removed-record summary
- `*.log`: detailed cleaning log and removed ID lists

## Long-Sequence Windowing

If DNA/RNA sequences may be very long, it is recommended to window DNA and RNA separately, then use `parent_id` to trace back to the original pairs and generate new unique `pair_id` values. Recommended defaults:

- `window_size=1000`
- `stride=500`

Each windowed sequence includes the following metadata:

- `parent_id`: original sequence ID
- `window_index`: window index (0 for unwindowed sequences; 1-based for split windows)
- `window_start` / `window_end`: interval in the original sequence

Recommended processing order:

1. Run cleaning first (deduplication, validation, ID mapping).
2. Run windowing next (generate window sequences and window metadata).
3. Expand windowed pairs using `parent_id` and generate new `pair_id` values.

The windowing module is `src/dnarna/data/pair/window.py`. Default output:
`*.windowed.csv` (or `.parquet`).

```bash
python -m dnarna.data.pair.window \
    --pair_file path/to/pairs.csv \
    --dna_file path/to/dna.csv \
    --rna_file path/to/rna.csv \
    --output_dir path/to/output_dir \
    --output_format csv \
    --window_size 1000 \
    --stride 500
```

If DNA and RNA require different window parameters, use:

- `--dna_window_size` / `--dna_stride`
- `--rna_window_size` / `--rna_stride`

When these are not specified, the script uses `window_size/stride`; if `stride` is omitted, it defaults to `window_size//2`.

Example (`window_size=1000`, `stride=500`, one long DNA and one long RNA):

- Input: `pair_id=P1`, `dna_id=D1(len=2500)`, `rna_id=R1(len=1800)`
- DNA window IDs: `D1_win_1`, `D1_win_2`, `D1_win_3`, `D1_win_4`
- RNA window IDs: `R1_win_1`, `R1_win_2`, `R1_win_3`
- New `pair_id` rule: `P1__d{dna_window_index}__r{rna_window_index}`
  - Examples: `P1__d1__r1`, `P1__d1__r2`, ..., `P1__d4__r3`
- Unsplit sequences use window index 0, so their pair IDs may contain `d0` / `r0`.
