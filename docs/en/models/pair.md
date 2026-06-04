# DNA-RNA Pair Prediction (Embedding Based)

This model predicts whether a given DNA-RNA pair is likely to bind. The core approach is to construct a pair feature from DNA/RNA embeddings and train a shallow MLP binary classifier.

## Data Preparation

1. DNA/RNA sequences must be embedded separately (DNABERT-2 / RNA-FM), producing `.npz` files:
   - Each file must contain `ids` and `embeddings`.
2. The pair file should contain at least:
   - `pair_id` (optional; generated automatically if absent)
   - `dna_id` / `rna_id`
   - `label` (optional; if absent, all pairs are treated as positives)
   - `split` (optional; specifies train/val and is used only when negatives are not sampled)
3. Each `dna_id` / `rna_id` must exist in the corresponding embedding `ids`; otherwise that pair is dropped.

For windowed data, make sure both `pairs` and `embeddings` are windowed versions and that IDs match.

## Negative Sampling

Two cases are supported:

- **Existing negatives (`label=0/1`)**: use them directly; no additional negative sampling is performed.
- **Positive pairs only**: treat all input pairs as positives and sample negatives according to `negative_ratio`. Negatives are random DNA ID and RNA ID combinations that do not overlap with the positive set.

The default `negative_ratio` is `1.0` (1:1 positive-to-negative ratio). These random negatives may contain true positives and should be treated as weak negatives when interpreting results.

## Pair Feature Construction

Pair features are constructed from DNA/RNA embeddings:

- `concat` (default): `[dna; rna]`, dimension = D_dna + D_rna
- `absdiff`: `|dna - rna|` (requires D_dna == D_rna)
- `mul`: `dna * rna` (requires D_dna == D_rna)
- `all`: `[dna; rna; |dna-rna|; dna*rna]` (requires D_dna == D_rna)

## Training Workflow

1. Generate pair features and save them as `<pairs>.pair_embeddings.npz`.
2. Generate training metadata `<pairs>.pair_metadata.csv` with fields such as `label` and `source`.
3. Train a binary classifier with an MLP (`hidden_dims` is required).
4. Train/validation split:
   - If `split` exists and no negatives are sampled, use `split`.
   - Otherwise, randomly split by `val_fraction` (default 0.1, i.e. 90% train / 10% val).

Training outputs:

- `model.pt` / `best_model.pt`
- `metrics.csv` / `metrics.json` / `metrics.pdf`
- If a validation set exists: `val_predictions.csv`, `roc_pr.pdf`
- Process report: `*.pair_report.json`

## Reusing Generated Pair Features

If `*.pair_embeddings.npz` and `*.pair_metadata.csv` are already available, you can skip feature generation and negative sampling:

```bash
python -m dnarna.models.pair.predict.train \
  --pairs_file path/to/pairs.csv \
  --pair_embeddings_input path/to/pairs.pair_embeddings.npz \
  --metadata_input path/to/pairs.pair_metadata.csv \
  --output_dir path/to/output \
  --hidden_dims 512,256
```

In this mode, `--negative_ratio` and `--max_pairs` are ignored, and the metadata must contain a `label` column.

## Training Example

```bash
python -m dnarna.models.pair.predict.train \
  --pairs_file path/to/pairs.csv \
  --dna_embeddings path/to/dna.embeddings.npz \
  --rna_embeddings path/to/rna.embeddings.npz \
  --output_dir path/to/output \
  --hidden_dims 512,256 \
  --feature_mode concat \
  --negative_ratio 1.0
```

## Inference

Inference must use the same `feature_mode` as training; otherwise the script reports a mismatch.
Run `python -m dnarna.models.pair.predict.infer` to output `prob/pred` for each pair.
Using `--output_dir` also writes `*.log`, `*.meta.json`, and, when window metadata can be inferred, an additional `*.summary.csv` aggregation report.

### Aggregated Scores in Windowed Workflows

When a long DNA/RNA sequence is split into multiple windows, the raw output gives one probability per `window-pair`.
The inference pipeline now also generates a summary that aggregates all window-pairs belonging to the same original DNA-RNA pair into one row. Common columns include:

- `pair_predictions.csv`: one row per `window-pair`
- `pair_predictions.summary.csv`: one row per original DNA-RNA pair
- `combined_score`: overall score indicating whether at least one window-pair is likely positive, based on noisy-or aggregation
- `prob_max`: strongest single window-pair score
- `prob_mean`: mean score across all window-pairs; this is normalized by pair count and does not directly indicate how many high-scoring windows exist
- `positive_window_pair_count`: number of window-pairs with probability greater than or equal to the current classification threshold (`threshold`)
- `positive_window_pair_fraction`: fraction of window-pairs with probability greater than or equal to the current threshold, i.e. `positive_window_pair_count / window_pair_count`
- `window_pair_count` / `dna_window_count` / `rna_window_count`
- `best_window_pair_id` and the corresponding window index/start/end

If the goal is to find whether there is a local strong binding site, prioritize `combined_score` and `prob_max`.
If the goal is to assess whether scores are broadly elevated, inspect `prob_mean` and `positive_window_pair_fraction` as well.

The outputs can be understood as two layers:

| File | One row represents | Typical use |
| --- | --- | --- |
| `pair_predictions.csv` | One `window-pair` | Inspect raw local window-pair scores |
| `pair_predictions.summary.csv` | One original DNA-RNA pair | Inspect the overall long-sequence pair result |

For example, suppose:

- One DNA sequence is split into 5 windows
- One RNA sequence is split into 3 windows

The raw predictions contain `5 x 3 = 15` `window-pair` records, but the summary contains only one row for the original DNA-RNA pair.

Raw `pair_predictions.csv` may look like:

| pair_id | dna_id | rna_id | prob | pred |
| --- | --- | --- | ---: | ---: |
| `dnaA_win_0__rnaB_win_0` | `dnaA_win_0` | `rnaB_win_0` | 0.82 | 1 |
| `dnaA_win_0__rnaB_win_1` | `dnaA_win_0` | `rnaB_win_1` | 0.11 | 0 |
| `dnaA_win_1__rnaB_win_0` | `dnaA_win_1` | `rnaB_win_0` | 0.64 | 1 |
| `...` | `...` | `...` | `...` | `...` |

The corresponding `pair_predictions.summary.csv` may look like:

| pair_group_id | dna_parent_id | rna_parent_id | dna_window_count | rna_window_count | window_pair_count | prob_max | prob_mean | combined_score | positive_window_pair_count | best_window_pair_id |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `dnaA__rnaB` | `dnaA` | `rnaB` | 5 | 3 | 15 | 0.91 | 0.37 | 0.98 | 4 | `dnaA_win_2__rnaB_win_1` |

In other words, the summary does not discard the raw window-pair details. It provides an additional table aggregated by original DNA-RNA pair.

### What `combined_score` Means

Assume an original DNA-RNA pair produces `k` window-pairs, with probabilities:

`p1, p2, ..., pk`

The combined score is:

`combined_score = 1 - (1-p1)(1-p2)...(1-pk)`

That is:

- Convert each window-pair probability to a non-hit probability `1-pi`.
- Multiply the probabilities that all window-pairs are non-hits.
- Subtract the result from 1.
- The result is the overall probability-like score that at least one window-pair is a hit.

Intuitively:

- A single strong local window can make `combined_score` high.
- Multiple moderately high windows can also raise `combined_score`.
- Therefore, it is not an average score; it is accumulated evidence that at least one local binding region may exist.

Simple examples:

- With one window, `p=[0.8]`
  - `combined_score = 0.8`
- With two windows, `p=[0.8, 0.1]`
  - `combined_score = 1 - (0.2 * 0.9) = 0.82`
  - This is slightly higher than `0.8` because the second window adds some evidence.
- With three moderate windows, `p=[0.4, 0.4, 0.4]`
  - `combined_score = 1 - (0.6^3) = 0.784`
  - None of the individual windows is very high, but several moderate signals lift the overall score.

Differences from other metrics:

- `prob_max`: only the strongest window
- `prob_mean`: average signal level
- `combined_score`: accumulated evidence that at least one local region may bind

### What Noisy-Or Means

`noisy-or` is a common aggregation idea in probabilistic graphical models.

The intuition is:

- Each `window-pair` is a local cause that may make the overall pair positive.
- The overall event is an OR over these local causes: if any one is active, the whole pair may be positive.
- Each local cause is uncertain, so the OR is "noisy" rather than a hard logical OR.

Here, noisy-or is not a new model output; it is a post-processing aggregation over existing `window-pair` probabilities.

### Practical Interpretation

- `combined_score` tends to increase more easily as the number of `window-pairs` increases.
- It is most suitable for asking whether a long DNA/RNA pair contains at least one local strong binding site.
- When comparing samples with different lengths or different numbers of windows, it is best to inspect:
  - `prob_max`
  - `prob_mean`
  - `positive_window_pair_fraction`

A practical reading guide:

- To find whether there is a clear hotspot, use `combined_score` and `prob_max`.
- To inspect whether many regions are elevated, use `prob_mean` and `positive_window_pair_fraction`.
- To count high-scoring local regions, use `positive_window_pair_count`.

### How to Interpret `prob_mean` and High-Scoring Window Counts

- `prob_mean` is a simple average, so it is not proportional to `window_pair_count`. More window-pairs do not automatically raise `prob_mean`.
- However, `prob_mean` also does not tell you how many high-scoring window-pairs exist; it only describes the overall average level.
- To judge whether a given DNA/RNA pair has more potentially high-binding local regions, `positive_window_pair_count` and `positive_window_pair_fraction` are more direct.
- Specifically:
  - `positive_window_pair_count` is an absolute count and is useful for asking how many high-scoring local sites exist.
  - `positive_window_pair_fraction` is a density/fraction and is usually more comparable across samples with different lengths and different window counts.

Useful interpretation:

- Use `combined_score` / `prob_max` to identify clear strong binding windows.
- Use `positive_window_pair_count` to measure the absolute number of strong windows.
- Use `positive_window_pair_fraction` / `prob_mean` to assess whether the whole pair is broadly elevated.

### Generate Summary from Existing Predictions

If a full `pair_predictions.csv` already exists and you do not want to rerun expensive pair inference, run the summary script separately.

For `all_pairs` workflows, it is recommended to provide windowed DNA/RNA files to recover parent/window metadata:

```bash
python -m dnarna.models.pair.predict.summarize \
  --predictions path/to/pair_predictions.csv \
  --dna_seq_file path/to/dna.windowed.csv \
  --rna_seq_file path/to/rna.windowed.csv \
  --chunksize 200000
```

For inference based on a windowed pair file, you can also provide the original `pairs.windowed.csv`:

```bash
python -m dnarna.models.pair.predict.summarize \
  --predictions path/to/pair_predictions.csv \
  --pairs_file path/to/pairs.windowed.csv
```

To inspect `positive_window_pair_count` / `positive_window_pair_fraction` under multiple thresholds, add `--thresholds`:

```bash
python -m dnarna.models.pair.predict.summarize \
  --predictions path/to/pair_predictions.csv \
  --dna_seq_file path/to/dna.windowed.csv \
  --rna_seq_file path/to/rna.windowed.csv \
  --thresholds 0.5,0.6,0.7,0.8,0.9
```

This generates an additional long-format file:

- `pair_predictions.summary.by_threshold.csv`

Each row represents:

- One original DNA-RNA pair
- Statistics under one threshold

Example:

| pair_group_id | threshold | window_pair_count | positive_window_pair_count | positive_window_pair_fraction |
| --- | ---: | ---: | ---: | ---: |
| `dnaA__rnaB` | 0.5 | 15 | 4 | 0.267 |
| `dnaA__rnaB` | 0.6 | 15 | 3 | 0.200 |
| `dnaA__rnaB` | 0.7 | 15 | 2 | 0.133 |

Additional notes:

- If `pair_predictions.csv` already contains columns such as `pair_parent_id` / `dna_parent_id` / `rna_parent_id`, these columns are used directly to generate the summary.
- If those columns are absent but `--pairs_file` or `--dna_seq_file/--rna_seq_file` is provided, the explicit metadata is used to recover parent information.
- If no extra metadata is provided, IDs such as `xxx_win_1`, `xxx_win_2` are parsed back to `xxx` automatically, and summaries are generated from that.
- If IDs do not have window suffixes, the original `dna_id/rna_id` values are treated as parents.
- By default, if the prediction file already has a `pred` column, `positive_window_pair_count` / `positive_window_pair_fraction` are computed from `pred`; for outputs produced by `infer`, this is equivalent to the inference threshold, which defaults to `prob >= 0.5`.
- Use `--ignore_pred_col` to ignore an existing `pred` column and recompute `positive_window_pair_count` from a new `--threshold`.
- The `by_threshold` report generated by `--thresholds` always recomputes from raw `prob >= threshold`, independent of any existing `pred` column.
- `by_threshold` keeps only fields that actually vary by threshold; fixed fields such as `combined_score` remain only in the main `summary.csv`.
- `summarize` now reads `pair_predictions.csv` in chunks by default via `--chunksize`, so it can handle very large CSV/TSV files without loading the whole table into memory.
- For very large jobs, prefer `--dna_seq_file/--rna_seq_file`; `--pairs_file` may still consume substantial memory if the pair file itself is large.
- If the dataset is known to be small, set `--chunksize 0` to use full-table loading.

Example:

```bash
python -m dnarna.models.pair.predict.infer \
  --pairs_file path/to/pairs.csv \
  --dna_embeddings path/to/dna.embeddings.npz \
  --rna_embeddings path/to/rna.embeddings.npz \
  --checkpoint path/to/model.pt \
  --output_dir path/to/output_dir \
  --feature_mode concat \
  --batch_size 256 \
  --device cuda:0
```

For trial runs, limit the size of `pairs_file`, for example by taking only the first N records:

```bash
python -m dnarna.models.pair.predict.infer \
  --pairs_file path/to/pairs.csv \
  --dna_embeddings path/to/dna.embeddings.npz \
  --rna_embeddings path/to/rna.embeddings.npz \
  --checkpoint path/to/model.pt \
  --output_dir path/to/output_dir \
  --feature_mode concat \
  --max_pairs 20000
```

To score all DNA x RNA combinations, use `--all_pairs`.
By default, this writes the raw `pair_predictions.csv`, which can be very large for large jobs. During trial runs, use `--max_dna/--max_rna` to limit scale first. If CPU resources are sufficient, use `--num_workers` to construct features in parallel:

```bash
python -m dnarna.models.pair.predict.infer \
  --all_pairs \
  --dna_embeddings path/to/dna.embeddings.npz \
  --rna_embeddings path/to/rna.embeddings.npz \
  --dna_seq_file path/to/dna.windowed.csv \
  --rna_seq_file path/to/rna.windowed.csv \
  --checkpoint path/to/model.pt \
  --output_dir path/to/output_dir \
  --feature_mode concat \
  --dna_block_size 128 \
  --rna_block_size 128 \
  --max_dna 1000 \
  --max_rna 500 \
  --num_workers 8 \
  --batch_size 256 \
  --device cuda:0
```

If the job is very large and only the aggregated `pair_predictions.summary.csv` is needed, use `--summary_only` (alias: `--skip_raw_output`):

```bash
python -m dnarna.models.pair.predict.infer \
  --all_pairs \
  --summary_only \
  --dna_embeddings path/to/dna.embeddings.npz \
  --rna_embeddings path/to/rna.embeddings.npz \
  --dna_seq_file path/to/dna.windowed.csv \
  --rna_seq_file path/to/rna.windowed.csv \
  --checkpoint path/to/model.pt \
  --output_dir path/to/output_dir \
  --feature_mode concat \
  --dna_block_size 128 \
  --rna_block_size 128 \
  --batch_size 256 \
  --device cuda:0
```

In this mode:

- The raw `pair_predictions.csv` is not written.
- Only the aggregated `pair_predictions.summary.csv` is written.
- Summary rows are appended in streaming parent-complete blocks, so the script does not need to keep every original DNA-RNA pair summary in memory until the end.
- `--dna_block_size` / `--rna_block_size` also control the window scale of each streaming summary block; smaller values reduce memory, while larger values usually improve throughput.
- Currently only `--all_pairs` is supported.
- Both `--dna_seq_file` and `--rna_seq_file` are required.
- Because results are appended block by block, `summary_only` output is not globally sorted by `combined_score`; if global sorting is needed, sort the result file afterward.
