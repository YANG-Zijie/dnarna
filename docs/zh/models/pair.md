# DNA-RNA配对预测（基于嵌入）

本模型用于判断给定的 DNA-RNA 对是否可能结合。核心做法是把 DNA/RNA 的嵌入拼成
“pair 特征”，再用一个浅层 MLP 进行二分类训练。

## 数据准备

1. DNA/RNA 序列分别完成嵌入（DNABERT2 / RNAFM），得到 `.npz`：
   - 必须包含 `ids` 与 `embeddings` 两个键
2. Pair 文件至少包含：
   - `pair_id`（可选，缺失时自动生成）
   - `dna_id` / `rna_id`
   - `label`（可选；若没有将视为全正例）
   - `split`（可选，指定 train/val；仅在不采样负例时生效）
3. `dna_id` / `rna_id` 必须能在对应 embeddings 的 `ids` 中找到，否则该 pair 会被丢弃。

若使用 windowed 数据，请确保 `pairs` 与 `embeddings` 都是 windowed 版本，ID 一致。

## 负例构造方式

两种情况：

- **已有负例（label=0/1）**：直接使用，不再采样。
- **只有正例**：将全部 pair 视为正例，按 `negative_ratio` 随机采样负例，
  负例来自 DNA IDs 与 RNA IDs 的随机组合，且不与正例重叠。

默认 `negative_ratio=1.0`（正负 1:1）。注意：这种“随机负例”可能包含真实正例，
属于弱负例，需在解释结果时谨慎。

## Pair 特征构造

使用 DNA/RNA embeddings 构造 pair 特征向量：

- `concat`（默认）：`[dna; rna]`，维度 = D_dna + D_rna
- `absdiff`：`|dna - rna|`（要求 D_dna == D_rna）
- `mul`：`dna * rna`（要求 D_dna == D_rna）
- `all`：`[dna; rna; |dna-rna|; dna*rna]`（要求 D_dna == D_rna）

## 训练流程

1. 生成 pair 特征并保存为 `<pairs>.pair_embeddings.npz`
2. 生成训练元数据 `<pairs>.pair_metadata.csv`（包含 label/source 等）
3. 使用 MLP（`hidden_dims` 必填）进行二分类训练  
4. 训练/验证划分：
   - 有 `split` 且未采样负例：按 split 划分
   - 否则：按 `val_fraction` 随机划分（默认 0.1，即 90% train / 10% val）

训练会输出：

- `model.pt` / `best_model.pt`
- `metrics.csv` / `metrics.json` / `metrics.pdf`
- 若有验证集：`val_predictions.csv`、`roc_pr.pdf`
- 过程报告：`*.pair_report.json`

## 复用已生成的 pair 特征

如果你已经生成过 `*.pair_embeddings.npz` 和 `*.pair_metadata.csv`，可以跳过特征生成与负例采样：

```bash
python -m dnarna.models.pair.predict.train \
  --pairs_file path/to/pairs.csv \
  --pair_embeddings_input path/to/pairs.pair_embeddings.npz \
  --metadata_input path/to/pairs.pair_metadata.csv \
  --output_dir path/to/output \
  --hidden_dims 512,256
```

注意：此模式会忽略 `--negative_ratio`/`--max_pairs`，并要求 metadata 中包含 `label` 列。

## 训练示例

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

## 推理

推理时需要与训练一致的 `feature_mode`，否则会提示不一致。
使用 `python -m dnarna.models.pair.predict.infer` 即可输出每个 pair 的 `prob/pred`。
使用 `--output_dir` 会在同目录写入 `*.log`、`*.meta.json`，以及在可识别 window 元数据时自动追加一个 `*.summary.csv` 聚合报告。

### Window 场景下的综合得分

当一条长 DNA / RNA 被切成多个 window 后，原始输出会给出每个 `window pair` 的概率。
现在会额外生成一个聚合 summary，把同一个原始 DNA-RNA 对的所有 window-pair 汇总成一条记录，常用列包括：

- `pair_predictions.csv`：1 行 = 1 个 `window-pair`
- `pair_predictions.summary.csv`：1 行 = 1 个原始 DNA-RNA 对
- `combined_score`：综合得分，表示“至少有一个 window-pair 命中”的整体强度（基于 noisy-or 聚合）
- `prob_max`：最强单个 window-pair 的分数
- `prob_mean`：所有 window-pair 的平均分；这是按 pair 数量归一化后的均值，不直接体现“高分窗口有多少个”
- `positive_window_pair_count`：概率大于等于当前分类阈值（`threshold`）的 window-pair 数量
- `positive_window_pair_fraction`：概率大于等于当前分类阈值（`threshold`）的 window-pair 比例，即 `positive_window_pair_count / window_pair_count`
- `window_pair_count` / `dna_window_count` / `rna_window_count`
- `best_window_pair_id` 及对应 window index/start/end

如果你更关注“是否存在局部强结合位点”，优先看 `combined_score` 和 `prob_max`；
如果更关注“整体上是否普遍偏高”，再结合 `prob_mean` 和 `positive_window_pair_fraction` 一起看。

可以把它理解成两层输出：

| 文件 | 1 行代表什么 | 典型用途 |
| --- | --- | --- |
| `pair_predictions.csv` | 1 个 `window-pair` | 看每个局部窗口对的原始分数 |
| `pair_predictions.summary.csv` | 1 个原始 DNA-RNA 对 | 看整对长序列的综合情况 |

例如，假设：

- 1 条 DNA 被切成 5 个 window
- 1 条 RNA 被切成 3 个 window

那么原始预测会有 `5 x 3 = 15` 条 `window-pair` 记录，但 summary 里只会有 1 条原始 DNA-RNA 对的汇总记录。

原始 `pair_predictions.csv` 可能类似于：

| pair_id | dna_id | rna_id | prob | pred |
| --- | --- | --- | ---: | ---: |
| `dnaA_win_0__rnaB_win_0` | `dnaA_win_0` | `rnaB_win_0` | 0.82 | 1 |
| `dnaA_win_0__rnaB_win_1` | `dnaA_win_0` | `rnaB_win_1` | 0.11 | 0 |
| `dnaA_win_1__rnaB_win_0` | `dnaA_win_1` | `rnaB_win_0` | 0.64 | 1 |
| `...` | `...` | `...` | `...` | `...` |

对应的 `pair_predictions.summary.csv` 可能类似于：

| pair_group_id | dna_parent_id | rna_parent_id | dna_window_count | rna_window_count | window_pair_count | prob_max | prob_mean | combined_score | positive_window_pair_count | best_window_pair_id |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `dnaA__rnaB` | `dnaA` | `rnaB` | 5 | 3 | 15 | 0.91 | 0.37 | 0.98 | 4 | `dnaA_win_2__rnaB_win_1` |

也就是说，summary 不是把原始 `window-pair` 明细丢掉，而是在保留明细文件的同时，再额外给你一张“按原始 DNA-RNA 对汇总后的表”。

### `combined_score` 到底是什么

设某个原始 DNA-RNA 对在 window 后一共得到 `k` 个 `window-pair`，对应概率为：

`p1, p2, ..., pk`

那么当前的综合分定义为：

`combined_score = 1 - (1-p1)(1-p2)...(1-pk)`

也就是：

- 先把每个 `window-pair` “不命中”的概率写成 `1-pi`
- 再把“所有 window-pair 都不命中”的概率相乘
- 最后用 `1 - 上面这个值`
- 得到“至少有一个 window-pair 命中”的整体分数

直观上，它表达的是：

- 只要有一个局部窗口很强，`combined_score` 就会高
- 如果没有特别强的单个窗口，但有很多中等偏高的窗口，`combined_score` 也会逐步升高
- 所以它不是“平均分”，而是“存在至少一个有效局部结合区域”的累积证据

几个简单例子：

- 如果只有一个窗口，`p=[0.8]`
  - `combined_score = 0.8`
- 如果有两个窗口，`p=[0.8, 0.1]`
  - `combined_score = 1 - (0.2 * 0.9) = 0.82`
  - 比 `0.8` 略高，因为第二个窗口也提供了一点额外证据
- 如果有三个中等窗口，`p=[0.4, 0.4, 0.4]`
  - `combined_score = 1 - (0.6^3) = 0.784`
  - 虽然单个窗口都不算特别高，但“多个地方都还可以”会把整体分数抬上来

所以它和别的指标的区别是：

- `prob_max`：只看最强的那个窗口
- `prob_mean`：看整体平均水平
- `combined_score`：看“至少有一个局部区域可能结合”的综合证据

### 什么是 noisy-or

`noisy-or` 是概率图模型里一个很常见的聚合思路。

它背后的想法是：

- 把每个 `window-pair` 看作一个“可能导致整体 pair 为阳性”的局部原因
- 整体事件相当于这些局部原因的 OR：只要任意一个成立，整体就可能成立
- 但每个局部原因不是 100% 可靠的，所以叫 `noisy`，意思是“带噪声的 OR”，不是硬性的逻辑 OR

在这里它不是一个新的模型输出，而是对已有 `window-pair` 概率做的后处理聚合。

### 使用时要注意什么

- `combined_score` 通常会随着 `window-pair` 数量增多而更容易升高
- 因此它更适合回答“这对长 DNA/RNA 中是否至少存在局部强结合位点”
- 如果你要比较不同长度、不同 window 数量的样本，最好同时看：
  - `prob_max`
  - `prob_mean`
  - `positive_window_pair_fraction`

一个简单建议是：

- 想找“有没有明显热点”，看 `combined_score` 和 `prob_max`
- 想看“是不是很多地方都偏高”，看 `prob_mean` 和 `positive_window_pair_fraction`
- 想看“高分局部区域到底有多少个”，看 `positive_window_pair_count`

### 如何理解 `prob_mean` 与“高分窗口数量”

- `prob_mean` 是简单平均值，因此与 `window_pair_count` 不成正比。换句话说，pair 数量变多并不会自动把 `prob_mean` 拉高。
- 但 `prob_mean` 也不会告诉你“到底有多少个高分 window-pair”，它只告诉你整体平均水平。
- 如果你想判断“这对给定的 DNA/RNA 是否在整体上存在更多可能高结合的局部区域”，`positive_window_pair_count` 和 `positive_window_pair_fraction` 更直接。
- 其中：
  - `positive_window_pair_count` 更像“绝对数量”，适合看高分局部位点有多少个。
  - `positive_window_pair_fraction` 更像“密度/占比”，在不同长度、不同 window 数量的样本之间通常更可比。

一个实用的读法是：

- 看 `combined_score` / `prob_max`：判断是否存在明显强结合窗口
- 看 `positive_window_pair_count`：判断强结合窗口的绝对数量
- 看 `positive_window_pair_fraction` / `prob_mean`：判断整体上是不是普遍偏高

### 已有预测结果时，单独生成 summary

如果你已经有了完整的 `pair_predictions.csv`，但不想重新跑一遍耗时的 pair 推理，可以单独运行 summary 脚本。

对于 `all_pairs` 场景，推荐提供 windowed DNA/RNA 文件来恢复 parent/window 元数据：

```bash
python -m dnarna.models.pair.predict.summarize \
  --predictions path/to/pair_predictions.csv \
  --dna_seq_file path/to/dna.windowed.csv \
  --rna_seq_file path/to/rna.windowed.csv \
  --chunksize 200000
```

对于基于 windowed pair 文件推理的场景，也可以直接提供原始的 `pairs.windowed.csv`：

```bash
python -m dnarna.models.pair.predict.summarize \
  --predictions path/to/pair_predictions.csv \
  --pairs_file path/to/pairs.windowed.csv
```

如果你还想一次性查看多个阈值下的 `positive_window_pair_count` / `positive_window_pair_fraction`，可以额外加 `--thresholds`：

```bash
python -m dnarna.models.pair.predict.summarize \
  --predictions path/to/pair_predictions.csv \
  --dna_seq_file path/to/dna.windowed.csv \
  --rna_seq_file path/to/rna.windowed.csv \
  --thresholds 0.5,0.6,0.7,0.8,0.9
```

这会在主 summary 之外，再额外生成一个 long-format 文件：

- `pair_predictions.summary.by_threshold.csv`

它的每一行表示：

- 1 个原始 DNA-RNA 对
- 在 1 个给定阈值下的统计结果

例如：

| pair_group_id | threshold | window_pair_count | positive_window_pair_count | positive_window_pair_fraction |
| --- | ---: | ---: | ---: | ---: |
| `dnaA__rnaB` | 0.5 | 15 | 4 | 0.267 |
| `dnaA__rnaB` | 0.6 | 15 | 3 | 0.200 |
| `dnaA__rnaB` | 0.7 | 15 | 2 | 0.133 |

补充说明：

- 如果 `pair_predictions.csv` 本身已经包含 `pair_parent_id` / `dna_parent_id` / `rna_parent_id` 等列，则直接使用这些列生成 summary。
- 如果没有这些列，但提供了 `--pairs_file` 或 `--dna_seq_file/--rna_seq_file`，则优先使用这些显式元数据恢复 parent 信息。
- 如果也没有额外元数据，程序会自动把形如 `xxx_win_1`、`xxx_win_2` 的 ID 解析回 `xxx`，并据此生成 summary。
- 如果 ID 本身不带 window 后缀，则会把原始 `dna_id/rna_id` 自身视作 parent。
- 默认情况下，如果预测文件里已经有 `pred` 列，则 `positive_window_pair_count` / `positive_window_pair_fraction` 直接按 `pred` 统计；对 `infer` 生成的结果，这等价于按推理时的 `threshold` 判断，默认是 `prob >= 0.5`。
- 使用 `--ignore_pred_col` 可以忽略现有 `pred` 列，按新的 `--threshold` 重新统计 `positive_window_pair_count`。
- `--thresholds` 生成的 `by_threshold` 报告总是按原始 `prob >= threshold` 重新计算，不依赖已有的 `pred` 列。
- `by_threshold` 只保留真正随阈值变化的统计列；像 `combined_score` 这类对同一个原始 DNA-RNA 对固定不变的列，仍然只保留在主 `summary.csv` 里。
- `summarize` 现在默认按 `--chunksize` 分块流式读取 `pair_predictions.csv`，因此可以处理非常大的 CSV/TSV 文件，而不需要整表载入内存。
- 对超大任务，优先推荐 `--dna_seq_file/--rna_seq_file` 这条路径；`--pairs_file` 仍可能因为自身文件过大而占用较多内存。
- 如果你确定数据量较小，也可以设置 `--chunksize 0`，退回到一次性整表读取模式。

示例：

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

试跑时可以限制 `pairs_file` 的规模，例如只取前 N 条：

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

如果需要对全部 DNA×RNA 组合打分，可使用 `--all_pairs`。
默认模式下会写出原始 `pair_predictions.csv`，因此在超大任务里文件体积可能非常大。试跑时建议配合 `--max_dna/--max_rna` 先限制规模，CPU 资源充足时可用 `--num_workers` 并行构造特征：

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

如果任务规模很大，而你最终只关心聚合后的 `pair_predictions.summary.csv`，推荐直接加 `--summary_only`（同义参数：`--skip_raw_output`）：

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

这个模式下：

- 不再写原始 `pair_predictions.csv`
- 只写聚合后的 `pair_predictions.summary.csv`
- summary 会按 parent-complete block 流式追加写入，不会把所有原始 DNA-RNA 对的 summary 状态一直留到最后
- `--dna_block_size` / `--rna_block_size` 同时控制每个流式 summary block 的窗口规模；值越小越省内存，值越大吞吐通常越好
- 目前只支持 `--all_pairs`
- 需要同时提供 `--dna_seq_file` 和 `--rna_seq_file`
- 因为是流式追加写入，`summary_only` 输出默认不是按 `combined_score` 全局排序；如果需要全局排序，可以在结果生成后再单独排序
