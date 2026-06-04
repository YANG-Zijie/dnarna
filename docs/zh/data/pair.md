# DNA-RNA对数据

该数据集由3个文件构成。

## 文件结构

### 文件1：DNA-RNA对数据

包含3列：

- `pair_id`：每对DNA-RNA序列的唯一标识符
- `dna_id`：对应DNA序列的ID
- `rna_id`：对应RNA序列的ID

### 文件2：DNA序列数据

包含2列：

- `dna_id`：DNA序列的ID
- `dna_seq`：DNA序列字符串

### 文件3：RNA序列数据

包含2列：

- `rna_id`：RNA序列的ID
- `rna_seq`：RNA序列字符串

## 数据清洗

加载数据时会进行以下清洗操作：

1. 对文件2/3，剔除空序列记录，并在日志中记录被移除的ID。
2. 对文件2/3，统一转为大写，并将RNA序列中的`U`替换为`T`。
3. 对文件2/3，剔除包含非`A/C/G/T`字符的序列，并在日志中记录被移除的ID。
4. 对文件2/3，移除重复ID（保留首次出现），并在日志中记录被移除的重复ID。
5. 对文件2/3，移除序列相同但ID不同的记录（保留首次出现），并在日志中记录被移除的ID。
   对应的 `dna_id` / `rna_id` 会在 pair 文件中自动映射到保留的ID，避免因序列去重导致交叉配对被误删。
6. 对文件1，移除引用不存在DNA或RNA ID的记录，并在日志中记录对应的`pair_id`。
7. 对文件1去重（相同 `dna_id` + `rna_id` 仅保留一次），并在日志中记录重复的`pair_id`。

### 数据清洗脚本

清洗脚本位于 `src/dnarna/data/pair/clean.py`，可通过命令行处理三份原始数据并输出清洗后的结果：

```bash
python -m dnarna.data.pair.clean \
    --pair_file path/to/pairs.csv \
    --dna_file path/to/dna.csv \
    --rna_file path/to/rna.csv \
    --output_dir path/to/output_dir \
    --output_format csv
```

参数说明：

- `--pair_file`：文件1（pair）路径，支持 `.csv` / `.parquet`
- `--dna_file`：文件2（DNA）路径，支持 `.csv` / `.parquet`
- `--rna_file`：文件3（RNA）路径，支持 `.csv` / `.parquet`
- `--output_dir`：输出目录
- `--output_format`：输出格式（`csv` 或 `parquet`）
- `--pair_output` / `--dna_output` / `--rna_output`：可选，指定输出文件名

输出文件默认以输入文件名为基准生成（扩展名随 `--output_format` 自动调整）。日志与报告文件名与 pair 输出文件一致：

- `<pair_output>.log`：详细清洗日志
- `<pair_output>.report.json`：统计信息与被移除条目摘要

### 清洗后输出

清洗完成后仍输出3个主数据文件（文件1/2/3），同时额外生成以下记录文件：

- `*.json`：记录清洗过程的统计信息与被移除条目摘要。
- `*.log`：记录清洗过程的详细日志与被移除ID列表。

## 超长序列滑窗（window）

如果 DNA/RNA 序列可能超长，建议先对 DNA/RNA 分别做滑窗切分，再用 `parent_id`
回溯原始配对并生成新的唯一 `pair_id`。推荐默认参数：

- `window_size=1000`
- `stride=500`

滑窗后每条序列会新增以下信息：

- `parent_id`：原始序列 ID
- `window_index`：窗口序号（未切分为 0，切分后从 1 开始）
- `window_start` / `window_end`：窗口在原序列的区间

处理顺序建议：

1. 先执行清洗（去重/校验/ID 映射）。
2. 再执行滑窗（生成窗口序列与窗口元数据）。
3. 用 `parent_id` 展开 windowed pairs 并生成新 `pair_id`。

滑窗模块：`src/dnarna/data/pair/window.py`。默认输出：
`*.windowed.csv`（或 `.parquet`）。

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

如果 DNA/RNA 需要不同窗口参数，可使用：

- `--dna_window_size` / `--dna_stride`
- `--rna_window_size` / `--rna_stride`

未指定时默认沿用 `window_size/stride`（stride 缺省为 `window_size//2`）。

示例（`window_size=1000`, `stride=500`，DNA/RNA 各 1 条超长序列）：

- 输入：`pair_id=P1`, `dna_id=D1(len=2500)`, `rna_id=R1(len=1800)`
- DNA 窗口 id：`D1_win_1`, `D1_win_2`, `D1_win_3`, `D1_win_4`
- RNA 窗口 id：`R1_win_1`, `R1_win_2`, `R1_win_3`
- 新 `pair_id` 规则：`P1__d{dna_window_index}__r{rna_window_index}`
  - 例：`P1__d1__r1`, `P1__d1__r2`, …, `P1__d4__r3`
- 未切分的序列窗口序号为 0，pair_id 中会出现 `d0` / `r0`。
