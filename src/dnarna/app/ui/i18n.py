from __future__ import annotations

from typing import Any

import streamlit as st

LANG_CHOICES = {"中文": "zh", "English": "en"}


def select_language(*, default: str = "en") -> str:
    """
    Render a compact language selector and return selected language code.

    Args:
        default: "en" or "zh"
    """
    if default not in {"en", "zh"}:
        raise ValueError("default must be 'en' or 'zh'")

    labels = list(LANG_CHOICES.keys())
    default_label = "English" if default == "en" else "中文"
    default_idx = labels.index(default_label)
    label = st.selectbox(
        "Language / 语言",
        options=labels,
        index=default_idx,
        label_visibility="collapsed",
    )
    return LANG_CHOICES[label]


def build_i18n() -> dict[str, dict[str, str]]:
    return {
        "zh": {
            "title": "AI 赋能的 DNA/RNA 配对预测",
            "subtitle": "粘贴/指定序列文件，(可选) 先做 top-K 预筛选，再用配对模型进行 AI 预测。",
            "overview_title": "输入要求 & 流程概览",
            "overview_body": (
                "- 支持 FASTA 或 `id,seq` CSV（文件路径或粘贴内容），自动标准化大小写与 `U→T`。\n"
                "- **两种运行模式**  \n"
                "  1) 未设置 top-K：对全部 windowed DNA×RNA 组合进行配对模型预测。  \n"
                "  2) 设置 top-K：先用 DNABERT-2 / RNA-FM checkpoint 逐条评分，取各自概率最高的前 K 条，再进行配对预测。  \n"
                "- 结果与中间文件会保存到 `output_dir`（`inputs/` / `processed/` / `predictions/` / `topk/` / `pair_predictions/`）。\n"
                "- References:\n"
                "  - DNABERT-2: Zhou et al., *ICLR*, 2024. DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genomes.\n"
                "  - RNA-FM: Shen et al., *Nature Methods*, 2024. Accurate RNA 3D structure prediction using a language model-based deep learning approach."
            ),
            "step_title": "步骤{n}：{title}",
            "step1": "输入序列",
            "step1_caption": "选择 DNA/RNA 的输入方式（二选一），然后提供对应内容。",
            "mode_label_dna": "DNA 输入方式",
            "mode_label_rna": "RNA 输入方式",
            "mode_path": "指定文件路径",
            "mode_paste": "粘贴内容",
            "mode_help": "二选一：从本地文件读取，或直接粘贴内容。",
            "dna_path": "DNA 文件路径（FASTA 或 id,seq CSV）",
            "rna_path": "RNA 文件路径（FASTA 或 id,seq CSV）",
            "dna_path_ph": "local/data/dna.csv",
            "rna_path_ph": "local/data/rna.csv",
            "path_help": "支持 .fa/.fasta 或带表头的 CSV（至少包含 id,seq 两列）。",
            "dna_paste": "DNA：粘贴 FASTA 或 id,seq CSV",
            "rna_paste": "RNA：粘贴 FASTA 或 id,seq CSV",
            "dna_paste_ph": ">dna1\nACGT...\n>dna2\nTTAG...\n",
            "rna_paste_ph": "id,seq\nrna1,ACGUU\nrna2,GGCAU\n",
            "dna_paste_help": "FASTA 以 > 开头；CSV 需包含 id,seq 表头；会自动 U→T、大写并校验碱基。",
            "rna_paste_help": "RNA 允许 U，会自动转为 T；其余规则同 DNA。",
            "dna_metric": "DNA 序列",
            "rna_metric": "RNA 序列",
            "seq_none": "未提供序列",
            "seq_summary": "{n} 条 | 长度范围 {min_len}-{max_len}",
            "window_size": "Window 长度",
            "window_size_help": (
                "用于滑窗拆分的最大窗口长度（最大 {max_len}）。超过该长度会按 window/stride 切分并生成新 id。"
                "示例：长度 2500、window=1000、stride=500 → 4 个片段：id_win_0(1-1000)、"
                "id_win_1(501-1500)、id_win_2(1001-2000)、id_win_3(1501-2500)。"
            ),
            "window_stride": "Window 步长",
            "window_stride_help": (
                "滑动步长（一般建议为 window 长度的 1/2）。与 window_size 一起决定切分重叠程度。"
            ),
            "window_skipped": "{which} 有 {n} 条序列包含非法字符，已跳过。",
            "dna": "DNA",
            "rna": "RNA",
            "step2": "模型预筛选（可选）",
            "step2_caption": "用 DNABERT-2 / RNA-FM 先做序列嵌入，再用单序列分类模型打分，取 top-K 以减少配对规模。",
            "topk_dna": "DNA top-K (DNABERT-2)",
            "topk_rna": "RNA top-K (RNA-FM)",
            "topk_dna_help": "0=不启用 DNABERT-2 预筛选；>0 表示按模型概率取最高的前 K 条 DNA 并导出。",
            "topk_rna_help": "0=不启用 RNA-FM 预筛选；>0 表示按模型概率取最高的前 K 条 RNA 并导出。",
            "dna_embed_batch_size": "DNA 编码 batch size",
            "dna_embed_batch_help": "DNABERT-2 嵌入的 batch 大小。",
            "rna_embed_batch_size": "RNA 编码 batch size",
            "rna_embed_batch_help": "RNA-FM 嵌入的 batch 大小。",
            "score_batch_size": "Top-K 评分 batch size",
            "score_batch_help": "用于单序列分类评分（top-K 预筛选）的 batch 大小。",
            "dna_device": "DNA 模型设备 (可空, 例如 cuda:0)",
            "dna_device_help": "DNABERT-2 嵌入/评分设备；留空=自动选择：有 CUDA 用 cuda，没有则用 CPU。",
            "rna_device": "RNA 模型设备 (可空, 例如 cuda:1)",
            "rna_device_help": "RNA-FM 嵌入/评分设备；留空=自动选择：有 CUDA 用 cuda，没有则用 CPU。",
            "dnabert_ckpt": "DNABERT-2 checkpoint 路径",
            "rnafm_ckpt": "RNA-FM checkpoint 路径",
            "dnabert_ckpt_ph": "models/dnabert2_classifier.pt",
            "rnafm_ckpt_ph": "models/rnafm_classifier.pt",
            "dnabert_ckpt_help": "启用 DNA top-K 时必填：默认指向 ./models 下的 checkpoint，仅在 top-K 启用时需要。",
            "rnafm_ckpt_help": "启用 RNA top-K 时必填：默认指向 ./models 下的 checkpoint，仅在 top-K 启用时需要。",
            "step3": "配对模型预测",
            "step3_caption": "设置配对模型的参数并进行 RNA×DNA 预测。",
            "pair_ckpt": "配对模型 checkpoint 路径",
            "pair_ckpt_ph": "path/to/pair_model.pt",
            "pair_ckpt_help": "必填：DNA-RNA 配对模型的 checkpoint。",
            "pair_feature_mode": "配对特征方式",
            "pair_feature_mode_help": "特征组合方式：concat / absdiff / mul / all。",
            "pair_threshold": "预测阈值",
            "pair_threshold_help": "概率 ≥ 阈值视为正例。",
            "pair_device": "配对模型设备 (可空, 例如 cuda:0)",
            "pair_device_help": "配对模型推理设备；留空=自动选择。",
            "pair_dna_block_size": "DNA block size",
            "pair_rna_block_size": "RNA block size",
            "pair_block_help": "all_pairs 模式下的分块大小，越小越省显存/内存但更慢。",
            "pair_batch_size": "配对推理 batch size",
            "pair_batch_help": "配对模型推理 batch 大小。",
            "pair_advanced": "高级参数",
            "pair_chunk_size": "特征构建 chunk size",
            "pair_chunk_help": "特征构建的分块大小。",
            "pair_num_workers": "特征构建进程数",
            "pair_num_workers_help": "CPU 并行构建特征时的进程数。",
            "pair_max_dna": "限制 DNA 数量 (max_dna)",
            "pair_max_rna": "限制 RNA 数量 (max_rna)",
            "pair_max_help": "仅在 all_pairs 时生效；0=不限制。",
            "step4": "设置输出目录",
            "step4_caption": "所有结果与中间文件都会写入该目录，便于复现与后续分析。",
            "output_dir": "output_dir（可编辑，默认已填推荐路径）",
            "output_help": "将把窗口化结果、预测分数、top-K 输出、配对预测等保存到该目录。",
            "run": "一键运行",
            "need_seqs": "请至少提供 1 条 DNA 和 1 条 RNA 序列。",
            "need_out": "请指定 output_dir。",
            "running": "任务运行中...",
            "failed": "运行失败：{msg}",
            "done": "任务完成！",
            "mode_line": "模式：{mode} | 最终 DNA={dna} 条，RNA={rna} 条",
            "mode_all": "不做 top-K",
            "mode_topk": "top-K 预筛选",
            "dna_scores": "**DNABERT-2 评分 (按概率降序)**",
            "rna_scores": "**RNA-FM 评分 (按概率降序)**",
            "pair_predictions": "**配对预测结果 (前 200 行预览)**",
            "pair_predictions_path": "配对预测输出：{path}",
            "pair_summary": "**配对综合汇总 (按原始 DNA/RNA 对聚合，前 200 行预览)**",
            "pair_summary_path": "配对综合汇总输出：{path}",
            "used_paths": "输出文件：DNA={dna} | RNA={rna} | Pairs={pairs}",
            "artifacts": "中间文件保存在 {out}（inputs/processed/predictions/topk/pair_predictions）。",
            "download_cfg": "下载配置快照 (JSON)",
            "status_title": "当前设置",
            "status_inputs": "输入",
            "status_prefilter": "预筛选",
            "status_pair": "配对模型",
            "status_output": "输出",
            "status_not_set": "未设置",
            "status_auto": "自动",
            "status_enabled": "已启用",
            "status_disabled": "未启用",
            "warning_no_cuda": "检测到无 CUDA：嵌入与配对预测会使用 CPU，可能非常慢。",
            "device_invalid": "{label} 不可用：{device}（请检查 CUDA 是否可用及设备编号）。",
            "need_pair_ckpt": "请提供配对模型 checkpoint 路径。",
            "pair_ckpt_missing": "找不到配对模型 checkpoint：{path}",
            "hint_run": "确认无误后点击右下角运行（任务执行中请保持 Streamlit 进程运行，关闭会中断任务）。",
        },
        "en": {
            "title": "AI-Powered DNA/RNA Pair Prediction",
            "subtitle": "Provide sequences (paste or file path). Optionally pre-filter with DNABERT-2 / RNA-FM (top-K), then run the pair model.",
            "overview_title": "Input & Workflow",
            "overview_body": (
                "- Supports FASTA or `id,seq` CSV (file path or pasted content). Normalizes case and converts `U→T`.\n"
                "- **Two modes**  \n"
                "  1) top-K = 0: run pair prediction on all windowed DNA×RNA pairs.  \n"
                "  2) top-K > 0: score DNA/RNA with DNABERT-2 / RNA-FM, keep top K, then run pair prediction.\n"
                "- Writes outputs and artifacts to `output_dir` (`inputs/` / `processed/` / `predictions/` / `topk/` / `pair_predictions/`).\n"
                "- References:\n"
                "  - DNABERT-2: Zhou et al., *ICLR*, 2024. DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genomes.\n"
                "  - RNA-FM: Shen et al., *Nature Methods*, 2024. Accurate RNA 3D structure prediction using a language model-based deep learning approach."
            ),
            "step_title": "Step {n}: {title}",
            "step1": "Enter sequences",
            "step1_caption": "Choose one input method for DNA/RNA, then provide the corresponding content.",
            "mode_label_dna": "DNA input",
            "mode_label_rna": "RNA input",
            "mode_path": "File path",
            "mode_paste": "Paste content",
            "mode_help": "Choose exactly one: read from a local file, or paste content directly.",
            "dna_path": "DNA file path (FASTA or id,seq CSV)",
            "rna_path": "RNA file path (FASTA or id,seq CSV)",
            "dna_path_ph": "local/data/dna.csv",
            "rna_path_ph": "local/data/rna.csv",
            "path_help": "Supports .fa/.fasta or CSV with header (must include id,seq).",
            "dna_paste": "DNA: paste FASTA or id,seq CSV",
            "rna_paste": "RNA: paste FASTA or id,seq CSV",
            "dna_paste_ph": ">dna1\nACGT...\n>dna2\nTTAG...\n",
            "rna_paste_ph": "id,seq\nrna1,ACGUU\nrna2,GGCAU\n",
            "dna_paste_help": "FASTA starts with >. CSV must include id,seq header. Auto normalizes U→T, uppercases, validates bases.",
            "rna_paste_help": "RNA allows U and will be converted to T; other rules are the same as DNA.",
            "dna_metric": "DNA sequences",
            "rna_metric": "RNA sequences",
            "seq_none": "No sequences",
            "seq_summary": "{n} seqs | length {min_len}-{max_len}",
            "window_size": "Window size",
            "window_size_help": (
                "Max window length for splitting (max {max_len}). Sequences longer than this are "
                "split by window/stride and get new ids. Example: len=2500, window=1000, stride=500 "
                "→ 4 windows: id_win_0(1-1000), id_win_1(501-1500), id_win_2(1001-2000), "
                "id_win_3(1501-2500)."
            ),
            "window_stride": "Window stride",
            "window_stride_help": (
                "Stride between windows (often half of window size). Controls overlap with window_size."
            ),
            "window_skipped": "{which} has {n} sequences skipped due to invalid bases.",
            "dna": "DNA",
            "rna": "RNA",
            "step2": "Optional pre-filter",
            "step2_caption": "Embed sequences with DNABERT-2 / RNA-FM, then score with the per-sequence classifier to keep top-K.",
            "topk_dna": "DNA top-K (DNABERT-2)",
            "topk_rna": "RNA top-K (RNA-FM)",
            "topk_dna_help": "0=disable DNABERT-2 pre-filter. >0 keeps top K DNA by model probability.",
            "topk_rna_help": "0=disable RNA-FM pre-filter. >0 keeps top K RNA by model probability.",
            "dna_embed_batch_size": "DNA embedding batch size",
            "dna_embed_batch_help": "Batch size for DNABERT-2 embedding.",
            "rna_embed_batch_size": "RNA embedding batch size",
            "rna_embed_batch_help": "Batch size for RNA-FM embedding.",
            "score_batch_size": "Top-K scoring batch size",
            "score_batch_help": "Batch size for per-sequence classifier inference (top-K only).",
            "dna_device": "DNA model device (optional, e.g. cuda:0)",
            "dna_device_help": "DNABERT-2 embedding/scoring device. Empty=auto: CUDA if available, otherwise CPU.",
            "rna_device": "RNA model device (optional, e.g. cuda:1)",
            "rna_device_help": "RNA-FM embedding/scoring device. Empty=auto: CUDA if available, otherwise CPU.",
            "dnabert_ckpt": "DNABERT-2 checkpoint path",
            "rnafm_ckpt": "RNA-FM checkpoint path",
            "dnabert_ckpt_ph": "models/dnabert2_classifier.pt",
            "rnafm_ckpt_ph": "models/rnafm_classifier.pt",
            "dnabert_ckpt_help": "Required when DNA top-K > 0; only needed when top-K is enabled. Defaults to checkpoints under ./models.",
            "rnafm_ckpt_help": "Required when RNA top-K > 0; only needed when top-K is enabled. Defaults to checkpoints under ./models.",
            "step3": "Pair prediction",
            "step3_caption": "Configure the pair model and run RNA×DNA prediction.",
            "pair_ckpt": "Pair model checkpoint path",
            "pair_ckpt_ph": "path/to/pair_model.pt",
            "pair_ckpt_help": "Required: checkpoint for the DNA-RNA pair model.",
            "pair_feature_mode": "Pair feature mode",
            "pair_feature_mode_help": "Feature combination: concat / absdiff / mul / all.",
            "pair_threshold": "Prediction threshold",
            "pair_threshold_help": "Probability ≥ threshold is marked positive.",
            "pair_device": "Pair model device (optional, e.g. cuda:0)",
            "pair_device_help": "Device for pair inference; empty=auto.",
            "pair_dna_block_size": "DNA block size",
            "pair_rna_block_size": "RNA block size",
            "pair_block_help": "Block size for all_pairs generation; smaller saves memory but is slower.",
            "pair_batch_size": "Pair inference batch size",
            "pair_batch_help": "Batch size for pair model inference.",
            "pair_advanced": "Advanced settings",
            "pair_chunk_size": "Feature chunk size",
            "pair_chunk_help": "Chunk size for feature construction.",
            "pair_num_workers": "Feature workers",
            "pair_num_workers_help": "CPU workers for feature construction.",
            "pair_max_dna": "Limit DNA count (max_dna)",
            "pair_max_rna": "Limit RNA count (max_rna)",
            "pair_max_help": "Only for all_pairs; 0 means no limit.",
            "step4": "Set output directory",
            "step4_caption": "All outputs and intermediate artifacts will be written here for reproducibility.",
            "output_dir": "output_dir (editable; recommended default provided)",
            "output_help": "Writes windowed outputs, prediction scores, top-K exports, and pair predictions into this directory.",
            "run": "Run",
            "need_seqs": "Please provide at least 1 DNA and 1 RNA sequence.",
            "need_out": "Please specify output_dir.",
            "running": "Running...",
            "failed": "Run failed: {msg}",
            "done": "Done!",
            "mode_line": "Mode: {mode} | final DNA={dna}, RNA={rna}",
            "mode_all": "no top-K",
            "mode_topk": "top-K filtered",
            "dna_scores": "**DNABERT-2 scores (sorted by prob)**",
            "rna_scores": "**RNA-FM scores (sorted by prob)**",
            "pair_predictions": "**Pair predictions (preview: first 200 rows)**",
            "pair_predictions_path": "Pair predictions output: {path}",
            "pair_summary": "**Pair summary (aggregated by original DNA/RNA pair, preview: first 200 rows)**",
            "pair_summary_path": "Pair summary output: {path}",
            "used_paths": "Outputs: DNA={dna} | RNA={rna} | Pairs={pairs}",
            "artifacts": "Artifacts saved to {out} (`inputs/` / `processed/` / `predictions/` / `topk/` / `pair_predictions/`).",
            "download_cfg": "Download config (JSON)",
            "status_title": "Current setup",
            "status_inputs": "Inputs",
            "status_prefilter": "Pre-filter",
            "status_pair": "Pair model",
            "status_output": "Output",
            "status_not_set": "Not set",
            "status_auto": "auto",
            "status_enabled": "Enabled",
            "status_disabled": "Disabled",
            "warning_no_cuda": "No CUDA detected: embeddings and pair prediction will run on CPU and may be very slow.",
            "device_invalid": "{label} is not available: {device} (check CUDA availability and device index).",
            "need_pair_ckpt": "Please provide the pair model checkpoint path.",
            "pair_ckpt_missing": "Pair model checkpoint not found: {path}",
            "hint_run": "When ready, click Run at the bottom-right (keep the Streamlit process alive; closing it will stop the job).",
        },
    }


def make_translator(lang: str, i18n: dict[str, dict[str, str]]):
    def _t(key: str, **kwargs: Any) -> str:
        text = i18n.get(lang, {}).get(key) or i18n["zh"].get(key, key)
        return text.format(**kwargs) if kwargs else text

    return _t
