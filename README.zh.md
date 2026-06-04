# DnaRna

DnaRna 是一个用于预测 DNA-RNA 潜在互作的开源模型与命令行流程。它支持输入序列标准化、长序列滑窗、DNABERT-2/RNA-FM 嵌入提取，并使用训练好的配对分类器对 DNA x RNA 组合进行打分。

## 仓库内容

- `src/dnarna/`：用于数据处理、嵌入、配对预测、绘图和任务编排的 Python 包
- `assets/checkpoints/pair_model/model.pt`：本次发布包含的训练后配对模型 checkpoint
- `scripts/pipeline-template.sh`：端到端预测的命令行 pipeline 模板
- `docs/`：由 VitePress 管理的中英文文档站点
- `tests/`：覆盖数据处理和预测工具的回归测试

## 快速开始

```bash
uv pip install .
python -m dnarna.models.pair.predict.infer --help
```

如需端到端运行，复制并修改 [scripts/pipeline-template.sh](scripts/pipeline-template.sh) 中的环境变量，然后执行：

```bash
bash scripts/pipeline-template.sh
```

该模板需要 DNA/RNA 输入 CSV 文件，并会输出滑窗序列、embeddings、原始配对预测和聚合后的 pair summary。

## 内置模型

本项目使用的训练后配对模型位于：

```text
assets/checkpoints/pair_model/model.pt
```

仅从可信来源加载模型 checkpoint 和 Hugging Face 模型代码。DNABERT-2 使用 Hugging Face remote model code，本地 PyTorch checkpoint 会在推理时被反序列化。

## 文档

文档由 VitePress 管理。可使用以下命令本地预览或构建：

```bash
pnpm install
pnpm docs:dev
pnpm docs:build
```

构建后的静态站点可部署到 GitHub Pages。

## Hugging Face 镜像

如果无法访问原始 Hugging Face 站点，可在运行嵌入或预测命令前设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=~/.cache/huggingface
```
