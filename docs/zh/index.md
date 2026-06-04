# DnaRna 文档

DnaRna 是一个用于预测 DNA-RNA 潜在互作的开源模型与命令行流程。

## 包含内容

- 序列输入与清洗工具
- DNA/RNA 候选长序列滑窗
- DNABERT-2 与 RNA-FM 嵌入流程
- 配对预测与窗口级结果聚合
- 面向公开发布的训练后配对模型 checkpoint

## 文档内容

- 数据格式：序列输入与 DNA-RNA pair 输入
- 配对模型：训练、推理与聚合得分
- 论文方法：模型架构、训练策略与方法局限

## 本地使用

```bash
uv pip install .
python -m dnarna.models.pair.predict.infer --help
```

端到端模板可修改并运行：

```bash
bash scripts/pipeline-template.sh
```
