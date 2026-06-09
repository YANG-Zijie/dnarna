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

## 可选图形界面

DnaRna 也包含一个可选的 Streamlit 界面，用于交互式提交预测任务：

Streamlit 图形界面是同一套命令行预测流程的交互式封装，适合小到中等规模任务试跑。对于超大规模任务、服务器后台运行或需要精细控制参数的场景，建议直接使用命令行 pipeline。

```bash
uv pip install .
uv run streamlit run src/dnarna/app/streamlit_app.py
```

默认本地访问地址为：

```text
http://localhost:8501
```

任务运行期间需要保持 Streamlit 进程存活。停止该进程会中断正在运行的任务。
