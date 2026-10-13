# 运行

## 查看和重算结果

```bash
python3 analysis/reproduce_table1.py
python3 analysis/verify_snapshot.py
python3 analysis/export_results.py --check
```

只需 Python 3.9+ 标准库。主表重算输出放在 `analysis_output/`。
第一条命令重用原评价函数，从保存的候选性质重算K1结果；不重新生成分子或预测性质。
第二条命令检查仓库中实验原件的哈希。
第三条命令检查 `results/` 中的新名称 CSV 与原件数值一致；去掉 `--check` 可重新导出。

## 重新运行生成实验

`original/` 保留集群脚本、配置和校验原件，代码位置见 [METHODS](METHODS.md)。
脚本包含原服务器路径、Slurm参数和文件校验；换机器时需要配置路径、模型、缓存和对应环境。

| 环境 | 本次实验版本 |
|---|---|
| 模型生成 | Python3.10、PyTorch2.7.0+cu128、Transformers4.46.3、PEFT0.14.0 |
| 分子解析/ADMET | `gellmo_inf`，RDKit2024.03.6；具体预测资产由`configs/properties.json`锁定 |
| 原性质评价 | Python3.6.13、RDKit2020.03.3、NumPy1.18.5、pandas1.0.5 |

模型、数据来源：[GeLLMO](https://github.com/ninglab/GeLLMO)、[NingLab模型与数据](https://huggingface.co/collections/NingLab/gellmo-67b527a2d221f06d09a240ef)。
基座为 Mistral-7B-Instruct-v0.3 和 Llama-3.1-8B-Instruct，使用对应 GeLLMO-P6 adapter。
仓库带有精确测试清单；训练集、权重、性质预测模型和完整生成缓存需单独准备。

## 文件来源

`evidence/source_manifest.json` 记录每个原件的来源、仓库位置和SHA256。
新写的可移植分析入口位于 `analysis/`；原生成算法、参数和已有结果未修改。
`original/paper/` 和 `original/results/` 保存改名前的原件；`paper/` 和 `results/` 使用新名称。
名称对照见 [NAMING](NAMING.md)。Overleaf 后续编辑需另行同步。

上游代码与模型保留原作者归属及许可；参考文献在 `paper/bibliography.bib`。
