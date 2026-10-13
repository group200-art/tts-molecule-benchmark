# 运行

## 结果复现

Python 3.9+，仅需标准库。

```bash
python3 analysis/reproduce_table1.py       # 从缓存性质重算 K1，输出到 analysis_output/
python3 analysis/verify_snapshot.py        # 核对原件哈希
python3 analysis/export_results.py --check # 核对 CSV 导出；去掉 --check 可重新导出
```

## 生成实验

代码入口见 [METHODS](METHODS.md)。`original/` 中的 Slurm 脚本使用集群路径；运行前配置模型、数据、缓存路径和环境。

| 环境 | 实验版本 |
|---|---|
| 模型生成 | Python 3.10、PyTorch 2.7.0+cu128、Transformers 4.46.3、PEFT 0.14.0 |
| 分子解析/ADMET | `gellmo_inf`、RDKit 2024.03.6；预测资产见 `configs/properties.json` |
| 原性质评价 | Python 3.6.13、RDKit 2020.03.3、NumPy 1.18.5、pandas 1.0.5 |

模型与数据：[GeLLMO](https://github.com/ninglab/GeLLMO)、[NingLab](https://huggingface.co/collections/NingLab/gellmo-67b527a2d221f06d09a240ef)。基座为 Mistral-7B-Instruct-v0.3 和 Llama-3.1-8B-Instruct，加载对应 GeLLMO-P6 adapter。
仓库包含测试清单与 K1 输出；训练集、权重、性质预测模型和完整生成缓存需单独准备。

## 文件来源

[source_manifest.json](../evidence/source_manifest.json) 记录来源、路径和 SHA256。`original/` 保存代码、结果及论文原件；`analysis/` 提供分析入口。`paper/` 与 Overleaf 需另行同步。
