# TTS Molecular Benchmark

把已发表的 TTS 方法接到 GeLLMO，比较它们在分子优化任务上的表现。

- 两个模型：GeLLMO-P6-Mistral 7B、GeLLMO-P6-Llama 8B。
- 九种方法，每个模型、每种方法使用全部 3,905 个任务输入。
- 主结果统一 K=1；K20 扩展单独记录。

## 先看这些

| 内容 | 文件 |
|---|---|
| 最新进度 | [STATUS](docs/STATUS.md) |
| 九种方法及代码入口 | [METHODS](docs/METHODS.md) |
| 运行方式 | [RUN](docs/RUN.md) |
| 主表结果 | [Table 1](results/k1/table1.csv) |
| 原 GeLLMO 指标补评 | [补充结果](results/original_metrics_k1/macro_metrics.csv) |
| P10 的 K1/K20 对照 | [P10](results/p10_n20_r20/macro.csv) |
| 论文源码快照 | [paper/main.tex](paper/main.tex) |

## 重算主表

```bash
python3 analysis/reproduce_table1.py
```

只需 Python 3.9+，无需 GPU。脚本核对全部 70,290 个终端输出，并重算主表。

`original/` 保存实验代码与配置原件。`data/` 保存测试清单和 K1 终端结果。
模型权重、原始训练集和大规模生成缓存保留在计算服务器；获取方式见 [RUN](docs/RUN.md)。

这是当前实验快照；更新日期及文件哈希见 [snapshot](evidence/snapshot.json)。
