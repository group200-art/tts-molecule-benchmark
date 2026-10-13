# TTS Molecular Benchmark

在固定 GeLLMO 模型上测评已发表的 TTS 方法：Mistral 7B、Llama 8B，九种方法、十个任务、每配置 3,905 个输入。主表 K=1；K=20 扩展进度见 STATUS。

| 内容 | 文件 |
|---|---|
| 实验进度 | [STATUS](docs/STATUS.md) |
| 方法配置与代码入口 | [METHODS](docs/METHODS.md) |
| 运行与依赖 | [RUN](docs/RUN.md) |
| 名称对照 | [NAMING](docs/NAMING.md) |
| 主表 | [Table 1](results/k1/table1.csv) |
| 原 GeLLMO 指标 | [补评结果](results/original_metrics_k1/macro_metrics.csv) |
| Vote 的 K1/K20 对照 | [Vote](results/vote_20_votes_20_runs/macro.csv) |
| 论文源码 | [paper/main.tex](paper/main.tex) |

```bash
python3 analysis/reproduce_table1.py
```

Python 3.9+ 标准库即可从 70,290 个保存的输出重算主表，结果写入 `analysis_output/`。

`data/` 保存测试清单与 K1 输出，`results/` 保存汇总，`original/` 保存实验原件。文件来源、哈希和快照时间见 [evidence](evidence/snapshot.json)。
