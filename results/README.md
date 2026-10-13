# 结果

这些 CSV 使用 [新名称](../docs/NAMING.md)，所有数值保持原样。

| 目录 | 内容 |
|---|---|
| `k1/` | 两模型、九方法主表 |
| `original_metrics_k1/` | 原 GeLLMO 指标补评 |
| `paper_tables/` | 分子质量、参考分组、逐任务结果 |
| `vote_20_votes_20_runs/` | 每轮20票的 Vote；含原400票单轮对照 |
| `mistral_s1_sampled_20_runs/` | Mistral 随机解码的 s1 budget forcing |
| `mistral_self_refine_1_revision_20_runs/` | Mistral Self-Refine，每轮修改一次 |

`first_run` 表示只评价同配置的第一轮（K=1）；`all_20_runs` 表示评价20轮最终结果（K=20）。
Vote 的名称同时写明每轮票数，400票单轮与20票多轮分别记录。

逐输入数据、JSON、完成审计、指标定义及改名前的 CSV 均在 [original/results](../original/results/)。
主表终端输出在 [data/k1_endpoints](../data/k1_endpoints/)。

```bash
python3 analysis/export_results.py --check
```
