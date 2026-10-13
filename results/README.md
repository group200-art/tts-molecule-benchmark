# 结果

| 目录 | 内容 |
|---|---|
| `k1/` | 两模型、九方法主表 |
| `original_metrics_k1/` | 原 GeLLMO 指标补评 |
| `paper_tables/` | 分子质量、参考分组、逐任务结果 |
| `vote_20_votes_20_runs/` | 每轮20票的 Vote，含400票单轮对照 |
| `mistral_s1_sampled_20_runs/` | Mistral 随机版 s1 budget forcing |
| `mistral_self_refine_1_revision_20_runs/` | Mistral Self-Refine，每轮修改一次 |

`first_run` 对应 K=1，`all_20_runs` 对应 K=20。字段映射见 [NAMING](../docs/NAMING.md)。

逐输入结果、JSON 和审计文件见 [original/results](../original/results/)；主表终端输出见 [data/k1_endpoints](../data/k1_endpoints/)。
