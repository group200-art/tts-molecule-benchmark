# 实验进度

2026-10-13 [快照](../evidence/progress_snapshot.json)。

**K=1 已完成：两个模型、九种方法、每配置 3,905 个输入，共 70,290 个输出。** [原 GeLLMO 七项指标补评](../results/original_metrics_k1/macro_metrics.csv)已完成。

## K=20

| 方法 | 配置 | Mistral | Llama |
|---|---|---|---|
| Beam | 20候选池 | 已有 | 已有 |
| Sampling | 原始前20候选 | 已有 | 已有 |
| Self-Refine | 每轮修改一次，20轮 | 完成 | 未齐 |
| ToT | 模型投票，20轮 | 未齐 | 未齐 |
| GoT | 原图20候选 | 已有 | 已有 |
| s1 budget forcing | 随机解码，20轮 | 生成与补评完成 | 未齐 |
| Vote | 每轮20票，20轮 | 生成与补评完成 | 生成与补评完成 |
| 4-shot ICL | 随机解码，20轮 | 生成完成，待补评 | 生成中 |
| Reflexion | 随机解码，20轮 | 试跑完成 | 未齐 |

候选池来自一次运行；多轮配置每轮返回一个结果。主表 Vote 使用400票单轮。随机版 s1、ICL、Reflexion 的 K1/K20 对照取同配置第一轮与全部20轮。

“已有”包含服务器上的缓存；仓库文件范围见 [来源清单](../evidence/source_manifest.json)。
