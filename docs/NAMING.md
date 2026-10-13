# 名称对照

GeLLMO 的模型名、IND/OOD、BDP 等任务名及 SR、Validity、RI、Sim、SAS、Novelty、APS 沿用原名称。
已发表的 TTS 方法沿用方法名；只把项目编号、配置简称和新增指标写清楚。数值、公式和实验设置不变。

## 方法

| 旧标签 | 显示名称 | 代码中的 method_id |
|---|---|---|
| Beam / beam20 | Beam Search | `beam_search` |
| Sampling + task verifier | Sampling + verification | `sampling_with_verification` |
| P5 / self_refine_r1 | Self-Refine（1次修改） | `self_refine` |
| P6 / tot_lm_vote | Tree of Thoughts（模型投票） | `tree_of_thoughts` |
| P8 / got | Graph of Thoughts | `graph_of_thoughts` |
| P9 / s1-style | s1 budget forcing | `s1_budget_forcing` |
| P10 / ev / Exact Voting | Vote | `vote` |
| Pplus / fewshot | 4-shot ICL | `few_shot_icl` |
| Ppro | Reflexion | `reflexion` |

`GeLLMO-P6` 是原模型名，保留；它与我们曾给 ToT 用的 `P6` 编号无关。
Sampling 仍使用验证器，s1 仍只移植 budget forcing；改名没有变成纯 Sampling 或完整 s1。
随机版 s1 单列为 `s1_budget_forcing_sampled`，不与主表的贪心解码混用。

## 配置与指标

| 旧标签 | 新名称 | 含义 |
|---|---|---|
| G12 | 12 initial nodes | GoT 初始12节点，总计28节点 |
| B512 | 512 reasoning tokens | s1 推理预算512 token，不含最终回答预算 |
| C20 | 20-candidate pool | 一次运行的20个候选 |
| R20 | 20 runs | 完整方法运行20次，各返回一个结果 |
| N20/R20 | 20 votes × 20 runs | 每轮20票，运行20轮；不同于400票单轮 |
| Macro10 | Macro avg. / `Macro_avg` | 十个任务等权平均 |
| SR_delta / SRΔ | Threshold SR / `SR_threshold` | 所有目标达到指定最小改进量的成功率 |
| RI_abs_eps01 / RI_0.1 | Stabilized RI / `RI_stabilized` | 分母为 `max(abs(source), 0.1)` 的扩展指标 |
| RI_official_code | `RI_original_code` | GeLLMO 公开代码的 RI，区别于论文公式及扩展指标 |
| Val_input / Val_slot | `input_validity_pct` / `candidate_validity_pct` | 输入级 / 候选级有效率 |
| Nov_raw_literal / Nov_canonical_reference | `Novelty_raw_smiles` / `Novelty_canonical_smiles` | 两种 SMILES 身份口径的诊断值 |

原指标补评保留 `SR`、`Sim`、`SAS`、`Novelty`、`APS`；`Validity_original` 指原评价环境，`Validity_current` 指当前保存的有效性。
`RI_original_cached`、`Sim_cached`、`SAS_cached` 保留旧缓存口径，不能与重新补评的列互换。
Threshold SR 和 Stabilized RI 是我们的扩展，不冒用原 GeLLMO 指标。

参考分组也改成描述实际条件的名称：

| 旧标签 | 新标签 | 固定 Mistral 参考池的条件 |
|---|---|---|
| Easy | `reference_beam_solved` | Beam20 至少一个成功 |
| Medium | `reference_large_pool_only` | Beam20 失败，但 Beam100 或 Sampling160 成功 |
| Hard | `reference_unsolved` | 三个参考池都失败 |
| OOD Hard | `OOD_reference_unsolved` | 上一组中的 OOD 输入 |

## 文件

`results/` 是新名称的 CSV；`paper/` 是同步名称后的 TeX。对照规则集中在 [analysis/names.py](../analysis/names.py)。
`original/results/`、`original/paper/` 保留改名前的原件；原实验代码和 `data/` 中的历史字段不变。
原件路径、文件哈希见 [source_manifest.json](../evidence/source_manifest.json)。

```bash
python3 analysis/export_results.py --check
```

该命令检查所有结果 CSV，仅允许名称变化；每个数值单元格保持原样。
