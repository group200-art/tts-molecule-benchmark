# 名称对照

GeLLMO 的模型、任务和指标沿用原名。历史标识与当前字段的完整映射见 [analysis/names.py](../analysis/names.py)。

| 历史方法标识 | 当前 method_id |
|---|---|
| beam20 | `beam_search` |
| sampling_verifier20 | `sampling_with_verification` |
| P5 / self_refine_r1 | `self_refine` |
| P6 / tot_lm_vote | `tree_of_thoughts` |
| P8 / got | `graph_of_thoughts` |
| P9 / s1 | `s1_budget_forcing` |
| P10 / ev / Exact Voting | `vote` |
| Pplus / fewshot | `few_shot_icl` |
| Ppro | `reflexion` |

`GeLLMO-P6` 保留原模型名。随机版 s1 使用 `s1_budget_forcing_sampled`。

| 旧标签 | 当前名称或含义 |
|---|---|
| G12 | 12 initial nodes，GoT 共28节点 |
| B512 | 512 reasoning tokens |
| C20 | 20-candidate pool |
| R20 | 20 runs，各返回一个结果 |
| N20/R20 | 20 votes × 20 runs |
| Macro10 | `Macro_avg`：十任务等权平均 |
| SR_delta / SRΔ | `SR_threshold`：所有目标达到最小改进量 |
| RI_abs_eps01 / RI_0.1 | `RI_stabilized`：分母为 `max(abs(source), 0.1)` |
| RI_official_code | `RI_original_code`：GeLLMO 公开代码的公式 |
| Val_input / Val_slot | `input_validity_pct` / `candidate_validity_pct` |
| Nov_raw_literal / Nov_canonical_reference | `Novelty_raw_smiles` / `Novelty_canonical_smiles` |

`Validity_original` 使用原评价环境，`Validity_current` 使用保存的有效性。`RI_original_cached`、`Sim_cached`、`SAS_cached` 对应历史缓存精度。

| 旧分组 | 当前字段 | Mistral 参考池条件 |
|---|---|---|
| Easy | `reference_beam_solved` | Beam20 成功 |
| Medium | `reference_large_pool_only` | Beam20 失败，Beam100 或 Sampling160 成功 |
| Hard | `reference_unsolved` | 三个池均失败 |
| OOD Hard | `OOD_reference_unsolved` | 上一组中的 OOD 输入 |
