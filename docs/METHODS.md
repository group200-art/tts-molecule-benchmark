# 方法与评价

K 为最终评价的分子数。SR 要求有效、可评价且全部目标严格改善；Mutagenicity 最小化，其余目标最大化。空、无效及已确认的分子计算失败计入分母。IND/OOD 各平均五个任务；Macro avg. 平均十个任务。

| 方法 | K=1 配置与输出 | 代码入口（`original/BenchmarkV2/` 下） |
|---|---|---|
| Beam | 宽度20，返回 rank 0 | `next3_ten_tasks_run01_v1/src/adapters.py` |
| Sampling + verification | 生成40个，以 SR 验证前20个；返回首个成功项，否则返回第0项 | `k1_nine_methods_execution_20261011_v2/code/export_cached.py` 的 `sampling_select` |
| Self-Refine | 修改一次，返回修改后的分子；模型反馈 | `p6_llama_main_ten_tasks_v1/src/self_refine.py`；`k1_nine_methods_execution_20261011_v2/code/self_refine_prepare.py` |
| ToT | 5个初始分支，模型投票保留3个；各扩展5个子节点，再投票选1个 | `k1_nine_methods_execution_20261011_v2/code/tot_worker.py` |
| GoT | 初始12节点，共28节点；返回图评分 rank 0 | `p6_llama_main_ten_tasks_v1/code/graph_adapter.py` 和 `scorer.py` |
| s1 budget forcing | 512个推理token，最多一次 Wait；单独生成最终答案 | `p6_llama_main_ten_tasks_v1/src/s1bf.py` |
| Vote | 400票，按分子身份归并，返回最高频类的首项 | `method_review_P10_v1/execution_vote_ev_v1/src/p10_vote.py` |
| 4-shot ICL | 固定四个示例，Beam4 rank 0 | `p6_llama_main_ten_tasks_v1/worker.py` 的 `run_terminal`；`configs/pplus_prompts_*.json` |
| Reflexion | 最多3次尝试，成功即停止；输入内记忆，返回最后一次动作 | `p6_llama_main_ten_tasks_v1/src/reflexion.py` |

Sampling、GoT、Reflexion 使用性质工具；s1 仅移植推理组件，ICL 为上下文基线。分子评分、反馈和解码细节见 [论文附录](../paper/appendices/implementation.tex)，文献见 [bibliography](../paper/bibliography.bib)。

原 GeLLMO 指标补评包括原代码 RI、未舍入 Sim/SAS、Novelty 和逐性质 APS。新增指标为 Threshold SR 和 Stabilized RI，定义见 [metrics](../paper/sections/metrics.tex)。
