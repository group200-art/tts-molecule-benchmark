# 方法与评价

K 表示最终评价的分子数。内部生成量和性质工具使用量允许不同。
成功要求分子有效、性质可评价，并严格改善所有目标；Mutagenicity 越低越好，其余越高越好。
无效、空输出及已确认的分子计算失败计入完整分母。IND/OOD 各平均五个任务，Macro avg. 平均十个任务。

| 方法 | 当前 K1 设置 | 主要代码入口（`original/BenchmarkV2/` 下） |
|---|---|---|
| Beam | 宽度20，返回模型 rank 0 | `next3_ten_tasks_run01_v1/src/adapters.py` |
| Sampling + verification | 生成40个；验证前20个，返回首个成功项；否则返回第0项 | `k1_nine_methods_execution_20261011_v2/code/export_cached.py` 的 `sampling_select` |
| Self-Refine | 一次修改，返回修改后的分子；不使用性质反馈 | `p6_llama_main_ten_tasks_v1/src/self_refine.py`；`k1_nine_methods_execution_20261011_v2/code/self_refine_prepare.py` |
| ToT | 5个初始分支；模型投票保留3个；各扩展5个子节点，再投票选1个 | `k1_nine_methods_execution_20261011_v2/code/tot_worker.py` |
| GoT | 初始12节点，共28节点；返回原图评分 rank 0 | `p6_llama_main_ten_tasks_v1/code/graph_adapter.py` 和 `scorer.py` |
| s1 budget forcing | 512个推理token；最多一次 Wait；返回单独生成的最终答案 | `p6_llama_main_ten_tasks_v1/src/s1bf.py` |
| Vote | 400票；按分子身份归并，取最高频类 | `method_review_P10_v1/execution_vote_ev_v1/src/p10_vote.py` |
| 4-shot ICL | 固定四个示例；Beam4 rank 0 | `p6_llama_main_ten_tasks_v1/worker.py` 的 `run_terminal`；同目录下 `configs/pplus_prompts_*.json` |
| Reflexion | 最多3次尝试；成功提前停止；记忆限于当前输入 | `p6_llama_main_ten_tasks_v1/src/reflexion.py` |

分子提示词、分子等价关系、GoT性质评分器和任务验证器是领域适配。s1只移植推理组件，没有做s1K训练；ICL是上下文基线。出处见 [论文参考文献](../paper/bibliography.bib)。

Sampling验证与最终成功条件一致，因此其K1成功率等于前20候选的成功覆盖率。这是已声明的选择规则。

K=20中需区分：一次运行的20候选池，与20次完整运行各返回一个结果。
新增 Vote 配置使用每轮20票、共20轮；原400票K1单列。随机版s1、ICL、Reflexion的K1应取同配置第一轮。

原GeLLMO指标补评使用原代码RI、未舍入Sim/SAS、Novelty和逐性质APS；与扩展指标 Stabilized RI 分开记录。
方法编号和旧字段的对应关系见 [NAMING](NAMING.md)。原代码路径保留，便于追溯。
