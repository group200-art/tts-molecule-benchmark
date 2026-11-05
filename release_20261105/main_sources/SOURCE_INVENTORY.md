# Accepted main-source inventory

All paths below are relative to `snapshot/BenchmarkV2/`. `MAIN_CONFIGURATIONS.json` gives the machine-readable 18-configuration map. Sources under a predecessor directory remain necessary dependencies; their older standalone configurations are not main-study rows.

| Method | Mistral source | Llama source | Accepted returned twenty |
|---|---|---|---|
| Beam | `next3_ten_tasks_run01_v1/pilot_worker.py` | `p6_llama_failure_recovery_20261003_v1/worker.py` | Native beam width 20 and rank order; cap 100 |
| Sampling | Same Beam entrypoints | Same Beam entrypoints | Fixed raw first twenty of the original forty samples; T=1, top-p=.95, top-k=50, cap 100 |
| Self-Refine | `self_refine_r1_k20_v1/generate.py` | `benchmark_paired_k1_k20_20261014_v1/llama_self_refine/code/run.py` | One revision from each replicate ID 51001–51020; T=.7, init/refine cap 250, feedback cap 400; retain the final feedback call |
| ToT | `benchmark_completion_20261023_v1/tot/runner.py` | Same, backbone-specific model argument | Five initial candidates; four LM-voted parents; five children per parent; five final votes rank all twenty; no property access during search |
| GoT | `method_review_P8_v1/design_after_p9_v1/execution_v1/code/worker.py` | `p6_llama_failure_recovery_20261003_v1/worker.py` | Twelve initial candidates plus six refinements, six pair merges, four aggregations; original full-pool scorer returns twenty |
| s1 | `p10_n20_r20_two_models_20261013_v1/next_k20/resume_mistral_s1.py` | `benchmark_paired_k1_k20_20261014_v1/llama_s1/runner.py` | Twenty sampled fresh-state trajectories; cumulative 512-token thinking budget, at most one Wait insertion, final cap 256 |
| Vote | `p10_n20_r20_two_models_20261013_v1/voting/extract.py` | Same selector, separate saved proposal pools | Twenty independent exact votes over consecutive groups of twenty original proposals; T=.1, proposal cap 256 |
| ICL | `k20_remaining_three_v1/worker.py` | `p10_n20_r20_two_models_20261013_v1/next_k20/llama_icl_sampled.py` | Twenty sampled outputs of the same input-specific frozen four-demonstration prompt; T=1, top-p=1, top-k=0, cap 100 |
| Reflexion | `benchmark_completion_20261023_v1/mistral_reflexion/runner.py` | `benchmark_completion_20261023_v1/llama_reflexion/worker.py` | Twenty fresh-memory sampled trajectories, first success or trial three within each; T=1, role cap 250 |

Generation seed formulas, physical batches, parser behavior, model assembly, and tokenizer details are part of the copied contracts and source. A replicate ID is not necessarily the literal seed of a batched generation call. Mistral Self-Refine and Llama Self-Refine also have different accepted physical batching. The published saved generations do not support a claim of seed stability.

The authoritative shared scientific definitions are:

- Success, thresholds, representative selection, and RI: `p6_llama_failure_recovery_20261003_v1/code/metrics_kernel.py`.
- Original round-two CSV conversion: `p6_llama_failure_recovery_20261003_v1/code/precision.py` and the accepted property producers. Raw candidate precision must not replace original comparison precision.
- GoT scoring: `p6_llama_failure_recovery_20261003_v1/code/scorer.py`, version `P8-MINMAX-TASK-3OVERM-v1`; graph prompts and control in its `graph_adapter.py`, with the Mistral tree also retained. The scorer handles supported missing-property cases explicitly; normalization and stable tie ordering must remain unchanged.
- Vote selection: `p10_n20_r20_two_models_20261013_v1/voting/extract.py`, function `count_vote`. It counts original exact `(class_type, class_value)` keys, including invalid/empty classes, resets for every physical group, and resolves ties by earliest original global slot; it does not use the property-success label.
- Reflexion controller and literal prompts: `p6_llama_main_ten_tasks_v1/src/reflexion.py` and `configs/reflexion_prompts.json`, with the Mistral counterparts in `k20_alignment_20261002_v1/`. The original actor receives reflection memory; raw feedback goes to the reflection prompt. Do not relabel the original main configuration as the uncompleted raw-feedback ablation.
- Final molecular-exception policy: `benchmark_completion_20261023_v1/recovery_20261027_v1/property_repair.py` plus final Mistral and Llama property-service adapters and freeze ancestry. Known isolated chemical failures retain their output positions; unknown/infrastructure errors remain errors.
- Statistics: `recomb_full_results_20261028_v1/build_statistics.py`. It resamples 2,860 original source identifiers jointly across tasks/methods, recomputes task denominators, averages tasks equally, and uses 2,000 replicates with seed 20261014. Pointwise intervals are conditional on the saved generation runs.

The exact task registry remains named `registries/task_registry.draft.json` because this is the historical file used by the completed execution. There are seven tasks of 500 inputs and three tasks of 118,191,96 inputs, totaling 3,905. No original task was substituted in this release.
