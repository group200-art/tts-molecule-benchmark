# Inference Strategies for Molecular Optimization

This repository contains code, fixed configurations, input manifests and saved-result reproduction materials for the RECOMB 2027 benchmark manuscript. **Use [`release_20261105/`](release_20261105/) for the current study.** The other top-level directories are the historical October 13 snapshot; their paper and results do not describe the final study.

## Study scope

| Analysis | Inputs per configuration | Configurations |
| --- | ---: | --- |
| Original comparison | 3,905 task–input pairs across ten tasks | Nine strategies on two fixed GeLLM³O P(6) backbones |
| Similarity constraint, GoT fixed-set ordering, Vote fixed-pool selection | The same 3,905 task–input pairs | Saved original outputs; no new generation |
| GoT generation control | 500 fixed inputs, 50 per task | Two backbones; GoT and independent Sampling with the same GoT scorer; one generation seed, 1729 |

The original comparison retains twenty ordered output positions, including empty, invalid and duplicate entries. Prefix success rates use the first 1, 5, 10 or 20 positions of the same saved output sequence. Output counts are not matched total compute budgets. The new 500-input control is separate and never replaces the original results.

## Reproduce the published statistics on a CPU

Use Python 3.10 with NumPy 2.2.6. These commands read the included saved per-input outcomes and fixed manifests. They do not load models, call molecular predictors, or generate new molecules.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy==2.2.6
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python release_20261105/main_results/reproduce.py --output reproduced/main

cd release_20261105/supplement
python reproduce_metrics.py \
  --data data/offline_per_input_metrics.csv.gz \
  --expected results/offline_summary.csv \
  --paired results/offline_paired_differences.csv \
  --manifest evidence/input_manifest_3905.csv \
  --expected-count 3905 --output reproduced/offline
python reproduce_metrics.py \
  --data data/got_control_per_input_metrics.csv.gz \
  --expected results/got_control_per_seed_summary.csv \
  --paired results/got_control_paired_differences.csv \
  --manifest evidence/input_manifest_500.csv \
  --expected-count 500 --output reproduced/got_control
```

Each command checks completeness against the fixed input scope and compares recomputed values with the frozen reference results. Confidence intervals use 2,000 bootstrap replicates clustered by raw source SMILES, equal task weights, and bootstrap seed 20261014. These intervals are conditional on the saved generations; they do not estimate variation across generation seeds.

The release was checked on November 5, 2026. Main-result and supplementary verification reports are included alongside the code. Supplementary checks cover 538 original-output summaries and 58 paired contrasts, plus 44 new-control summaries and 14 paired contrasts; the maximum numerical discrepancy was below 3e-14.

## Code and data map

- [`main_results`](release_20261105/main_results/): frozen per-input readouts, task manifests, portable aggregation and bootstrap programs, reference tables and checks.
- [`main_sources`](release_20261105/main_sources/): main generation, prompting, selection, property evaluation and configuration snapshots; source provenance and environment requirements. See its README for the method-to-source map and external assets.
- [`supplement`](release_20261105/supplement/): fixed-pool/similarity analysis, paired GoT control, saved per-input outcomes, exact scientific source snapshots and reproduction instructions.
- [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md): upstream code, dataset and model attribution.

## Reproduction boundaries

Saved-outcome aggregation is self-contained in the current release. Fresh model generation is a separate workflow that requires the upstream model weights, property predictor assets, their original software environments, and adaptation of the documented deployment paths. The scientific source snapshot is not a one-command portable training or generation installer. Internal scheduler wrappers, troubleshooting records and user-planning documents are excluded from this public copy.

The repository does not distribute model weights or all original model responses and candidate pools. Original Mistral Beam and Sampling raw response prose was not retained in the source archive; extracted candidates and property readouts were available for the reported analysis. Complete new-control candidate archives are retained separately by the authors and are indexed by checksum. The compact release contains their per-input outcomes and the scripts used to check candidate-level records. See [`supplement/README.md`](release_20261105/supplement/README.md).

For workload reproduction, per-input records support reaggregation of the four final ToT/Reflexion configurations; the other fourteen configurations retain audited aggregate totals, as documented in `main_results`. Do not interpret the availability of aggregate statistics as availability of every raw generation log.

The public supplementary per-arm workload table excludes complete internal runtime traces, internal filesystem paths and raw trace identifiers; all 2,000 rows and retained scientific counters and timings are unchanged.

The planned 996-input, three-generation-seed study and Reflexion ablation were not completed and are not reported as results. Some archived source files retain their original planning defaults; the completed supplementary scope is the fixed 500-input, single-seed scope in `supplement/evidence/scope.json` and its completion record.

## Upstream resources

- GeLLM³O code: https://github.com/NingLab/GeLLMO
- Dataset: https://huggingface.co/datasets/NingLab/MuMOInstruct
- Adapters: https://huggingface.co/NingLab/GeLLMO-P6-Mistral and https://huggingface.co/NingLab/GeLLMO-P6-Llama
- Base models: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3 and https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

Llama access requires the upstream provider's access process. No access credentials or model weight redistribution is provided here. Retain the upstream license and citation requirements when using the code, dataset or checkpoints.
