#!/bin/bash
set -euo pipefail
unset PYTHONPATH PYTHONHOME
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export THREE_K20_EXECUTION=BV2-REMAINING-THREE-K20-v1
exec /work/users/w/a/wanzhou/conda_envs/gellmo_mistral_gen/bin/python -B /work/users/w/a/wanzhou/GeLLMO_main_orig/GeLLMO-main/BenchmarkV2/p10_n20_r20_two_models_20261013_v1/next_k20/resume_mistral_s1.py --unit-index "$SLURM_ARRAY_TASK_ID"
