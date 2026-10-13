#!/bin/bash
set -euo pipefail
unset PYTHONPATH PYTHONHOME
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec /work/users/w/a/wanzhou/conda_envs/gellmo_mistral_gen/bin/python -B /work/users/w/a/wanzhou/GeLLMO_main_orig/GeLLMO-main/BenchmarkV2/p10_n20_r20_two_models_20261013_v1/properties/code/audit_cache_raw_differences.py
