#!/bin/bash
set -euo pipefail
unset PYTHONPATH PYTHONHOME
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''
exec /work/users/w/a/wanzhou/conda_envs/gellmo_eval_py36/bin/python -B /work/users/w/a/wanzhou/GeLLMO_main_orig/GeLLMO-main/BenchmarkV2/gellmo_eval_alignment_20261013_v1/k1/code/evaluate_k1.py
