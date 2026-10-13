#!/bin/bash
#SBATCH --account=rc_hzh_pi
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail
unset PYTHONPATH PYTHONHOME
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
cd /work/users/w/a/wanzhou/GeLLMO_main_orig/GeLLMO-main
exec /work/users/w/a/wanzhou/conda_envs/gellmo_eval_py36/bin/python -B BenchmarkV2/p10_n20_r20_two_models_20261013_v1/next_k20/evaluate_complete_policy.py --scope "$1"
