#!/bin/bash
#SBATCH --job-name=p10_n20r20_metrics
#SBATCH --account=rc_hzh_pi
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/work/users/w/a/wanzhou/GeLLMO_main_orig/GeLLMO-main/BenchmarkV2/p10_n20_r20_two_models_20261013_v1/evaluation-%j.log
set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
cd /work/users/w/a/wanzhou/GeLLMO_main_orig/GeLLMO-main
/work/users/w/a/wanzhou/conda_envs/gellmo_eval_py36/bin/python -B BenchmarkV2/p10_n20_r20_two_models_20261013_v1/evaluate_votes.py
