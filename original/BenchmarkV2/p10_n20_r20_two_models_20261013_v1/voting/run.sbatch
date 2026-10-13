#!/bin/bash
#SBATCH --job-name=p10_n20_r20_vote
#SBATCH --account=rc_hzh_pi
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/work/users/w/a/wanzhou/GeLLMO_main_orig/GeLLMO-main/BenchmarkV2/p10_n20_r20_two_models_20261013_v1/voting/run-%j.log
set -euo pipefail
unset PYTHONPATH PYTHONHOME
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec /work/users/w/a/wanzhou/conda_envs/gellmo_mistral_gen/bin/python -B /work/users/w/a/wanzhou/GeLLMO_main_orig/GeLLMO-main/BenchmarkV2/p10_n20_r20_two_models_20261013_v1/voting/extract.py
