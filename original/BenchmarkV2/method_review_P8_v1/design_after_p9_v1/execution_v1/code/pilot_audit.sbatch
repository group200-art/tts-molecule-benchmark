#!/bin/bash
#SBATCH --job-name=P8_pilot_audit
#SBATCH --account=rc_hzh_pi
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/work/users/w/a/wanzhou/GeLLMO_main_orig/GeLLMO-main/BenchmarkV2/method_review_P8_v1/design_after_p9_v1/execution_v1
#SBATCH --output=/work/users/w/a/wanzhou/GeLLMO_main_orig/GeLLMO-main/BenchmarkV2/method_review_P8_v1/design_after_p9_v1/execution_v1/logs/pilot_audit_%A_%a.log
#SBATCH --error=/work/users/w/a/wanzhou/GeLLMO_main_orig/GeLLMO-main/BenchmarkV2/method_review_P8_v1/design_after_p9_v1/execution_v1/logs/pilot_audit_%A_%a.log
set -euo pipefail
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export P8_EXECUTION_ID=BV2-P8-GWIDTH-THREE-BUDGETS-TEN-TASKS-RUN01-v1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=""
exec /work/users/w/a/wanzhou/conda_envs/gellmo_mistral_gen/bin/python -B -u /work/users/w/a/wanzhou/GeLLMO_main_orig/GeLLMO-main/BenchmarkV2/method_review_P8_v1/design_after_p9_v1/execution_v1/code/stages.py audit
