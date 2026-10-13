#!/bin/bash
#SBATCH --job-name=P8_pilot
#SBATCH --account=rc_hzh_pi
#SBATCH --partition=hzhlab
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/work/users/w/a/wanzhou/GeLLMO_main_orig/GeLLMO-main/BenchmarkV2/method_review_P8_v1/design_after_p9_v1/execution_v1
#SBATCH --output=/work/users/w/a/wanzhou/GeLLMO_main_orig/GeLLMO-main/BenchmarkV2/method_review_P8_v1/design_after_p9_v1/execution_v1/logs/pilot_%A_%a.log
#SBATCH --error=/work/users/w/a/wanzhou/GeLLMO_main_orig/GeLLMO-main/BenchmarkV2/method_review_P8_v1/design_after_p9_v1/execution_v1/logs/pilot_%A_%a.log
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --array=0-1%2
set -euo pipefail
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export P8_EXECUTION_ID=BV2-P8-GWIDTH-THREE-BUDGETS-TEN-TASKS-RUN01-v1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
exec /work/users/w/a/wanzhou/conda_envs/gellmo_mistral_gen/bin/python -B -u /work/users/w/a/wanzhou/GeLLMO_main_orig/GeLLMO-main/BenchmarkV2/method_review_P8_v1/design_after_p9_v1/execution_v1/code/worker.py pilot "${SLURM_ARRAY_TASK_ID}"
