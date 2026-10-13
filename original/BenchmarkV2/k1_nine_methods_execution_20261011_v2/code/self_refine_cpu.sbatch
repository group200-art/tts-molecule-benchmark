#!/bin/bash
#SBATCH --job-name=k1_r1_props
#SBATCH --account=rc_hzh_pi
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-9%4
#SBATCH --no-requeue
#SBATCH --mail-type=NONE
#SBATCH --output=/work/users/w/a/wanzhou/GeLLMO_main_orig/GeLLMO-main/BenchmarkV2/k1_nine_methods_execution_20261011_v2/logs/self_refine_%A_%a.out
#SBATCH --error=/work/users/w/a/wanzhou/GeLLMO_main_orig/GeLLMO-main/BenchmarkV2/k1_nine_methods_execution_20261011_v2/logs/self_refine_%A_%a.err
set -euo pipefail
unset PYTHONPATH PYTHONHOME
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec /work/users/w/a/wanzhou/conda_envs/gellmo_mistral_gen/bin/python -B /work/users/w/a/wanzhou/GeLLMO_main_orig/GeLLMO-main/BenchmarkV2/k1_nine_methods_execution_20261011_v2/code/self_refine_run.py
