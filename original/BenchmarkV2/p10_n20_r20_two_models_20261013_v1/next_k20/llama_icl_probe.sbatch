#!/bin/bash
set -euo pipefail
unset PYTHONPATH PYTHONHOME
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export NEXT_K20_EXECUTION=GELLMO-LLAMA-ICL4-SAMPLED-R20-20261013-v1
stage_cache=/work/users/w/a/wanzhou/GeLLMO_main_orig/GeLLMO-main/BenchmarkV2/p10_n20_r20_two_models_20261013_v1/next_k20/llama_icl_sampled/cache/$SLURM_JOB_ID
mkdir -p "$stage_cache"
export TMPDIR="$stage_cache" HF_HOME="$stage_cache/hf" XDG_CACHE_HOME="$stage_cache/xdg" TORCH_HOME="$stage_cache/torch" MPLCONFIGDIR="$stage_cache/mpl" TRITON_CACHE_DIR="$stage_cache/triton" CUDA_CACHE_PATH="$stage_cache/cuda" NUMBA_CACHE_DIR="$stage_cache/numba"
exec /work/users/w/a/wanzhou/conda_envs/gellmo_mistral_gen/bin/python -B /work/users/w/a/wanzhou/GeLLMO_main_orig/GeLLMO-main/BenchmarkV2/p10_n20_r20_two_models_20261013_v1/next_k20/llama_icl_sampled.py --probe
