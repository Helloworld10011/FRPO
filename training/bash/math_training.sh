#!/usr/bin/env bash
set -euo pipefail
export WANDB_PROJECT=    ### set your wandb project name here ###
AVAILABLE_GPUS=(0 1 2 3 4 5 6 7) #### Available GPUs ####
GPU_LIST=$(IFS=,; echo "${AVAILABLE_GPUS[*]}")

wait_for_free_gpus() {
  while true; do
    busy="$(nvidia-smi -i "$GPU_LIST" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk 'NF')"
    [[ -z "$busy" ]] && break
    sleep 30
  done
}

export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=0
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE_DIR=$(dirname "$0")
PY1="$BASE_DIR/../math_training/GRPO_math_nocot.py"
PY2="$BASE_DIR/../math_training/FRPO_math_nocot.py"

model_id=""   ### Set this to your desired model path ###
output_dir_base=""   ### Set this to your desired output directory base path ###

wait_for_free_gpus
CUDA_VISIBLE_DEVICES="$GPU_LIST" $CONDA_PREFIX/bin/accelerate launch --multi_gpu --num_processes "${#AVAILABLE_GPUS[@]}" "$PY1" --exp GRPO_math_nocot --model_path "$model_id" --output_dir_base "$output_dir_base"

LAMBDAS=(10.0 4.0 2.0 1.0 0.5 0.2 0.1)
LR=(0.000006 0.000006 0.000006 0.000006 0.000005 0.000005 0.000005)

for i in "${!LAMBDAS[@]}"; do
  wait_for_free_gpus
  CUDA_VISIBLE_DEVICES="$GPU_LIST" $CONDA_PREFIX/bin/accelerate launch --multi_gpu --num_processes "${#AVAILABLE_GPUS[@]}" "$PY2" \
    --exp FRPO_math_nocot \
    --model_path "$model_id" \
    --lamb "${LAMBDAS[i]}" \
    --lr "${LR[i]}" \
    --output_dir_base "$output_dir_base" \
    --jackknife \
    --baseline
done