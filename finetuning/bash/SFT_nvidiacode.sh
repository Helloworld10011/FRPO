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
BASE_DIR=$(dirname "$0")
PY=$BASE_DIR/../SFT_nvidiacode.py

### Add the models and experiment names you want to run here. Make sure they are in the same order.
MODELS=(
        "/home/ubuntu/Models/Qwen/Qwen2.5-7B-Instruct"
        )
EXPS=(
  "SFT_nvidiacode/Qwen2.5-7B-Instruct"
)
output_dir_base="" ### Set your output directory base here. This is where the model checkpoints will be saved. ###

for i in "${!MODELS[@]}"; do
  wait_for_free_gpus
  accelerate launch --multi_gpu --num_processes 8 "$PY" \
    --exp "${EXPS[i]}" \
    --model_path "${MODELS[i]}" \
    --seed 42 \
    --output_dir_base "$output_dir_base"
done
