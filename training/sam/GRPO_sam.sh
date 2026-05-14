#!/usr/bin/env bash
set -euo pipefail

AVAILABLE_GPUS=(0 1 2 3 4 5 6 7) #### Available GPUs ####
GPU_LIST=$(IFS=,; echo "${AVAILABLE_GPUS[*]}")

wait_for_free_gpus() {
  while true; do
    busy="$(nvidia-smi -i "$GPU_LIST" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk 'NF')"
    [[ -z "$busy" ]] && break
    sleep 30
  done
}

PY1=/home/ubuntu/Training/FT/sam/SAM_GRPO_beta.py

wait_for_free_gpus
accelerate launch --multi_gpu --num_processes 8 "$PY1" --exp SAM_GRPO_beta:0.1 --beta 0.1