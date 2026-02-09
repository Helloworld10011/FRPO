#!/usr/bin/env bash
set -euo pipefail
export WANDB_PROJECT=    ### set your wandb project name here ###
AVAILABLE_GPUS=(0 1 2 3 4 5 6 7) #### Available GPUs ####
GPU_LIST=$(IFS=,; echo "${AVAILABLE_GPUS[*]}")

export OPENAI_API_KEY=    ### Set your OpenAI API key here -- This is required for Moderation API ###


wait_for_free_gpus() {
  while true; do
    busy="$(nvidia-smi -i "$GPU_LIST" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk 'NF')"
    [[ -z "$busy" ]] && break
    sleep 30
  done
}

BASE_DIR=$(dirname "$0")
PY="$BASE_DIR/../SFT_gsm8k.py"

### Add the models and experiment names you want to run here. Make sure they are in the same order.
MODELS=(
        "/home/ubuntu/Models/Qwen/Qwen2.5-7B-Instruct"
        )
EXPS=(
      "SFT_gsm8k_lr6e-6/Qwen2.5-7B-Instruct"
      )

RESULTS_DIR="" ### Set your results directory here. This is where the Harmbench results will be saved. ###
OUTPUT_DIR_BASE="" ### Set your output directory base here. This is where the model checkpoints will be saved. ###
PATH_TO_SR=""   ### Load the StrongREJECT model here. Read: https://colab.research.google.com/drive/1wC0nCH9_Umxkq87gQMEceTzktyP4_ZJn?usp=sharing#scrollTo=QNedWPA1rnhY"
PATH_TO_RE="allenai/Llama-3.1-8B-Instruct-RM-RB2"

for i in "${!MODELS[@]}"; do
  wait_for_free_gpus
  CUDA_VISIBLE_DEVICES="$GPU_LIST" $CONDA_PREFIX/bin/accelerate launch --multi_gpu --num_processes 8 "$PY" \
    --exp "${EXPS[i]}" \
    --model_path "${MODELS[i]}" \
    --results_dir "${RESULTS_DIR}" \
    --seed 42 \
    --output_dir_base "${OUTPUT_DIR_BASE}" \
    --path_to_SR "${PATH_TO_SR}" \
    --path_to_RE "${PATH_TO_RE}"
done