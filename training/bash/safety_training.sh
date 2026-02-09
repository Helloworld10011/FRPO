#!/usr/bin/env bash
set -euo pipefail
export WANDB_PROJECT=    ### set your wandb project name here ###
AVAILABLE_GPUS=(0 1 2 3 4 5 6 7) #### Available GPUs ####
GPU_LIST=$(IFS=,; echo "${AVAILABLE_GPUS[*]}")

export OPENAI_API_KEY=   ### Set your OpenAI API key here -- This is required for Moderation API ###

wait_for_free_gpus() {
  while true; do
    busy="$(nvidia-smi -i "$GPU_LIST" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk 'NF')"
    [[ -z "$busy" ]] && break
    sleep 30
  done
}
BASE_DIR=$(dirname "$0")
PY1="$BASE_DIR/../safety_training/GRPO_training_mistral.py"
PY2="$BASE_DIR/../safety_training/FRPO_training_mistral.py"

model_id=""   ### Set this to your desired model path ###
training_data="$BASE_DIR/../data/train_lambda_0.5.parquet"
path_to_SR=""   ### Load the StrongREJECT model here. Read: https://colab.research.google.com/drive/1wC0nCH9_Umxkq87gQMEceTzktyP4_ZJn?usp=sharing#scrollTo=QNedWPA1rnhY"
path_to_RE="allenai/Llama-3.1-8B-Instruct-RM-RB2"
output_dir_base=""  ### Set this to your desired output directory base path ###

wait_for_free_gpus
GPU_LIST=$(IFS=,; echo "${AVAILABLE_GPUS[*]}")
CUDA_VISIBLE_DEVICES="$GPU_LIST" $CONDA_PREFIX/bin/accelerate launch --multi_gpu --num_processes 8 "$PY1" --exp GRPO_safety_mistral --beta 0.1 --model_id "$model_id" --training_data "$training_data" --path_to_SR "$path_to_SR" --path_to_RE "$path_to_RE" --output_dir_base "$output_dir_base"

LAMBDAS=(5.0 2.0 1.0 0.5 0.2 0.1)
BETAS=(0.0002 0.0005 0.001 0.002 0.005 0.1)

for i in "${!LAMBDAS[@]}"; do
  wait_for_free_gpus
  CUDA_VISIBLE_DEVICES="$GPU_LIST" $CONDA_PREFIX/bin/accelerate launch --multi_gpu --num_processes 8 "$PY2" \
    --exp "FRPO_safety_mistral" \
    --beta "${BETAS[$i]}" \
    --lamb "${LAMBDAS[$i]}" \
    --model_id "$model_id" \
    --training_data "$training_data" \
    --path_to_SR "$path_to_SR" \
    --path_to_RE "$path_to_RE" \
    --output_dir_base "$output_dir_base" \
    --jackknife \
    --baseline
done