#!/bin/bash
set -euo pipefail

# 1. GPU Configuration
BASE_DIR=   ### Set this to your desired base directory for logs and results
AVAILABLE_GPUS=(0 1 2 3)
GPU_LIST=$(IFS=,; echo "${AVAILABLE_GPUS[*]}")
NUM_GPUS=${#AVAILABLE_GPUS[@]}

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=1

### Add the models and experiment names you want to run here. Make sure they are in the same order.
MODELS=(

        ) 

EXPS=(
    
)


# Helper Function to ensure isolation
wait_for_free_gpus() {
  echo "Checking for free GPUs ($GPU_LIST)..."
  while true; do
    busy="$(nvidia-smi -i "$GPU_LIST" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk 'NF')"
    if [[ -z "$busy" ]]; then
        echo "GPUs $GPU_LIST are free. Starting next run."
        break
    fi
    echo "GPUs busy. Waiting 30s..."
    sleep 30
  done
}

mkdir -p "$BASE_DIR/logs"
mkdir -p "$BASE_DIR/results"

for i in "${!MODELS[@]}"; do
    wait_for_free_gpus
    
    MODEL="${MODELS[$i]}"
    EXP_NAME="${EXPS[$i]}"
    
    echo "========================================================"
    echo "Running MBPP for: $EXP_NAME"
    echo "Using $NUM_GPUS GPUs (Tensor Parallelism)"
    echo "========================================================"
    
    # Create experiment output directory
    mkdir -p "$BASE_DIR/results/$EXP_NAME"
    
    CUDA_VISIBLE_DEVICES="$GPU_LIST" evalplus.evaluate \
        --model "$MODEL" \
        --dataset mbpp \
        --backend vllm \
        --tp "$NUM_GPUS" \
        --greedy \
        --trust_remote_code \
        2>&1 | tee "$BASE_DIR/logs/${EXP_NAME}.log"
    
    # Move evalplus results to experiment folder
    # EvalPlus outputs to: evalplus_results/mbpp/
    if [ -d "evalplus_results/mbpp" ]; then
        mv evalplus_results/mbpp/* "$BASE_DIR/results/$EXP_NAME/" 2>/dev/null || true
    fi
    
done

echo "========================================================"
echo "ALL RUNS COMPLETE. FINAL SCORES:"
echo "========================================================"
# Print results from logs
for i in "${!EXPS[@]}"; do
    EXP_NAME="${EXPS[$i]}"
    echo "=== $EXP_NAME ==="
    grep -E "(pass@1|Base|Plus)" "$BASE_DIR/logs/${EXP_NAME}.log" | tail -5
done