#!/bin/bash
set -euo pipefail
# =============================================================================
# SFT Training Script for Qwen3.5-35B-A3B (MoE) using ms-swift
# Platform: Merlin (Arnold) multi-node training
# =============================================================================
# Usage:
#   1. Set MODEL_PATH, DATASET_NAME, DATASET_INFO, OUTPUT_DIR below
#   2. Submit via Merlin with desired node count and GPUs per node
#   3. ARNOLD_NUM, ARNOLD_WORKER_GPU, ARNOLD_ID, ARNOLD_WORKER_0_HOST,
#      ARNOLD_WORKER_0_PORT are auto-set by Merlin
#
# Data format (.jsonl, one JSON per line):
#   {"messages": [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]}
# =============================================================================

# ─── Dependencies ───
echo "[$(date '+%H:%M:%S')] Installing dependencies..."
pip install -U ms-swift deepspeed 2>&1 | tail -1
echo "[$(date '+%H:%M:%S')] ms-swift + deepspeed done"

pip install -U "transformers>=5.2.0,<5.3.0" "qwen_vl_utils>=0.0.14" peft liger-kernel 2>&1 | tail -1
echo "[$(date '+%H:%M:%S')] transformers + peft done"

# flash-linear-attention & causal-conv1d (OPTIONAL)
# These provide optimized GatedDeltaNet kernels for Qwen3.5.
# Without them, training still works but uses slower eager attention.
# To enable: pre-build wheels on a GPU machine with matching CUDA/Python,
# then install from the wheel files here. See README for instructions.
#
# pip install /mnt/bn/youxiang-lf/packages/wheels/fla-*.whl
# pip install /mnt/bn/youxiang-lf/packages/wheels/causal_conv1d-*.whl

echo "[$(date '+%H:%M:%S')] All dependencies installed (fla/causal_conv1d skipped — using eager fallback)"

# =============================================================================
# USER CONFIG
# =============================================================================
MODEL_PATH="/mnt/bn/youxiang-lf/models/Qwen3.5-35B-A3B"
DATASET_NAME="topic_generation_policy_sft-ms-swift"
DATASET_INFO="/mnt/bn/youxiang-lf/data/dataset_info_ms_swift.json"

run_name="Qwen3.5-35B-A3B-topic-sfted-human-label-data-1k"
OUTPUT_DIR="/mnt/bn/youxiang-lf/models/${run_name}"

export WANDB_PROJECT="Qwen3.5-SFT"
export WANDB_NAME="$run_name"

# =============================================================================
# Multi-node detection and setup (Merlin/Arnold)
# =============================================================================
nnodes=${ARNOLD_NUM:-1}
nproc_per_node=${ARNOLD_WORKER_GPU:-8}
export NNODES=$nnodes
export NODE_RANK=${ARNOLD_ID:-0}
export NPROC_PER_NODE=$nproc_per_node
export MASTER_ADDR=${MASTER_ADDR:=$ARNOLD_WORKER_0_HOST}
export MASTER_PORT=${MASTER_PORT:=$(echo "$ARNOLD_WORKER_0_PORT" | cut -d "," -f 1)}

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

echo "═══════════════════════════════════════════════════════════════"
echo "Qwen3.5 SFT Training on Merlin"
echo "Nodes: $NNODES, GPUs/Node: $NPROC_PER_NODE, Total GPUs: $((NNODES * NPROC_PER_NODE))"
echo "Master: $MASTER_ADDR:$MASTER_PORT, Node Rank: $NODE_RANK"
echo "═══════════════════════════════════════════════════════════════"

mkdir -p "$OUTPUT_DIR"

# ─── Log files ───
STDOUT_LOG="stdout.log"
STDERR_LOG="stderr.log"

# =============================================================================
# Launch Training
# =============================================================================
swift sft \
    --model ${MODEL_PATH} \
    --custom_dataset_info ${DATASET_INFO} \
    --dataset "${DATASET_NAME}" \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --target_modules all-linear \
    --experts_impl grouped_mm \
    --router_aux_loss_coef 1e-3 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --group_by_length true \
    --split_dataset_ratio 0.05 \
    --output_dir ${OUTPUT_DIR} \
    --save_steps 50 \
    --save_total_limit 3 \
    --save_only_model true \
    --eval_steps 50 \
    --logging_steps 5 \
    --max_length 2048 \
    --warmup_ratio 0.05 \
    --dataset_num_proc 8 \
    --dataloader_num_workers 8 \
    --report_to wandb \
    --run_name "$run_name" \
    --seed 42 \
    --data_seed 42 \
    --deepspeed zero3 \
    > >(tee "$STDOUT_LOG") 2> >(tee "$STDERR_LOG" >&2)

# =============================================================================
# After training, run inference on the trained model:
#
# CUDA_VISIBLE_DEVICES=0 \
# swift infer \
#     --model ${OUTPUT_DIR}/vx-xxx/checkpoint-xxx \
#     --stream true \
#     --experts_impl grouped_mm \
#     --enable_thinking false \
#     --max_new_tokens 512
# =============================================================================
