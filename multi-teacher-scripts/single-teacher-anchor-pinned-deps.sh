#!/bin/bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════════
# Single-Teacher GKD: anchor_memory teacher — PINNED DEPENDENCIES
# Purpose: isolate whether single-teacher regression on v1.0.1 is caused by
#          dependency version mismatch vs code changes.
#
# Key difference from single-teacher-anchor-debug.sh:
#   - Pins transformers and trl to specific versions matching the original
#     fresh ms-swift clone environment
#   - Does NOT run `pip install transformers -U` (which could install incompatible versions)
#
# If this script produces improvement while single-teacher-anchor-debug.sh
# does not, the issue is environment/dependency mismatch, not code.
# ═══════════════════════════════════════════════════════════════════════════

# Install dependencies with PINNED versions
pip install python-dotenv dotenv==0.9.9 --no-deps 2>/dev/null || true
pip install -r requirements.txt
pip install -e .

# Pin to versions known to work with the original ms-swift fresh clone.
# IMPORTANT: Do NOT use -U here — we want exact versions.
# Adjust these versions to match what `pip list` shows on a working original-repo Arnold job.
pip install transformers==4.51.3 trl==0.28.0
pip install msgspec deepspeed vllm -U
pip install protobuf==3.20.3 --break-system-packages

# ─── Multi-node detection and setup (Merlin/Arnold) ───
nnodes=${ARNOLD_NUM:-1}
nproc_per_node=${ARNOLD_WORKER_GPU:-8}
export NNODES=$nnodes
export NODE_RANK=${ARNOLD_ID:-0}
export NPROC_PER_NODE=$nproc_per_node
export MASTER_ADDR=${MASTER_ADDR:=$ARNOLD_WORKER_0_HOST}
export MASTER_PORT=${MASTER_PORT:=$(echo "$ARNOLD_WORKER_0_PORT" | cut -d "," -f 1)}

# ─── Environment ───
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export WANDB_PROJECT="ms-swift-multi-teacher-gkd-upgrade"

echo "═══════════════════════════════════════════════════════════════"
echo "Single-Teacher GKD (anchor_memory) — PINNED DEPS"
echo "Nodes: $NNODES, GPUs/Node: $NPROC_PER_NODE, Total GPUs: $((NNODES * NPROC_PER_NODE))"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "═══════════════════════════════════════════════════════════════"

# Print installed versions for debugging
echo "Installed versions:"
pip show transformers trl vllm deepspeed 2>/dev/null | grep -E "^(Name|Version):"

# ─── Models ───
STUDENT_MODEL="/mnt/bn/youxiang-lf/models/AwemeLM6-1.7B-v4.7.0-live-lm-v1/checkpoint-45200"
ANCHOR_MEMORY_TEACHER="/mnt/bn/youxiang-lf/models/qwen3_8b_instruct-anchor-memory-teacher-200700/checkpoint-3136"

CUSTOM_DATASET_INFO="/mnt/bn/youxiang-lf/data/dataset_info_ms_swift.json"

run_name="Qwen3-1.7B-single-teacher-gkd-anchor-8b-200700-pinned-deps"
OUTPUT_DIR="/mnt/bn/youxiang-lf/models/$run_name"
export WANDB_NAME="$run_name"

mkdir -p "$OUTPUT_DIR"

STDOUT_LOG="stdout.log"
STDERR_LOG="stderr.log"

echo "Logging stdout to: $STDOUT_LOG"
echo "Logging stderr to: $STDERR_LOG"

# ─── Launch Training ───
swift rlhf \
    --rlhf_type gkd \
    --model "$STUDENT_MODEL" \
    --model_type qwen3 \
    --template qwen3 \
    --teacher_model "$ANCHOR_MEMORY_TEACHER" \
    --teacher_model_type qwen3 \
    --train_type full \
    --custom_dataset_info "$CUSTOM_DATASET_INFO" \
    --dataset anchor_memory_policy_sft200700-ms-swift \
    --split_dataset_ratio 0.01 \
    --seq_kd false \
    --lmbda 1 \
    --beta 0.9 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 4 \
    --eval_steps 10 \
    --save_steps 75 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --max_length 4096 \
    --max_completion_length 1024 \
    --output_dir "$OUTPUT_DIR" \
    --warmup_ratio 0.05 \
    --save_only_model true \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --deepspeed zero2 \
    --attn_impl sdpa \
    --teacher_deepspeed zero2 \
    --offload_teacher_model true \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.1 \
    --report_to wandb \
    --log_completions true \
    --seed 42 \
    --data_seed 42 \
    --run_name "$run_name" \
    --sleep_level 0 \
    > >(tee "$STDOUT_LOG") 2> >(tee "$STDERR_LOG" >&2)
