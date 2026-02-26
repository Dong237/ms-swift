#!/bin/bash
set -euo pipefail

# Install dependencies (but NOT ms-swift from PyPI — use the local repo)
pip install python-dotenv dotenv==0.9.9 --no-deps 2>/dev/null || true
pip install -r requirements.txt
pip install -e .  # Install YOUR local ms-swift with multi-teacher support
pip install msgspec transformers deepspeed vllm -U
# Do NOT force-downgrade protobuf — vllm needs protobuf >= 5.29.6
# The byted-wandb protobuf warnings are non-fatal.



# ═══════════════════════════════════════════════════════════════════════════
# Multi-Teacher GKD: Multi-Node Training
# Platform: Merlin (Arnold) multi-node training
# Mode: colocate vLLM + multi-teacher with per-sample routing
#
# NOTE: --teacher_deepspeed zero3 is NOT allowed for multi-teacher
#       (per-sample routing causes AllGather deadlocks across ranks).
#       Use zero2 for teacher, or omit to let teachers run without DS.

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
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python  # byted-wandb/databus compat with protobuf 6.x
export WANDB_PROJECT="ms-swift-multi-teacher-gkd-upgrade"

echo "═══════════════════════════════════════════════════════════════"
echo "Multi-Teacher GKD Training on Merlin"
echo "Nodes: $NNODES, GPUs/Node: $NPROC_PER_NODE, Total GPUs: $((NNODES * NPROC_PER_NODE))"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "═══════════════════════════════════════════════════════════════"

# ─── Models ───
STUDENT_MODEL="/mnt/bn/youxiang-lf/models/AwemeLM6-1.7B-v4.7.0-live-lm-v1/checkpoint-45200"
CODE_TEACHER="/mnt/bn/youxiang-lf/models/Qwen3-8B"
ANCHOR_MEMORY_TEACHER="/mnt/bn/youxiang-lf/models/qwen3_8b_instruct-anchor-memory-teacher-200700/checkpoint-3136"

CUSTOM_DATASET_INFO="/mnt/bn/youxiang-lf/data/dataset_info_ms_swift.json"

# NOTE: single-line valid JSON, no trailing commas
TEACHER_DOMAIN_MAP="{\"code\":\"${CODE_TEACHER}\",\"anchor_memory\":\"${ANCHOR_MEMORY_TEACHER}\"}"
TEACHER_TYPE_MAP='{"code":"qwen3","anchor_memory":"qwen3"}'

# ─── Per-teacher hyperparameters (optional) ───
# Uncomment to set per-channel beta/temperature for JSD loss.
# Channels not listed fall back to the global --beta / --temperature.
# TEACHER_BETA_MAP='{"math": 0.9, "anchor_memory": 0.5}'
# TEACHER_TEMPERATURE_MAP='{"math": 0.7, "anchor_memory": 1.0}'

run_name="Qwen3-1.7B-multi-teacher-gkd-code-8B-anchor-8b-200700"
OUTPUT_DIR="/mnt/bn/youxiang-lf/models/$run_name"
export WANDB_NAME="$run_name"

mkdir -p "$OUTPUT_DIR"

# ─── Log Files in Current Working Directory ───
STDOUT_LOG="stdout.log"
STDERR_LOG="stderr.log"

echo "Logging stdout to: $STDOUT_LOG"
echo "Logging stderr to: $STDERR_LOG"

# ─── Build optional per-teacher args ───
EXTRA_ARGS=""
if [ -n "${TEACHER_BETA_MAP:-}" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --teacher_beta_map $TEACHER_BETA_MAP"
fi
if [ -n "${TEACHER_TEMPERATURE_MAP:-}" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --teacher_temperature_map $TEACHER_TEMPERATURE_MAP"
fi

# ─── Launch Training ───
swift rlhf \
    --rlhf_type gkd \
    --model "$STUDENT_MODEL" \
    --model_type qwen3 \
    --template qwen3 \
    --teacher_domain_map "$TEACHER_DOMAIN_MAP" \
    --teacher_type_map "$TEACHER_TYPE_MAP" \
    --train_type full \
    --custom_dataset_info "$CUSTOM_DATASET_INFO" \
    --dataset anchor_memory_policy_sft200700_sample_multi_teacher_test-ms-swift code_master_gkd_10w_livecodebench\
    --split_dataset_ratio 0.01 \
    --seq_kd false \
    --lmbda 1 \
    --beta 0.9 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 4 \
    --eval_steps 10 \
    --save_steps 781 \
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
    $EXTRA_ARGS \
    > >(tee "$STDOUT_LOG") 2> >(tee "$STDERR_LOG" >&2)
