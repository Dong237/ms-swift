#!/bin/bash
set -euo pipefail

# Install dependencies (but NOT ms-swift from PyPI — use the local repo)
pip install python-dotenv dotenv==0.9.9 --no-deps 2>/dev/null || true
pip install -r requirements.txt
pip install -e .  # Install YOUR local ms-swift with multi-teacher support
pip install msgspec transformers deepspeed vllm -U
pip install protobuf==3.20.3 --break-system-packages

# ═══════════════════════════════════════════════════════════════════════════
# Multi-Teacher GKD: v3 ABLATION — code + anchor_memory
# Purpose: reproduce main-branch behavior by DISABLING v2 features
#
# v2 features disabled:
#   --enable_weighted_domain_loss false  (use global token-weighted loss like main branch)
#   --interleave false                   (use shuffled mixed-domain batches like main branch)
#   --log_domain_routing true            (keep for observability)
#
# Expected result: should match main-branch performance where code domain
# improved from 10.71 → ~29 LiveCodeBenchV6. If it does, this confirms the
# v2 equal-domain-weighting was causing the regression.
# ═══════════════════════════════════════════════════════════════════════════

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
echo "Multi-Teacher GKD v3 ABLATION (v2 features OFF)"
echo "Nodes: $NNODES, GPUs/Node: $NPROC_PER_NODE, Total GPUs: $((NNODES * NPROC_PER_NODE))"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "═══════════════════════════════════════════════════════════════"

# ─── Models ───
STUDENT_MODEL="/mnt/bn/youxiang-lf/models/AwemeLM6-1.7B-v4.7.0-live-lm-v1/checkpoint-45200"
CODE_TEACHER="/mnt/bn/youxiang-lf/models/Qwen3-8B"
ANCHOR_MEMORY_TEACHER="/mnt/bn/youxiang-lf/models/qwen3_8b_instruct-anchor-memory-teacher-200700/checkpoint-3136"

CUSTOM_DATASET_INFO="/mnt/bn/youxiang-lf/data/dataset_info_ms_swift.json"

TEACHER_DOMAIN_MAP="{\"code\":\"${CODE_TEACHER}\",\"anchor_memory\":\"${ANCHOR_MEMORY_TEACHER}\"}"
TEACHER_TYPE_MAP='{"code":"qwen3","anchor_memory":"qwen3"}'

run_name="LiveLLM-1.7B-v1-multi-teacher-gkd-v1.1.0-code-8B-anchor-8b-200700-ablation"
OUTPUT_DIR="/mnt/bn/youxiang-lf/models/$run_name"
export WANDB_NAME="$run_name"

mkdir -p "$OUTPUT_DIR"

STDOUT_LOG="stdout.log"
STDERR_LOG="stderr.log"

echo "Logging stdout to: $STDOUT_LOG"
echo "Logging stderr to: $STDERR_LOG"

# ─── Launch Training ───
# v3 ablation: v2 features disabled to match main-branch behavior
#   --enable_weighted_domain_loss false  global token-weighted loss (main-branch default)
#   --interleave false                   shuffled mixed-domain batches (main-branch default)
#   --log_domain_routing true            keep routing visibility
swift rlhf \
    --rlhf_type gkd \
    --model "$STUDENT_MODEL" \
    --model_type qwen3 \
    --template qwen3 \
    --teacher_domain_map "$TEACHER_DOMAIN_MAP" \
    --teacher_type_map "$TEACHER_TYPE_MAP" \
    --train_type full \
    --custom_dataset_info "$CUSTOM_DATASET_INFO" \
    --dataset anchor_memory_policy_sft200700-ms-swift code_master_gkd_10w_livecodebench \
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
    --enable_weighted_domain_loss false \
    --interleave false \
    --log_domain_routing true \
    > >(tee "$STDOUT_LOG") 2> >(tee "$STDERR_LOG" >&2)
