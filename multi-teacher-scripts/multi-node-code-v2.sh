#!/bin/bash
set -euo pipefail

# Install dependencies (but NOT ms-swift from PyPI — use the local repo)
pip install python-dotenv dotenv==0.9.9 --no-deps 2>/dev/null || true
pip install -r requirements.txt
pip install -e .  # Install YOUR local ms-swift with multi-teacher support
# IMPORTANT: do NOT use -U on transformers — it can upgrade past compatible versions
# and break TRL/transformers interop (e.g. peft_config removal in transformers 5.3+).
# Version caps are defined in requirements/framework.txt.
pip install msgspec deepspeed vllm -U
pip install 'transformers>=4.33,<5.3.0' -U
pip install protobuf==3.20.3 --break-system-packages

# ═══════════════════════════════════════════════════════════════════════════
# Multi-Teacher GKD: Multi-Node Training — CODE + ANCHOR_MEMORY (v2)
# Platform: Merlin (Arnold) multi-node training
# Mode: colocate vLLM + multi-teacher with per-sample routing
#
# What changed from v1 (multi-node-code.sh):
#
#   --interleave true   (NEW, default=true)
#       Disables dataset shuffle so samples are concatenated in domain order:
#       all anchor_memory samples first, then all code samples (or vice versa).
#       This produces homogeneous batches where every sample in a batch belongs
#       to the same teacher domain. Without this, concat+shuffle creates ~96%
#       mixed-teacher batches — the high-KL code tokens dominate the gradient
#       and wash out the anchor_memory signal.
#       Set --interleave false to revert to the old shuffle behaviour.
#
#   --enable_weighted_domain_loss true   (NEW, default=true)
#       Instead of dividing total JSD loss by total tokens across all domains
#       (which lets high-KL domains dominate), this computes the token-averaged
#       JSD loss per domain and then takes the simple mean:
#           loss = (loss_code + loss_anchor_memory) / 2
#       This gives equal gradient weight to each domain regardless of how many
#       tokens or how high KL each domain has.
#       Set --enable_weighted_domain_loss false to revert to token-averaging.
#
#   --log_domain_routing true   (NEW, default=true)
#       Prints per-batch routing counts to stdout on rank 0 every step:
#           [Step 42] Routing: anchor_memory=4, code=4
#       Useful for confirming that --interleave is working (all-same-domain
#       batches should show only one channel) and that routing is correct.
#       Set --log_domain_routing false to silence this output.
#
# NOTE: --teacher_deepspeed zero3 is NOT allowed for multi-teacher
#       (per-sample routing causes AllGather deadlocks across ranks).
#       Use zero2 for teacher, or omit to let teachers run without DS.
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
echo "Multi-Teacher GKD Training on Merlin (v2)"
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
# Uncomment to apply different JSD beta or temperature per domain.
# Keys must be a subset of TEACHER_DOMAIN_MAP keys.
# Channels not listed fall back to the global --beta / --temperature.
#
# Example: anchor_memory is a specialised fine-tuned teacher, keep beta high
#          to stay close to teacher distribution; code uses vanilla Qwen3-8B,
#          reduce beta so student can deviate more from the prior.
# TEACHER_BETA_MAP='{"code": 0.7, "anchor_memory": 0.95}'
# TEACHER_TEMPERATURE_MAP='{"code": 1.0, "anchor_memory": 0.8}'

run_name="LiveLLM-1.7B-v1-multi-teacher-gkd-v1.1.0-code-8B-anchor-8b-200700"
OUTPUT_DIR="/mnt/bn/youxiang-lf/models/$run_name"
export WANDB_NAME="$run_name"

mkdir -p "$OUTPUT_DIR"

# ─── Log Files in Current Working Directory ───
# stdout.log  — training metrics + routing lines ([Step N] Routing: ...)
# stderr.log  — domain loss summary ([Step N] domain_loss/code = ...)
#               and all framework warnings/errors
STDOUT_LOG="stdout.log"
STDERR_LOG="stderr.log"

echo "Logging stdout to: $STDOUT_LOG  (includes per-step routing)"
echo "Logging stderr to: $STDERR_LOG  (includes per-domain loss summaries)"

# ─── Build optional per-teacher args ───
EXTRA_ARGS=""
if [ -n "${TEACHER_BETA_MAP:-}" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --teacher_beta_map $TEACHER_BETA_MAP"
fi
if [ -n "${TEACHER_TEMPERATURE_MAP:-}" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --teacher_temperature_map $TEACHER_TEMPERATURE_MAP"
fi

# ─── Launch Training ───
# v2 new args:
#   --interleave true              concat datasets in domain order; avoids ~96% mixed-domain batches caused by shuffle
#   --enable_weighted_domain_loss  false = global token-weighted loss (default); true = equal per-domain weighting (experimental)
#   --log_domain_routing true      print "Routing: code->[1]=4, anchor_memory->[0]=4" to stdout every step
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
    --interleave true \
    --enable_weighted_domain_loss false \
    --log_domain_routing true \
    $EXTRA_ARGS \
    > >(tee "$STDOUT_LOG") 2> >(tee "$STDERR_LOG" >&2)
