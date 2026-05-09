#!/bin/bash
# ============================================================================
# Qwen3.5-9B IP Image Embedding Training — 6 nodes × 8 H20 GPUs (48 GPUs)
# ============================================================================
set -e

pip install -e .
pip install 'transformers<5.3.0,>=4.33' --break-system-packages
pip install qwen_vl_utils decord deepspeed -U

# ── Multi-node distributed setup (aligned with Stage 2 scripts) ──
export NPROC_PER_NODE=8
export NNODES=${ARNOLD_WORKER_NUM:-${WORLD_SIZE:-${NNODES:-6}}}
export NODE_RANK=${ARNOLD_ID:-${RANK:-${NODE_RANK:-0}}}
export MASTER_ADDR=${ARNOLD_WORKER_0_HOST:-${MASTER_ADDR:-}}
export MASTER_PORT=${ARNOLD_WORKER_0_PORT:-${MASTER_PORT:-29500}}

if [ "${NNODES}" -gt 1 ]; then
    if [ -z "${MASTER_ADDR}" ] || [ "${MASTER_ADDR}" = "127.0.0.1" ] || [ "${MASTER_ADDR}" = "localhost" ]; then
        echo "ERROR: NNODES=${NNODES} but MASTER_ADDR='${MASTER_ADDR}'"
        exit 1
    fi
fi

echo "=== Distributed config ==="
echo "NNODES=${NNODES}  NODE_RANK=${NODE_RANK}  NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "MASTER_ADDR=${MASTER_ADDR}  MASTER_PORT=${MASTER_PORT}"
echo "=========================="

# ── InfoNCE ──
export INFONCE_TEMPERATURE=0.05
export INFONCE_USE_BATCH=False

# ── Memory ──
export IMAGE_MAX_TOKEN_NUM=1024
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# ── Logging ──
export WANDB_PROJECT="qwen3-5-embedding-ip"
run_name="qwen3-5-9B-ip-embedding-stage1-merged-infonce"
export WANDB_NAME="$run_name"

# ── Paths ──
MODEL_PATH=${MODEL_PATH:-"/mnt/bn/youxiang-lf/models/Qwen3.5-9B"}
DATASET_PATH=${DATASET_PATH:-"/mnt/bn/youxiang-hl/data/facial_ip/output_train_test_set/training/train_merged_infonce.jsonl"}
OUTPUT_DIR="/mnt/bn/youxiang-lf/models/$run_name"

BATCH_SIZE=2
GRAD_ACCUM=1
LEARNING_RATE=6e-6
NUM_EPOCHS=3
MAX_LENGTH=1536

# Steps/epoch = 152,850 * 0.98 / 96 ≈ 1,560 (merged infonce data, 2% val split, 48 GPUs × batch 2)
# Eval twice per epoch, save every epoch
EVAL_STEPS=780
SAVE_STEPS=1560

swift sft \
    --model "${MODEL_PATH}" \
    --model_type qwen3_5_emb \
    --task_type embedding \
    --loss_type infonce \
    --system "提取该IP角色的身份特征，关注角色本身而非背景或姿态" \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --attn_impl sdpa \
    --dataset "${DATASET_PATH}" \
    --split_dataset_ratio 0.02 \
    --load_from_cache_file true \
    --learning_rate ${LEARNING_RATE} \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --adam_beta2 0.999 \
    --num_train_epochs ${NUM_EPOCHS} \
    --max_length ${MAX_LENGTH} \
    --truncation_strategy right \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --gradient_checkpointing true \
    --eval_strategy steps \
    --eval_steps ${EVAL_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit 5 \
    --logging_steps 5 \
    --dataloader_drop_last true \
    --dataloader_num_workers 8 \
    --dataset_num_proc 64 \
    --deepspeed zero2 \
    --output_dir "${OUTPUT_DIR}" \
    --report_to wandb

echo "=== Training finished at $(date) ==="
