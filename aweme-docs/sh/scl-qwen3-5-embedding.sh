#!/bin/bash
# =============================================================================
# Qwen3-VL-Embedding Training Script (Single Node, 8 GPUs)
# =============================================================================
# Usage:
#   bash train_single_node.sh
#
# Configuration via environment variables:
#   DATASET_PATH=/path/to/train_multimodal.jsonl
#   MODEL_PATH=Qwen/Qwen3-VL-Embedding-2B  (or local path)
#   OUTPUT_DIR=output/qwen3-vl-emb-multimodal
# =============================================================================

set -e

# --- Logging setup ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
STDOUT_LOG="${LOG_DIR}/train_${TIMESTAMP}.stdout.log"
STDERR_LOG="${LOG_DIR}/train_${TIMESTAMP}.stderr.log"

# Redirect stdout and stderr to separate files (while still printing to console)
exec > >(tee "${STDOUT_LOG}") 2> >(tee "${STDERR_LOG}" >&2)

echo "=== Training started at $(date) ==="
echo "stdout log: ${STDOUT_LOG}"
echo "stderr log: ${STDERR_LOG}"

pip install -e .
pip install 'transformers<5.3.0,>=4.33' --break-system-packages
pip install qwen_vl_utils decord deepspeed -U

# Configuration
export NPROC_PER_NODE=${NPROC_PER_NODE:-2}

# InfoNCE settings
export INFONCE_TEMPERATURE=${INFONCE_TEMPERATURE:-0.1}
export INFONCE_USE_BATCH=${INFONCE_USE_BATCH:-True}

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# WandB logging
export WANDB_PROJECT="qwen3-5-embedding"
run_name="qwen3-5-embedding-emotion-label-text-only-initial-test" # also the output dir
export WANDB_NAME="$run_name"

# Model and data paths
MODEL_PATH=${MODEL_PATH:-"/mnt/bn/youxiang-lf/models/Qwen3.5-0.8B-Base"}
DATASET_PATH=${DATASET_PATH:-"/mnt/bn/youxiang-lf/data/contrastive_ablution_data_pretrain/emotion/stage2_train/train_label_text_only.jsonl"}
OUTPUT_DIR="/mnt/bn/youxiang-lf/models/$run_name"

# Training hyperparameters
BATCH_SIZE=${BATCH_SIZE:-32}
GRAD_ACCUM=${GRAD_ACCUM:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
NUM_EPOCHS=${NUM_EPOCHS:-5}
MAX_LENGTH=${MAX_LENGTH:-4096}

swift sft \
    --model "${MODEL_PATH}" \
    --model_type qwen3_5_emb \
    --task_type embedding \
    --loss_type infonce \
    --dataset "${DATASET_PATH}" \
    --tuner_type full \
    --learning_rate ${LEARNING_RATE} \
    --attn_impl sdpa \
    --torch_dtype bfloat16 \
    --load_from_cache_file true \
    --split_dataset_ratio 0.02 \
    --eval_strategy steps \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --num_train_epochs ${NUM_EPOCHS} \
    --max_length ${MAX_LENGTH} \
    --save_steps 500 \
    --eval_steps 500 \
    --save_total_limit 3 \
    --logging_steps 5 \
    --warmup_ratio 0.05 \
    --dataloader_drop_last true \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --deepspeed zero2 \
    --output_dir "${OUTPUT_DIR}" \
    --report_to wandb

echo "=== Training finished at $(date) ==="

#     --padding_free false \
#     --target_modules all-linear \