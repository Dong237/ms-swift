#!/bin/bash
# ============================================================================
# Stage 2: Multi-positive InfoNCE + Sub-center ArcFace
# Qwen3.5-9B — 6 nodes × 8 H20 GPUs (48 GPUs)
#
# Uses stage2_ip_embedding loss = multi_positive_infonce + lambda * subcenter_arcface
#
# IMPORTANT: This requires ip_id in every JSONL row.
#   Regenerate data with: --include_ip_id (or INCLUDE_IP_ID=true)
#   The ip_id_map.json sidecar file tells you SUBCENTER_NUM_CLASSES.
#
# Dataset: train_merged_multipos_p3.jsonl (WITH ip_id field)
# Starting from: Stage 1 infonce checkpoint
# ============================================================================
set -e

# ── Fail-fast: check required env vars and paths BEFORE pip install ──
STAGE1_CHECKPOINT=${STAGE1_CHECKPOINT:?"ERROR: Set STAGE1_CHECKPOINT to your Stage 1 output (e.g. /mnt/bn/.../checkpoint-XXXX)"}
DATASET_PATH=${DATASET_PATH:-"/mnt/bn/youxiang-hl/data/facial_ip/output_train_test_set/training/train_merged_multipos_p3.jsonl"}

if [ ! -e "${STAGE1_CHECKPOINT}" ]; then
    echo "ERROR: STAGE1_CHECKPOINT does not exist: ${STAGE1_CHECKPOINT}"
    exit 1
fi
if [ ! -f "${DATASET_PATH}" ]; then
    echo "ERROR: DATASET_PATH does not exist: ${DATASET_PATH}"
    exit 1
fi

# Verify ip_id is present in the dataset (required for stage2_ip_embedding)
python3 -c "
import json, sys
row = json.loads(open('${DATASET_PATH}', encoding='utf-8').readline())
if 'ip_id' not in row:
    print('ERROR: ip_id not found in first row of ${DATASET_PATH}')
    print('Regenerate with INCLUDE_IP_ID=true')
    print('Row keys:', sorted(row.keys()))
    sys.exit(1)
print('ip_id preflight OK: ip_id=' + str(row['ip_id']))
" || exit 1

pip install -e .
pip install 'transformers<5.3.0,>=4.33' --break-system-packages
pip install qwen_vl_utils decord deepspeed -U

# ── Multi-node distributed setup ──
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

# ── InfoNCE (Stage 2: lower temperature) ──
export INFONCE_TEMPERATURE=0.03
export INFONCE_USE_BATCH=False

# ── Sub-center ArcFace ──
# DATASET_PATH already set in fail-fast block above
# Auto-read num_classes from ip_id_map.json if not set manually
if [ -z "${SUBCENTER_NUM_CLASSES}" ]; then
    IP_ID_MAP="${DATASET_PATH}.ip_id_map.json"
    if [ -f "${IP_ID_MAP}" ]; then
        SUBCENTER_NUM_CLASSES=$(python3 -c "import json; print(json.load(open('${IP_ID_MAP}'))['num_classes'])")
        echo "Auto-detected SUBCENTER_NUM_CLASSES=${SUBCENTER_NUM_CLASSES} from ${IP_ID_MAP}"
    else
        echo "ERROR: SUBCENTER_NUM_CLASSES not set and ${IP_ID_MAP} not found."
        echo "Either set SUBCENTER_NUM_CLASSES or ensure ip_id_map.json exists next to the dataset."
        exit 1
    fi
fi
export SUBCENTER_NUM_CLASSES
export SUBCENTER_K=${SUBCENTER_K:-5}
export SUBCENTER_SCALE=${SUBCENTER_SCALE:-64}
export SUBCENTER_MARGIN=${SUBCENTER_MARGIN:-0.2}
export SUBCENTER_LAMBDA=${SUBCENTER_LAMBDA:-0.1}

# ── Memory ──
export IMAGE_MAX_TOKEN_NUM=1024
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# ── Logging ──
export WANDB_PROJECT="qwen3-5-embedding-ip"
run_name="qwen3-5-9B-ip-embedding-stage2-subcenter-k${SUBCENTER_K}-lam${SUBCENTER_LAMBDA}"
export WANDB_NAME="$run_name"

# ── Output ──
OUTPUT_DIR="/mnt/bn/youxiang-lf/models/$run_name"

# ── Hyperparameters ──
BATCH_SIZE=2
GRAD_ACCUM=1
LEARNING_RATE=2e-6
NUM_EPOCHS=3
MAX_LENGTH=1536

# Steps/epoch = 198,786 * 0.98 / 96 ≈ 2,029 (multipos p3 data, 2% val split, 48 GPUs × batch 2)
# Eval twice per epoch, save every epoch
EVAL_STEPS=1015
SAVE_STEPS=2029

echo "=== Sub-center config ==="
echo "SUBCENTER_NUM_CLASSES=${SUBCENTER_NUM_CLASSES}"
echo "SUBCENTER_K=${SUBCENTER_K}  SUBCENTER_SCALE=${SUBCENTER_SCALE}"
echo "SUBCENTER_MARGIN=${SUBCENTER_MARGIN}  SUBCENTER_LAMBDA=${SUBCENTER_LAMBDA}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "STAGE1_CHECKPOINT=${STAGE1_CHECKPOINT}"
echo "========================="

swift sft \
    --model "${STAGE1_CHECKPOINT}" \
    --model_type qwen3_5_emb \
    --task_type embedding \
    --loss_type stage2_ip_embedding \
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

echo "=== Stage 2 (multi-positive + sub-center) finished at $(date) ==="
