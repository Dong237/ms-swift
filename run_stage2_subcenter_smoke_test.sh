#!/bin/bash
# ============================================================================
# Stage 2 Sub-center ArcFace Smoke Test
# Qwen3.5-9B IP Embedding — multi-positive InfoNCE + Sub-center ArcFace
#
# Purpose:
#   This script is for validating that the upgraded training code path works:
#     - loss_type=stage2_ip_embedding is registered and importable
#     - multi-positive labels can be parsed
#     - Sub-center ArcFace proxies can be created
#     - one short distributed training run can enter forward/backward
#
# This is NOT a real Stage 2 continuation script. It defaults to MODEL_PATH as
# the base model so you can quickly test the loss plumbing. For real Stage 2,
# set MODEL_PATH/STAGE1_CHECKPOINT to a Stage 1 checkpoint.
#
# Recommended usage:
#   bash /private/tmp/run_stage2_subcenter_smoke_test.sh
#
# Optional overrides:
#   MODEL_PATH=/path/to/stage1/checkpoint \
#   DATASET_PATH=/path/to/train_merged_multipos_p3.jsonl \
#   MAX_STEPS=20 \
#   SUBCENTER_K=3 \
#   SUBCENTER_LAMBDA=0.05 \
#   bash /private/tmp/run_stage2_subcenter_smoke_test.sh
# ============================================================================
set -euxo pipefail
trap 'echo "FAILED at line ${LINENO}: ${BASH_COMMAND} (exit=$?)" >&2' ERR

echo "=== USER SCRIPT START ==="
date
hostname
pwd
whoami
git rev-parse --show-toplevel || true
git rev-parse --abbrev-ref HEAD || true
git rev-parse --short HEAD || true
echo "========================="

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_PATH=${MODEL_PATH:-"/mnt/bn/youxiang-lf/models/Qwen3.5-9B"}
DATASET_PATH=${DATASET_PATH:-"/mnt/bn/youxiang-lf/data/facial_ip/output_train_test_set/training/train_merged_multipos_p3.jsonl"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"/mnt/bn/youxiang-lf/models_emb_ip"}

if [ ! -e "${MODEL_PATH}" ]; then
    echo "ERROR: MODEL_PATH does not exist: ${MODEL_PATH}"
    exit 1
fi
if [ ! -f "${DATASET_PATH}" ]; then
    echo "ERROR: DATASET_PATH does not exist: ${DATASET_PATH}"
    exit 1
fi

echo "=== Preflight paths ==="
echo "MODEL_PATH=${MODEL_PATH}"
echo "DATASET_PATH=${DATASET_PATH}"
ls -ld "${MODEL_PATH}"
ls -lh "${DATASET_PATH}"
ls -lh "${DATASET_PATH}.ip_id_map.json" || true
head -n 1 "${DATASET_PATH}" | cut -c 1-800
echo "======================="

# ── Install local code and runtime deps ───────────────────────────────────────
pip install -e .
pip install 'transformers<5.3.0,>=4.33' --break-system-packages
pip install qwen_vl_utils decord deepspeed -U

# ── Code/data preflight after editable install ────────────────────────────────
python3 - <<PY
import json
import os
import swift
from swift.loss.mapping import loss_map

dataset = "${DATASET_PATH}"
ip_map = dataset + ".ip_id_map.json"

print("swift imported from:", swift.__file__)
print("has multi_positive_infonce:", "multi_positive_infonce" in loss_map)
print("has stage2_ip_embedding:", "stage2_ip_embedding" in loss_map)
if "stage2_ip_embedding" not in loss_map:
    raise SystemExit("ERROR: stage2_ip_embedding is not registered in loss_map.")

with open(dataset, encoding="utf-8") as f:
    row = json.loads(f.readline())

print("first row keys:", sorted(row.keys()))
if "ip_id" not in row:
    raise SystemExit("ERROR: ip_id not found in first row. Regenerate data with INCLUDE_IP_ID=true.")
if "positive_images" not in row or len(row["positive_images"]) < 1:
    raise SystemExit("ERROR: positive_images missing or empty.")
print("first row ip_id:", row["ip_id"])
print("first row positives:", len(row["positive_images"]))
print("first row negatives:", len(row.get("negative_images", [])))

if not os.path.isfile(ip_map):
    raise SystemExit(f"ERROR: ip_id_map file not found: {ip_map}")
with open(ip_map, encoding="utf-8") as f:
    mapping = json.load(f)
print("num_classes:", mapping["num_classes"])
PY

# ── Multi-node distributed setup ──────────────────────────────────────────────
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export NNODES=${ARNOLD_WORKER_NUM:-${NNODES:-6}}
export NODE_RANK=${ARNOLD_ID:-${RANK:-${NODE_RANK:-0}}}
export MASTER_ADDR=${ARNOLD_WORKER_0_HOST:-${MASTER_ADDR:-}}
export MASTER_PORT=${ARNOLD_WORKER_0_PORT:-${MASTER_PORT:-29500}}

if [ "${NNODES}" -gt 1 ]; then
    if [ -z "${MASTER_ADDR}" ] || [ "${MASTER_ADDR}" = "127.0.0.1" ] || [ "${MASTER_ADDR}" = "localhost" ]; then
        echo "ERROR: NNODES=${NNODES} but MASTER_ADDR='${MASTER_ADDR}'"
        env | grep -iE "arnold|master|worker|rank|host" | sort || true
        exit 1
    fi
fi

echo "=== Distributed config ==="
echo "NNODES=${NNODES}  NODE_RANK=${NODE_RANK}  NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "MASTER_ADDR=${MASTER_ADDR}  MASTER_PORT=${MASTER_PORT}"
env | grep -iE "arnold|master|worker|rank|host" | sort || true
echo "=========================="

# ── InfoNCE config ────────────────────────────────────────────────────────────
export INFONCE_TEMPERATURE=${INFONCE_TEMPERATURE:-0.03}
export INFONCE_USE_BATCH=${INFONCE_USE_BATCH:-False}

# ── Sub-center ArcFace config ─────────────────────────────────────────────────
if [ -z "${SUBCENTER_NUM_CLASSES:-}" ]; then
    IP_ID_MAP="${DATASET_PATH}.ip_id_map.json"
    SUBCENTER_NUM_CLASSES=$(python3 -c "import json; print(json.load(open('${IP_ID_MAP}', encoding='utf-8'))['num_classes'])")
fi
export SUBCENTER_NUM_CLASSES
export SUBCENTER_K=${SUBCENTER_K:-3}
export SUBCENTER_SCALE=${SUBCENTER_SCALE:-64}
export SUBCENTER_MARGIN=${SUBCENTER_MARGIN:-0.2}
export SUBCENTER_LAMBDA=${SUBCENTER_LAMBDA:-0.05}

# ── Runtime config ────────────────────────────────────────────────────────────
export IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM:-1024}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-'expandable_segments:True'}

export WANDB_PROJECT=${WANDB_PROJECT:-"qwen3-5-embedding-ip"}
run_name=${RUN_NAME:-"qwen3-5-9B-ip-embedding-stage2-subcenter-smoke-k${SUBCENTER_K}-lam${SUBCENTER_LAMBDA}"}
export WANDB_NAME="$run_name"
OUTPUT_DIR="${OUTPUT_ROOT}/${run_name}"

# ── Smoke-test hyperparameters ────────────────────────────────────────────────
BATCH_SIZE=${BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-1}
LEARNING_RATE=${LEARNING_RATE:-2e-6}
MAX_LENGTH=${MAX_LENGTH:-1536}
MAX_STEPS=${MAX_STEPS:-20}

# Keep eval/save steps larger than MAX_STEPS so the smoke test focuses on train.
EVAL_STEPS=${EVAL_STEPS:-100000}
SAVE_STEPS=${SAVE_STEPS:-100000}

echo "=== Sub-center smoke config ==="
echo "SUBCENTER_NUM_CLASSES=${SUBCENTER_NUM_CLASSES}"
echo "SUBCENTER_K=${SUBCENTER_K}"
echo "SUBCENTER_SCALE=${SUBCENTER_SCALE}"
echo "SUBCENTER_MARGIN=${SUBCENTER_MARGIN}"
echo "SUBCENTER_LAMBDA=${SUBCENTER_LAMBDA}"
echo "INFONCE_TEMPERATURE=${INFONCE_TEMPERATURE}"
echo "INFONCE_USE_BATCH=${INFONCE_USE_BATCH}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "==============================="

swift sft \
    --model "${MODEL_PATH}" \
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
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --adam_beta2 0.999 \
    --max_steps ${MAX_STEPS} \
    --max_length ${MAX_LENGTH} \
    --truncation_strategy right \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --gradient_checkpointing true \
    --eval_strategy steps \
    --eval_steps ${EVAL_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit 1 \
    --logging_steps 1 \
    --dataloader_drop_last true \
    --dataloader_num_workers 8 \
    --dataset_num_proc 64 \
    --deepspeed zero2 \
    --output_dir "${OUTPUT_DIR}" \
    --report_to wandb

echo "=== Stage 2 sub-center smoke test finished at $(date) ==="
