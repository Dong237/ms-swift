#!/bin/bash
# ============================================================================
# IP Embedding Retrieval Evaluation
# ============================================================================
# Two modes:
#   1. Full pipeline: embed test images + evaluate (needs GPU/NPU)
#   2. Re-evaluate:   load saved embeddings + compute metrics (CPU only)
#
# Toggle by setting MODE below.
# ============================================================================
set -e

# ── Mode: "full" or "reeval" ──
MODE=${MODE:-"full"}

# ── Model (only used in full mode) ──
MODEL_PATH=${MODEL_PATH:-"/mnt/bn/youxiang-lf/models/qwen3-5-0.8B-ip-embedding-64gpu/checkpoint-2870"}
MODEL_TYPE=${MODEL_TYPE:-"qwen3_5_emb"}          # qwen3_5_emb | qwen3_vl_emb
INSTRUCTION=${INSTRUCTION:-"提取该IP角色的身份特征，关注角色本身而非背景或姿态"}

# ── Data ──
TEST_DIR=${TEST_DIR:-"/mnt/bn/youxiang-lf/data/facial_ip/IP_image_val"}

# ── Output ──
OUTPUT_EMBEDDINGS=${OUTPUT_EMBEDDINGS:-"/mnt/bn/youxiang-lf/data/facial_ip/eval_output/eval_embeddings.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"/mnt/bn/youxiang-lf/data/facial_ip/eval_output"}

# ── Eval config ──
K=${K:-10}                    # Recall@K, CMC@K, mAP@K

# ── Inference speed (only used in full mode) ──
BATCH_SIZE=${BATCH_SIZE:-64}  # Per-GPU batch size. Lower if OOM.
NUM_GPUS=${NUM_GPUS:-""}      # Empty = auto-detect all available
NUM_WORKERS=${NUM_WORKERS:-8} # Folder scanning parallelism

# ── Build command ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ "$MODE" = "reeval" ]; then
    echo "=== Re-evaluate mode (skip inference, CPU only) ==="
    python "${SCRIPT_DIR}/eval_ip_retrieval.py" \
        --embeddings "${OUTPUT_EMBEDDINGS}" \
        --output_dir "${OUTPUT_DIR}" \
        --K ${K}
else
    echo "=== Full pipeline mode (embed + evaluate) ==="
    CMD="python ${SCRIPT_DIR}/eval_ip_retrieval.py \
        --model ${MODEL_PATH} \
        --model_type ${MODEL_TYPE} \
        --instruction \"${INSTRUCTION}\" \
        --test_dir ${TEST_DIR} \
        --output ${OUTPUT_EMBEDDINGS} \
        --output_dir ${OUTPUT_DIR} \
        --K ${K} \
        --batch_size ${BATCH_SIZE} \
        --num_workers ${NUM_WORKERS}"

    # Only pass --num_gpus if explicitly set
    if [ -n "${NUM_GPUS}" ]; then
        CMD="${CMD} --num_gpus ${NUM_GPUS}"
    fi

    eval ${CMD}
fi
