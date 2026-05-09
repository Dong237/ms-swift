#!/bin/bash
# Encode Protocol-A evaluation manifest with a swift embedding checkpoint.
#
# Required:
#   MODEL=/mnt/bn/.../checkpoint-best
#
# Defaults:
#   MANIFEST=${OUTPUT_DIR}/eval_manifest.jsonl
#   OUTPUT=${OUTPUT_DIR}/eval_embeddings.jsonl
#
# Set REUSE_EXISTING=true to skip inference when OUTPUT already exists.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/encode_ip_eval_manifest.py"

OUTPUT_DIR=${OUTPUT_DIR:-"/mnt/bn/youxiang-hl/data/facial_ip/output_train_test_set/eval_protocol_a"}
MANIFEST=${MANIFEST:-"${OUTPUT_DIR}/eval_manifest.jsonl"}
OUTPUT=${OUTPUT:-"${OUTPUT_DIR}/eval_embeddings.jsonl"}
MODEL=${MODEL:-""}
MODEL_TYPE=${MODEL_TYPE:-"qwen3_5_emb"}
INSTRUCTION=${INSTRUCTION:-"提取该IP角色的身份特征，关注角色本身而非背景或姿态"}
DEVICE_BACKEND=${DEVICE_BACKEND:-"auto"}
NUM_DEVICES=${NUM_DEVICES:-""}
BATCH_SIZE=${BATCH_SIZE:-64}
TORCH_DTYPE=${TORCH_DTYPE:-"auto"}
REUSE_EXISTING=${REUSE_EXISTING:-false}

mkdir -p "${OUTPUT_DIR}"

args=(
  --manifest "${MANIFEST}"
  --output "${OUTPUT}"
  --model_type "${MODEL_TYPE}"
  --instruction "${INSTRUCTION}"
  --device_backend "${DEVICE_BACKEND}"
  --batch_size "${BATCH_SIZE}"
  --torch_dtype "${TORCH_DTYPE}"
)

if [ -n "${MODEL}" ]; then
  args+=(--model "${MODEL}")
fi
if [ -n "${NUM_DEVICES}" ]; then
  args+=(--num_devices "${NUM_DEVICES}")
fi
if [ "${REUSE_EXISTING}" = "true" ]; then
  args+=(--reuse_existing)
fi

python3 "${PY_SCRIPT}" "${args[@]}"
