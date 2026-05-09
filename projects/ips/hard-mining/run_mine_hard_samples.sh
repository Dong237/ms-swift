#!/usr/bin/env bash
set -euo pipefail

# Patch 4 / Step 1: encode vault and mine hard positive/negative candidates.

MODEL=${MODEL:-"/path/to/checkpoint"}
VAULT_DIR=${VAULT_DIR:-"/path/to/IP_image_train"}
VAULT_MANIFEST=${VAULT_MANIFEST:-""}
EMBEDDINGS=${EMBEDDINGS:-""}
OUTPUT_DIR=${OUTPUT_DIR:-"projects/ips/hard-mining/output/round1"}
MODEL_TYPE=${MODEL_TYPE:-"qwen3_5_emb"}
INSTRUCTION=${INSTRUCTION:-"提取图片中IP角色的视觉特征表示"}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_GPUS=${NUM_GPUS:-0}
TOP_K=${TOP_K:-100}
HARD_POSITIVE_PER_QUERY=${HARD_POSITIVE_PER_QUERY:-3}
HARD_NEGATIVE_PER_QUERY=${HARD_NEGATIVE_PER_QUERY:-10}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

CMD=(
  python3 "${SCRIPT_DIR}/mine_hard_samples.py"
  --output_dir "${OUTPUT_DIR}"
  --model_type "${MODEL_TYPE}"
  --instruction "${INSTRUCTION}"
  --batch_size "${BATCH_SIZE}"
  --top_k "${TOP_K}"
  --hard_positive_per_query "${HARD_POSITIVE_PER_QUERY}"
  --hard_negative_per_query "${HARD_NEGATIVE_PER_QUERY}"
  --infer_script "${REPO_ROOT}/aweme-docs/sh/infer_embedding_multimodal_swift.py"
)

if [[ -n "${EMBEDDINGS}" ]]; then
  CMD+=(--embeddings "${EMBEDDINGS}")
else
  CMD+=(--model "${MODEL}")
  if [[ -n "${VAULT_MANIFEST}" ]]; then
    CMD+=(--vault_manifest "${VAULT_MANIFEST}")
  else
    CMD+=(--vault_dir "${VAULT_DIR}")
  fi
fi

if [[ "${NUM_GPUS}" -gt 0 ]] 2>/dev/null; then
  CMD+=(--num_gpus "${NUM_GPUS}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
