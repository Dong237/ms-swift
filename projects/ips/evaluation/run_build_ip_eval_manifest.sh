#!/bin/bash
# Build Protocol-A IP retrieval evaluation manifest from train/test merge JSON.
#
# Defaults use the current facial_ip train/test split JSONs:
#   TRAIN_JSON=/mnt/bn/youxiang-hl/data/facial_ip/output_train_test_set/train_set.json
#   TEST_JSON=/mnt/bn/youxiang-hl/data/facial_ip/output_train_test_set/test_set.json
#
# Output:
#   ${OUTPUT_DIR}/eval_manifest.jsonl
#   ${OUTPUT_DIR}/eval_manifest.jsonl.report.json
#
# Duplicate-folder conflicts inside each split are auto-resolved by keeping the
# standard IP with the largest folder count. Set STRICT_DUPLICATES=true to fail
# instead of auto-resolving.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/build_ip_eval_manifest.py"

TRAIN_JSON=${TRAIN_JSON:-"/mnt/bn/youxiang-hl/data/facial_ip/output_train_test_set/train_set.json"}
TEST_JSON=${TEST_JSON:-"/mnt/bn/youxiang-hl/data/facial_ip/output_train_test_set/test_set.json"}
OUTPUT_DIR=${OUTPUT_DIR:-"/mnt/bn/youxiang-hl/data/facial_ip/output_train_test_set/eval_protocol_a"}
OUTPUT_NAME=${OUTPUT_NAME:-"eval_manifest.jsonl"}
IMAGE_ROOT=${IMAGE_ROOT:-""}
NUM_WORKERS=${NUM_WORKERS:-64}
RECURSIVE=${RECURSIVE:-false}
STRICT_DUPLICATES=${STRICT_DUPLICATES:-false}

mkdir -p "${OUTPUT_DIR}"

args=(
  --train_json "${TRAIN_JSON}"
  --test_json "${TEST_JSON}"
  --output "${OUTPUT_DIR}/${OUTPUT_NAME}"
  --num_workers "${NUM_WORKERS}"
)

if [ -n "${IMAGE_ROOT}" ]; then
  args+=(--image_root "${IMAGE_ROOT}")
fi
if [ "${RECURSIVE}" = "true" ]; then
  args+=(--recursive)
fi
if [ "${STRICT_DUPLICATES}" = "true" ]; then
  args+=(--strict_duplicates)
fi

python3 "${PY_SCRIPT}" "${args[@]}"
