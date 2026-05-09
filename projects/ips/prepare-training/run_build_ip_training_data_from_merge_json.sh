#!/bin/bash
# ============================================================================
# Build IP embedding training JSONL from merged standard-IP annotation JSON.
#
# Background:
#   The source annotation JSON reflects the latest IP merge policy. Each top-level
#   key is a standard IP name, and its "文件夹列表" can contain one or more legacy
#   image folders. Folders listed under the same standard IP are treated as the
#   same class, so they can be sampled as positives for each other and will never
#   be sampled as negatives for each other.
#
# What this script produces:
#   - A ms-swift embedding JSONL dataset with:
#       messages/images
#       positive_messages/positive_images
#       negative_messages/negative_images
#   - A sidecar <output>.report.json with scan/build/dedup statistics.
#   - A sidecar <output>.ip_id_map.json with deterministic standard-IP ids.
#
# Loss/data modes:
#   LOSS_TYPE=infonce:
#     Build old Stage-1 style data. Each row has exactly one positive image.
#
#   LOSS_TYPE=multi_positive_infonce:
#     Build Stage-2 multi-positive data. Each row has up to NUM_POSITIVES
#     same-IP positives. This is the data format consumed by the Patch-1
#     multi_positive_infonce loss.
#
#   LOSS_TYPE=supcon:
#     Alias for multi_positive_infonce in data construction. Training should
#     still use --loss_type multi_positive_infonce because that is the registered
#     ms-swift loss name.
#
# Defaults:
#   MD5_DEDUP=true:
#     Deduplicate exact same image bytes inside each standard IP after folders
#     are merged. This removes repeated physical files without changing class
#     semantics.
#
#   INCLUDE_IP_ID=true:
#     Write ip_id/ip_name into each JSONL row. ip_id is assigned by sorted
#     standard IP name and is intended for Patch-2/Patch-3 Sub-center ArcFace
#     proxy indexing. Current infonce/multi_positive_infonce training ignores
#     these fields until the framework is patched to preserve them.
#
# Important variables:
#   MERGE_JSON            Path to merged annotation JSON.
#   OUTPUT_DIR            Directory for generated JSONL and sidecars.
#   OUTPUT_NAME           Optional output filename. If empty, inferred.
#   LOSS_TYPE             infonce | multi_positive_infonce | supcon.
#   NUM_POSITIVES         Positives per row for multi-positive data.
#   NUM_NEGATIVES         Explicit negatives per row.
#   MAX_SAMPLES_PER_IP    Max rows generated per standard IP.
#   NUM_WORKERS           Parallel folder scan / MD5 dedup workers.
#   BUILD_BOTH=true       Generate both comparison datasets in one run.
#
# Typical use:
#   MERGE_JSON=/mnt/bn/youxiang-hl/data/facial_ip/ip_merge.json \
#   OUTPUT_DIR=/mnt/bn/youxiang-lf/data/facial_ip/train_test_data \
#   LOSS_TYPE=multi_positive_infonce \
#   NUM_POSITIVES=3 \
#   bash projects/ips/prepare-training/run_build_ip_training_data_from_merge_json.sh
#
# Set BUILD_BOTH=true to generate both:
#   1. merged InfoNCE data with one positive per row
#   2. merged multi-positive data for multi_positive_infonce
#
# Example:
#   MERGE_JSON=/mnt/bn/youxiang-hl/data/facial_ip/ip_merge.json \
#   OUTPUT_DIR=/mnt/bn/youxiang-lf/data/facial_ip/train_test_data \
#   BUILD_BOTH=true \
#   NUM_POSITIVES=3 \
#   NUM_NEGATIVES=20 \
#   MAX_SAMPLES_PER_IP=25 \
#   bash projects/ips/prepare-training/run_build_ip_training_data_from_merge_json.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/build_ip_training_data_from_merge_json.py"

# Input / output
MERGE_JSON=${MERGE_JSON:-"/mnt/bn/youxiang-hl/data/facial_ip/ip_merge.json"}
OUTPUT_DIR=${OUTPUT_DIR:-"/mnt/bn/youxiang-lf/data/facial_ip/train_test_data"}
OUTPUT_NAME=${OUTPUT_NAME:-""}
IMAGE_ROOT=${IMAGE_ROOT:-""}

# Data mode
LOSS_TYPE=${LOSS_TYPE:-"multi_positive_infonce"}  # infonce | multi_positive_infonce | supcon
NUM_POSITIVES=${NUM_POSITIVES:-3}
NUM_NEGATIVES=${NUM_NEGATIVES:-20}
MAX_SAMPLES_PER_IP=${MAX_SAMPLES_PER_IP:-25}
MIN_IMAGES=${MIN_IMAGES:-2}
SEED=${SEED:-42}

# Runtime
NUM_WORKERS=${NUM_WORKERS:-64}
MD5_DEDUP=${MD5_DEDUP:-true}
RECURSIVE=${RECURSIVE:-false}
INCLUDE_IP_ID=${INCLUDE_IP_ID:-true}
INCLUDE_METADATA=${INCLUDE_METADATA:-false}
STRICT_DUPLICATES=${STRICT_DUPLICATES:-false}
BUILD_BOTH=${BUILD_BOTH:-false}

mkdir -p "${OUTPUT_DIR}"

common_args=(
  --merge_json "${MERGE_JSON}"
  --num_negatives "${NUM_NEGATIVES}"
  --max_samples_per_ip "${MAX_SAMPLES_PER_IP}"
  --min_images "${MIN_IMAGES}"
  --num_workers "${NUM_WORKERS}"
  --seed "${SEED}"
)

if [ -n "${IMAGE_ROOT}" ]; then
  common_args+=(--image_root "${IMAGE_ROOT}")
fi
if [ "${MD5_DEDUP}" = "true" ]; then
  common_args+=(--md5_dedup)
fi
if [ "${RECURSIVE}" = "true" ]; then
  common_args+=(--recursive)
fi
if [ "${INCLUDE_IP_ID}" = "true" ]; then
  common_args+=(--include_ip_id)
fi
if [ "${INCLUDE_METADATA}" = "true" ]; then
  common_args+=(--include_metadata)
fi
if [ "${STRICT_DUPLICATES}" = "true" ]; then
  common_args+=(--strict_duplicates)
fi

run_one() {
  local loss_type="$1"
  local num_positives="$2"
  local output="$3"

  echo "================================================================"
  echo "Building IP embedding data"
  echo "  merge_json:      ${MERGE_JSON}"
  echo "  output:          ${output}"
  echo "  loss_type:       ${loss_type}"
  echo "  num_positives:   ${num_positives}"
  echo "  num_negatives:   ${NUM_NEGATIVES}"
  echo "  md5_dedup:       ${MD5_DEDUP}"
  echo "  include_ip_id:   ${INCLUDE_IP_ID}"
  echo "================================================================"

  python3 "${PY_SCRIPT}" \
    "${common_args[@]}" \
    --output "${output}" \
    --loss_type "${loss_type}" \
    --num_positives "${num_positives}"
}

if [ "${BUILD_BOTH}" = "true" ]; then
  run_one "infonce" "1" "${OUTPUT_DIR}/train_merged_infonce.jsonl"
  run_one "multi_positive_infonce" "${NUM_POSITIVES}" "${OUTPUT_DIR}/train_merged_multipos_p${NUM_POSITIVES}.jsonl"
else
  if [ -z "${OUTPUT_NAME}" ]; then
    if [ "${LOSS_TYPE}" = "infonce" ]; then
      OUTPUT_NAME="train_merged_infonce.jsonl"
    else
      OUTPUT_NAME="train_merged_multipos_p${NUM_POSITIVES}.jsonl"
    fi
  fi
  run_one "${LOSS_TYPE}" "${NUM_POSITIVES}" "${OUTPUT_DIR}/${OUTPUT_NAME}"
fi
