#!/usr/bin/env bash
set -euo pipefail

# Patch 4 / Step 3: mix original training data with filtered hard samples.

TRAIN_DATA=${TRAIN_DATA:-"/path/to/train.jsonl"}
FILTERED_DIR=${FILTERED_DIR:-"projects/ips/hard-mining/output/round1/filtered"}
HARD_POSITIVES=${HARD_POSITIVES:-"${FILTERED_DIR}/filtered_hard_positives.jsonl"}
HARD_NEGATIVES=${HARD_NEGATIVES:-"${FILTERED_DIR}/filtered_hard_negatives.jsonl"}
OUTPUT=${OUTPUT:-"projects/ips/hard-mining/output/stage3_train.jsonl"}
NUM_NEGATIVES=${NUM_NEGATIVES:-5}
NEGATIVE_MIX=${NEGATIVE_MIX:-"2:3:5"}
MAX_EXTRA_POSITIVES=${MAX_EXTRA_POSITIVES:-1}
SEED=${SEED:-42}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

python3 "${SCRIPT_DIR}/build_stage3_data.py" \
  --train_data "${TRAIN_DATA}" \
  --hard_positives "${HARD_POSITIVES}" \
  --hard_negatives "${HARD_NEGATIVES}" \
  --output "${OUTPUT}" \
  --num_negatives "${NUM_NEGATIVES}" \
  --negative_mix "${NEGATIVE_MIX}" \
  --max_extra_positives "${MAX_EXTRA_POSITIVES}" \
  --seed "${SEED}"
