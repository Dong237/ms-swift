#!/usr/bin/env bash
set -euo pipefail

# Patch 4 / Step 2: filter mined candidates with rules and optional external scores.

INPUT_DIR=${INPUT_DIR:-"projects/ips/hard-mining/output/round1"}
OUTPUT_DIR=${OUTPUT_DIR:-"${INPUT_DIR}/filtered"}
HARD_POSITIVES=${HARD_POSITIVES:-"${INPUT_DIR}/hard_positives.jsonl"}
HARD_NEGATIVES=${HARD_NEGATIVES:-"${INPUT_DIR}/hard_negatives.jsonl"}
MIN_POSITIVE_SIM=${MIN_POSITIVE_SIM:-0.05}
MIN_NEGATIVE_SIM=${MIN_NEGATIVE_SIM:-0.30}
POSITIVE_SCORE_THRESHOLD=${POSITIVE_SCORE_THRESHOLD:-0.70}
NEGATIVE_SAME_IP_SCORE_MAX=${NEGATIVE_SAME_IP_SCORE_MAX:-0.30}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

python3 "${SCRIPT_DIR}/filter_hard_samples.py" \
  --hard_positives "${HARD_POSITIVES}" \
  --hard_negatives "${HARD_NEGATIVES}" \
  --output_dir "${OUTPUT_DIR}" \
  --min_positive_sim "${MIN_POSITIVE_SIM}" \
  --min_negative_sim "${MIN_NEGATIVE_SIM}" \
  --positive_score_threshold "${POSITIVE_SCORE_THRESHOLD}" \
  --negative_same_ip_score_max "${NEGATIVE_SAME_IP_SCORE_MAX}"
