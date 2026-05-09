#!/bin/bash
# Compute Protocol-A retrieval metrics from encoded eval embeddings.
#
# Protocol A:
#   query = all test images
#   gallery = train + test images
#   current query image_path is masked from gallery for that query
#
# Output:
#   ${RESULT_DIR}/eval_metrics.json
#   ${RESULT_DIR}/per_query_results.jsonl
#   ${RESULT_DIR}/failure_queries.jsonl
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/evaluate_ip_retrieval_protocol_a.py"

OUTPUT_DIR=${OUTPUT_DIR:-"/mnt/bn/youxiang-hl/data/facial_ip/output_train_test_set/eval_protocol_a"}
EMBEDDINGS=${EMBEDDINGS:-"${OUTPUT_DIR}/eval_embeddings.jsonl"}
RESULT_DIR=${RESULT_DIR:-"${OUTPUT_DIR}/results"}
K=${K:-10}
EXTRA_K=${EXTRA_K:-30}
QUERY_CHUNK_SIZE=${QUERY_CHUNK_SIZE:-256}
TOP_DETAIL_K=${TOP_DETAIL_K:-20}

mkdir -p "${RESULT_DIR}"

python3 "${PY_SCRIPT}" \
  --embeddings "${EMBEDDINGS}" \
  --output_dir "${RESULT_DIR}" \
  --K "${K}" \
  --extra_k "${EXTRA_K}" \
  --query_chunk_size "${QUERY_CHUNK_SIZE}" \
  --top_detail_k "${TOP_DETAIL_K}"
