#!/bin/bash
# Multi-GPU smoke test for multi-teacher GKD with per-sample routing.
# Verifies no deadlocks when different ranks route to different teachers.
#
# Requirements: 2+ GPUs, ms-swift installed, Qwen2.5-0.5B downloaded.
# Usage: CUDA_VISIBLE_DEVICES=0,1 bash multi-teacher/test_multi_gpu.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${REPO_DIR}/output_multi_gpu_test"
DATA_FILE="${OUTPUT_DIR}/test_data.jsonl"

# Cleanup previous runs
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Create test data with alternating channels
python3 -c "
import json
with open('$DATA_FILE', 'w') as f:
    for i in range(100):
        ch = 'math' if i % 2 == 0 else 'code'
        sample = {
            'messages': [
                {'role': 'user', 'content': f'Question {i}'},
                {'role': 'assistant', 'content': f'Answer {i}'},
            ],
            'channel': ch,
        }
        f.write(json.dumps(sample) + '\n')
"

echo "=== Multi-GPU Multi-Teacher GKD Test ==="
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Data: $DATA_FILE"
echo ""

# Run with NPROC_PER_NODE=2 (2 GPUs)
NPROC_PER_NODE=2 swift rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen2.5-0.5B \
    --teacher_domain_map '{"math": "Qwen/Qwen2.5-0.5B", "code": "Qwen/Qwen2.5-0.5B"}' \
    --dataset "$DATA_FILE" \
    --split_dataset_ratio 0.0 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --save_steps 999 \
    --output_dir "$OUTPUT_DIR/checkpoint" \
    --deepspeed zero2

echo ""
echo "=== PASSED: Multi-GPU multi-teacher GKD completed without deadlocks ==="

# Cleanup
rm -rf "$OUTPUT_DIR"
