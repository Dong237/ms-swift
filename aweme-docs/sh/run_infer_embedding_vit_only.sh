#!/bin/bash
# ============================================================================
# Qwen3.5 ViT-Only Embedding 批量推理启动脚本
#
# 直接从 ViT 视觉编码器提取 embedding，跳过 LLM 层
# 用于与 full-pipeline (EOS pooling) 做性能对比
#
# 用法:
#   bash aweme-docs/sh/run_infer_embedding_vit_only.sh
# ============================================================================

set -euo pipefail

# ─── 必填：模型路径 ─────────────────────────────────────────────────────────
MODEL="/mnt/bn/youxiang-lf/models/Qwen3.5-0.8B"

# ─── 必填：输入文件（支持任意数量）─────────────────────────────────────────
INPUT_FILES=(
    "/mnt/bn/youxiang-lf/data/train.jsonl"
    "/mnt/bn/youxiang-lf/data/test.jsonl"
)

# ─── 可选参数 ──────────────────────────────────────────────────────────────
OUTPUT_DIR=""              # 留空则自动放在模型目录同级 embeddings_vit/
BATCH_SIZE=128             # ViT 很小，可以开大 batch
NUM_GPUS=${ARNOLD_WORKER_GPU:-0}  # 0 = 自动检测
POOLING="mean"             # mean | max
DTYPE="bfloat16"           # bfloat16 | float16 | float32 (V100 用 float16)

# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="${SCRIPT_DIR}/infer_embedding_vit_only.py"

CMD=(
    python3 "$SCRIPT"
    --model "$MODEL"
    --input "${INPUT_FILES[@]}"
    --batch_size "$BATCH_SIZE"
    --pooling "$POOLING"
    --dtype "$DTYPE"
)

if [ -n "$OUTPUT_DIR" ]; then
    CMD+=(--output_dir "$OUTPUT_DIR")
fi

if [ "$NUM_GPUS" -gt 0 ] 2>/dev/null; then
    CMD+=(--num_gpus "$NUM_GPUS")
fi

echo "============================================"
echo "Mode:        ViT-only (no LLM layers)"
echo "Model:       $MODEL"
echo "Input files: ${INPUT_FILES[*]}"
echo "Batch size:  $BATCH_SIZE"
echo "Pooling:     $POOLING"
echo "Dtype:       $DTYPE"
echo "Num GPUs:    ${NUM_GPUS:-auto}"
echo "Output dir:  ${OUTPUT_DIR:-auto}"
echo "============================================"

"${CMD[@]}"
