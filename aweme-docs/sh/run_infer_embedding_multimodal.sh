#!/bin/bash
# ============================================================================
# Qwen3.5 多模态 Embedding 批量推理启动脚本
#
# 用法:
#   bash aweme-docs/sh/run_infer_embedding_multimodal.sh
#
# 必须修改下方 MODEL 和 INPUT_FILES
# 其余参数可按需调整
# ============================================================================

set -euo pipefail


# ─── 必填：模型路径 ─────────────────────────────────────────────────────────
MODEL="/mnt/bn/youxiang-lf/models/Qwen3.5-0.8B"

# ─── 必填：输入文件（支持任意数量）─────────────────────────────────────────
INPUT_FILES=(
    "/mnt/bn/youxiang-lf/data/train.jsonl"
    "/mnt/bn/youxiang-lf/data/test.jsonl"
    # 按需添加更多文件:
    # "/mnt/bn/youxiang-lf/data/val.jsonl"
)

# ─── 可选：输出目录（留空则自动放在模型目录同级 embeddings/）─────────────────
OUTPUT_DIR=""

# ─── 可选：模型与推理参数 ──────────────────────────────────────────────────
MODEL_TYPE="qwen3_5_emb"          # qwen3_5_emb | qwen3_vl_emb
BATCH_SIZE=64                      # 每 GPU 每批样本数，OOM 时降低
NUM_GPUS=${ARNOLD_WORKER_GPU:-0}   # 0 = 自动检测全部可用 GPU

# ─── 可选：全局指令（引导 embedding 提取方向）──────────────────────────────
INSTRUCTION="根据输入的图文内容，提取语义特征表示"
# 不使用指令：INSTRUCTION=""
# 情感分类：  INSTRUCTION="提取该内容的情感特征表示"
# 意图识别：  INSTRUCTION="提取该内容的用户意图特征表示"

# ============================================================================
# 以下无需修改
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="${SCRIPT_DIR}/infer_embedding_multimodal_swift.py"

CMD=(
    python3 "$SCRIPT"
    --model "$MODEL"
    --input "${INPUT_FILES[@]}"
    --model_type "$MODEL_TYPE"
    --batch_size "$BATCH_SIZE"
    --instruction "$INSTRUCTION"
)

if [ -n "$OUTPUT_DIR" ]; then
    CMD+=(--output_dir "$OUTPUT_DIR")
fi

if [ "$NUM_GPUS" -gt 0 ] 2>/dev/null; then
    CMD+=(--num_gpus "$NUM_GPUS")
fi

echo "============================================"
echo "Model:       $MODEL"
echo "Input files: ${INPUT_FILES[*]}"
echo "Model type:  $MODEL_TYPE"
echo "Batch size:  $BATCH_SIZE"
echo "Instruction: ${INSTRUCTION:-(none)}"
echo "Num GPUs:    ${NUM_GPUS:-auto}"
echo "Output dir:  ${OUTPUT_DIR:-auto}"
echo "============================================"

"${CMD[@]}"
