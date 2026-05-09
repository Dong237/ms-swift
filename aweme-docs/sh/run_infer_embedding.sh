#!/bin/bash
# Batch embedding inference for trained Qwen3-Embedding model
# Outputs JSONL files with "embedding" key added to each record

export http_proxy=http://sys-proxy-rd-relay.byted.org:8118 https_proxy=http://sys-proxy-rd-relay.byted.org:8118 no_proxy=.byted.org
pip3 install vllm
pip3 install -e .

# ====== CONFIGURE THESE ======
MODEL_PATH="/mnt/bn/youxiang-hl/models_emb_intent/Qwen3-EMB-0.6B-intent-l1-labels-text/checkpoint-XXX"  # <-- set your checkpoint
TRAIN_DATA="/path/to/train.jsonl"   # <-- set your train data path
TEST_DATA="/path/to/test.jsonl"     # <-- set your test data path
OUTPUT_DIR="/mnt/bn/youxiang-hl/models_emb_intent/Qwen3-EMB-0.6B-intent-l1-labels-text/embeddings"
TP_SIZE=${ARNOLD_WORKER_GPU:-1}     # auto-detect GPU count, or set manually
# ==============================

python3 aweme-docs/sh/infer_embedding_vllm.py \
    --model "$MODEL_PATH" \
    --input "$TRAIN_DATA" "$TEST_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --prompt_key prompt \
    --max_model_len 512 \
    --dtype bfloat16 \
    --tensor_parallel_size "$TP_SIZE"
