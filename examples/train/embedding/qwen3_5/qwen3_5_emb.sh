# Qwen3.5 Embedding Training Example
# Uses EOS token pooling + InfoNCE contrastive learning
# Follows the Qwen3-Embedding approach adapted for Qwen3.5 hybrid architecture
#
# Data format: docs/source_en/BestPractices/Embedding.md
# --dataloader_drop_last must be true or eval gather will throw error
#
# Phase 1: Weakly supervised contrastive pre-training (large LR, large data)
# Phase 2: Supervised fine-tuning (small LR, high-quality data)

# === Phase 1: Weakly Supervised Pre-training ===
# Replace dataset path with your synthetic query-document pairs
CUDA_VISIBLE_DEVICES=0,1 \
INFONCE_TEMPERATURE=0.1 \
NPROC_PER_NODE=2 \
swift sft \
    --model Qwen/Qwen3.5-0.8B \
    --model_type qwen3_5_emb \
    --task_type embedding \
    --loss_type infonce \
    --tuner_type full \
    --learning_rate 1e-4 \
    --dataset sentence-transformers/stsb:positive \
    --attn_impl sdpa \
    --torch_dtype bfloat16 \
    --load_from_cache_file true \
    --split_dataset_ratio 0.02 \
    --eval_strategy steps \
    --output_dir output/qwen3_5_emb_phase1 \
    --save_steps 500 \
    --eval_steps 500 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --num_train_epochs 3 \
    --max_length 4096 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --warmup_ratio 0.05 \
    --dataloader_drop_last true \
    --deepspeed zero2

# === Phase 2: Supervised Fine-tuning ===
# Uncomment and modify for phase 2 training with high-quality labeled data
# Use smaller learning rate and enable hard negatives
#
# CUDA_VISIBLE_DEVICES=0,1 \
# INFONCE_TEMPERATURE=0.1 \
# INFONCE_INCLUDE_QQ=True \
# INFONCE_INCLUDE_DD=True \
# INFONCE_MASK_FAKE_NEGATIVE=True \
# NPROC_PER_NODE=2 \
# swift sft \
#     --model output/qwen3_5_emb_phase1/checkpoint-xxx \
#     --model_type qwen3_5_emb \
#     --task_type embedding \
#     --loss_type infonce \
#     --tuner_type full \
#     --learning_rate 1e-5 \
#     --dataset /path/to/your/high_quality_data.jsonl \
#     --attn_impl sdpa \
#     --torch_dtype bfloat16 \
#     --load_from_cache_file true \
#     --split_dataset_ratio 0.02 \
#     --eval_strategy steps \
#     --output_dir output/qwen3_5_emb_phase2 \
#     --save_steps 500 \
#     --eval_steps 500 \
#     --save_total_limit 3 \
#     --logging_steps 10 \
#     --num_train_epochs 2 \
#     --max_length 4096 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 2 \
#     --gradient_accumulation_steps 1 \
#     --dataloader_num_workers 4 \
#     --dataset_num_proc 4 \
#     --warmup_ratio 0.05 \
#     --dataloader_drop_last true \
#     --deepspeed zero2
