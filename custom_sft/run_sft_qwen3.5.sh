#!/bin/bash
# =============================================================================
# SFT Training Script for Qwen3.5 (Dense/MoE) using ms-swift
# =============================================================================
# Usage:
#   1. Edit dataset_info.json: set "dataset_path" to your .jsonl file
#   2. Set MODEL_PATH below to your local model path or HuggingFace/ModelScope ID
#   3. Set DATASET_NAME to match the dataset_path in your dataset_info.json
#   4. Run: bash run_sft_qwen3.5.sh
#
# Your training data (.jsonl) should be in this format (one JSON per line):
#   {"messages": [{"role": "user", "content": "your question"}, {"role": "assistant", "content": "your answer"}]}
#   {"messages": [{"role": "system", "content": "system prompt"}, {"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]}
#
# Argument reference:
#   - Full docs:     https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html
#   - Model args:    swift/arguments/base_args/model_args.py
#   - Data args:     swift/arguments/base_args/data_args.py
#   - Template args: swift/arguments/base_args/template_args.py
#   - Tuner args:    swift/arguments/tuner_args.py
#   - Training args: swift/trainers/arguments.py (extends HuggingFace Seq2SeqTrainingArguments)
#   - SFT args:      swift/arguments/sft_args.py
#   - Quant args:    swift/arguments/base_args/quant_args.py
# =============================================================================

# ---- ENVIRONMENT SETUP (run once, then comment out) ----
# pip install -U ms-swift deepspeed
# pip install -U "transformers>=5.2.0,<5.3.0" "qwen_vl_utils>=0.0.14" peft liger-kernel
# pip install -U git+https://github.com/fla-org/flash-linear-attention   # required for Qwen3.5 GatedDeltaNet
# pip install -U git+https://github.com/Dao-AILab/causal-conv1d --no-build-isolation

# =============================================================================
# USER CONFIG
# =============================================================================
MODEL_PATH="/mnt/bn/youxiang-lf/models/Qwen3.5-2B"
DATASET_NAME="topic_generation_policy_sft-ms-swift"
DATASET_INFO="/mnt/bn/youxiang-lf/data/dataset_info_ms_swift.json"
OUTPUT_DIR="test/Qwen3.5-2B-sft"
NUM_GPUS=4

export WANDB_PROJECT="Qwen3.5-SFT"
export WANDB_NAME="$OUTPUT_DIR"

# =============================================================================
# GPU & MEMORY CONFIG
# =============================================================================
GPU_IDS=$(seq -s ',' 0 $((NUM_GPUS - 1)))

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=${NUM_GPUS} \
CUDA_VISIBLE_DEVICES=${GPU_IDS} \
swift sft \
    \
    # =========================================================================
    # MODEL ARGS  (model_args.py)
    # =========================================================================
    --model ${MODEL_PATH} \
    --torch_dtype bfloat16 \
        # bfloat16 | float16 | float32. Model weight precision.
    # --model_type qwen3_5 \
        # Auto-detected from model config. Override only if detection fails.
    # --attn_impl flash_attention_2 \
        # sdpa | eager | flash_attn | flash_attention_2 | flash_attention_3
        # NOTE: Qwen3.5 uses GatedDeltaNet (linear attention) which has its own impl.
    --experts_impl grouped_mm \
        # grouped_mm | batched_mm | eager. MoE expert execution kernel.
        # grouped_mm is fastest; use 'eager' as fallback if grouped_mm errors.
    # --rope_scaling '{"type": "yarn", "factor": 4.0}' \
        # RoPE scaling for context extension. JSON string or type name.
    # --device_map auto \
        # auto | sequential | balanced | balanced_low_0 | JSON dict.
        # Usually NOT needed with DeepSpeed (DeepSpeed handles sharding).
    # --max_model_len 4096 \
        # Override model's default max sequence length from config.
    # --model_kwargs '{"trust_remote_code": true}' \
        # Extra kwargs passed to AutoModelForCausalLM.from_pretrained(). JSON string.
    \
    # =========================================================================
    # DATA ARGS  (data_args.py)
    # =========================================================================
    --custom_dataset_info ${DATASET_INFO} \
        # Path to your dataset_info.json for dataset registration.
    --dataset "${DATASET_NAME}" \
        # Dataset ID(s) or path(s). Formats:
        #   'dataset_id'              - use registered dataset
        #   'dataset_id:subset'       - specific subset
        #   'dataset_id#1000'         - sample 1000 rows
        #   '/path/to/train.jsonl'    - local file directly
        # Multiple datasets: --dataset 'ds1#500' 'ds2#500'
    # --val_dataset '/path/to/val.jsonl' \
        # Explicit validation set. If not set, use split_dataset_ratio.
    --split_dataset_ratio 0.05 \
        # Auto-split this fraction from training set as validation (0.0 = no split).
    --dataset_num_proc 4 \
        # Number of processes for dataset preprocessing/tokenization.
    --load_from_cache_file true \
        # Cache tokenized dataset to disk. Set true for repeated runs on same data.
    # --dataset_shuffle true \
        # Shuffle training data (default: true).
    # --data_seed 42 \
        # Random seed for dataset shuffling (default: 42).
    # --columns '{"input": "query", "output": "response"}' \
        # Column name mapping. JSON string. Maps your column names to swift's expected names.
        # Standard names: query, response, system, history, images, videos, audios, objects.
    # --strict false \
        # If true, raise error on malformed data rows instead of skipping.
    # --streaming false \
        # Enable streaming mode for very large datasets that don't fit in memory.
    # --interleave_prob '[0.5, 0.3, 0.2]' \
        # Sampling probabilities when using multiple datasets (must sum to 1.0).
    # --model_name 'MyModel' '我的模型' \
        # Model name [English, Chinese] for self-cognition dataset.
    # --model_author 'MyOrg' '我的组织' \
        # Model author [English, Chinese] for self-cognition dataset.
    \
    # =========================================================================
    # TEMPLATE ARGS  (template_args.py)
    # =========================================================================
    --max_length 2048 \
        # Max tokens per sample. Samples exceeding this are handled by truncation_strategy.
    # --truncation_strategy delete \
        # delete | left | right | split. How to handle over-length samples.
        # delete: discard sample; left/right: truncate; split: split into chunks.
    # --system 'You are a helpful assistant.' \
        # Custom system prompt. Can also be a path to a .txt file.
    --group_by_length true \
        # Group similar-length samples to reduce padding. Causes loss curve fluctuations
        # but speeds up training. Required for Qwen3.5 (no packing support).
    # --padding_side right \
        # left | right. Padding side (default: right for training).
    # --padding_free false \
        # Flatten batch to avoid padding entirely. NOT supported for Qwen3.5 GatedDeltaNet.
    # --packing false \
        # Pack multiple samples into one sequence. NOT supported for Qwen3.5 GatedDeltaNet.
    --loss_scale default \
        # Loss weighting strategy. Options:
        #   'default'             - standard loss
        #   'ignore_empty_think'  - ignore empty <think></think> blocks (for hybrid thinking)
        #   JSON string for custom per-round weights, e.g. '{"0": 0.2, "1": 1.0}'
    --add_non_thinking_prefix true \
        # Add <think>\n\n</think>\n\n prefix for non-thinking training samples.
        # Set true for Qwen3.5 (hybrid thinking model).
    # --enable_thinking false \
        # Enable/disable thinking mode during inference. Not used during training.
    # --template qwen3_5 \
        # Auto-detected. Override only if needed.
    # --sequence_parallel_size 1 \
        # Sequence parallelism degree. >1 splits long sequences across GPUs.
    \
    # =========================================================================
    # TUNER ARGS  (tuner_args.py)
    # =========================================================================
    --tuner_type full \
        # Tuning method:
        #   full      - full parameter fine-tuning (all weights updated)
        #   lora      - LoRA (low-rank adaptation)
        #   adalora   - adaptive LoRA (auto rank adjustment)
        #   llamapro  - LLaMA-Pro (insert new blocks)
        #   adapter   - bottleneck adapter
        #   vera      - VeRA (shared random matrices)
        #   boft      - Butterfly OFT
        #   fourierft - FourierFT
        #   reft      - Representation Fine-Tuning
        #   bone      - Bone tuning
    --target_modules all-linear \
        # Which modules to apply tuner to. 'all-linear' targets all nn.Linear layers.
        # Specific modules: 'q_proj k_proj v_proj o_proj gate_proj up_proj down_proj'
        # Use specific list to exclude router: avoids training MoE routing weights.
    # --tuner_backend peft \
        # peft | unsloth. Backend for PEFT methods.
    \
    # --- LoRA-specific (only used when tuner_type=lora) ---
    # --lora_rank 8 \
        # LoRA rank. Higher = more capacity but more memory. Typical: 8, 16, 32, 64.
    # --lora_alpha 32 \
        # LoRA scaling factor. Effective scale = alpha/rank. Common: 2x-4x of rank.
    # --lora_dropout 0.05 \
        # Dropout in LoRA layers (default: 0.05).
    # --lora_bias none \
        # none | all. Whether to make bias parameters trainable.
    # --lora_dtype bfloat16 \
        # Data type for LoRA weights. None = follow model dtype.
    # --use_rslora false \
        # Use Rank-Stabilized LoRA variant (scales by 1/sqrt(rank) instead of 1/rank).
    # --use_dora false \
        # Use DoRA (Weight-Decomposed LoRA) variant. More expressive but slower.
    # --lorap_lr_ratio 16.0 \
        # LoRA+ learning rate ratio. Set 10-16 for LoRA+ optimizer.
    # --init_weights true \
        # LoRA init: true | false | gaussian | pissa | pissa_niter_4 | olora | loftq | lora-ga
    # --modules_to_save '' \
        # Additional modules to save in full (e.g., 'lm_head embed_tokens').
    \
    # --- Freeze control (for full or selective training) ---
    # --freeze_llm false \
        # Freeze LLM backbone (multimodal models). Train only vision/aligner.
    # --freeze_vit true \
        # Freeze vision encoder (multimodal). Default true.
    # --freeze_aligner true \
        # Freeze vision-language aligner (multimodal). Default true.
    # --freeze_parameters '' \
        # Space-separated parameter name prefixes to freeze.
    # --freeze_parameters_ratio 0.0 \
        # Freeze bottom N% of parameters (0.0-1.0).
    # --trainable_parameters '' \
        # Explicitly set parameter prefixes as trainable (overrides freeze).
    \
    # =========================================================================
    # OPTIMIZER & LR SCHEDULE  (trainers/arguments.py, HuggingFace TrainingArguments)
    # =========================================================================
    --learning_rate 1e-4 \
        # Peak learning rate. Recommendations:
        #   full fine-tuning: 1e-5 to 5e-5
        #   LoRA:             1e-4 to 2e-4
    --lr_scheduler_type cosine \
        # cosine | linear | constant | constant_with_warmup | polynomial |
        # inverse_sqrt | reduce_lr_on_plateau | cosine_with_restarts |
        # warmup_stable_decay
    # --lr_scheduler_kwargs '{}' \
        # Extra kwargs for LR scheduler (JSON). E.g., for cosine_with_restarts:
        # '{"num_cycles": 3}'
    --warmup_ratio 0.05 \
        # Fraction of total steps for linear warmup (0.0-1.0).
    # --warmup_steps 0 \
        # Explicit warmup steps (overrides warmup_ratio if > 0).
    --weight_decay 0.1 \
        # L2 regularization. Default 0.1. Set 0.0 for LoRA typically.
    # --adam_beta1 0.9 \
        # Adam optimizer beta1.
    # --adam_beta2 0.95 \
        # Adam optimizer beta2. Default 0.95 in ms-swift (HF default is 0.999).
    # --adam_epsilon 1e-8 \
        # Adam optimizer epsilon.
    --max_grad_norm 1.0 \
        # Gradient clipping max norm. Prevents exploding gradients.
    # --optim adamw_torch \
        # Optimizer. Options:
        #   adamw_torch | adamw_hf | adafactor | sgd | adagrad |
        #   adamw_bnb_8bit | adamw_apex_fused | lion_8bit | lion_32bit |
        #   paged_adamw_32bit | paged_adamw_8bit | galore_adamw |
        #   ademamix_8bit | ...
    # --optim_args '' \
        # Optimizer constructor args (JSON string).
    \
    # =========================================================================
    # TRAINING LOOP  (trainers/arguments.py)
    # =========================================================================
    --num_train_epochs 3 \
        # Number of training epochs. For 1k samples, 3-5 epochs is common.
    # --max_steps -1 \
        # Max training steps. Overrides num_train_epochs if > 0.
        # Required for --streaming mode.
    --per_device_train_batch_size 2 \
        # Batch size per GPU. Reduce if OOM.
    --per_device_eval_batch_size 1 \
        # Batch size per GPU for evaluation.
    --gradient_accumulation_steps 4 \
        # Accumulate gradients over N steps before updating.
        # Effective batch size = NUM_GPUS * per_device_batch * grad_accum
        # If not set, auto-calculated as: 16 / (batch_size * world_size).
    --gradient_checkpointing true \
        # Trade compute for memory. Recomputes activations during backward pass.
        # Almost always set true for large models.
    # --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
        # Extra kwargs. use_reentrant=false is safer with DDP.
    \
    # =========================================================================
    # MoE-SPECIFIC ARGS  (model_args.py, trainers/arguments.py)
    # =========================================================================
    --router_aux_loss_coef 1e-3 \
        # Auxiliary load-balancing loss coefficient for MoE models.
        # Encourages balanced expert utilization. Typical: 1e-3 to 1e-6.
        # Set 0 to disable. Only relevant for MoE models (e.g., Qwen3.5-35B-A3B).
    \
    # =========================================================================
    # CHECKPOINTING & OUTPUT  (trainers/arguments.py)
    # =========================================================================
    --output_dir ${OUTPUT_DIR} \
        # Output directory for checkpoints and logs.
    # --save_strategy steps \
        # no | steps | epoch. When to save checkpoints.
    --save_steps 50 \
        # Save checkpoint every N steps.
    --save_total_limit 3 \
        # Keep only the N most recent checkpoints (saves disk space).
    # --save_only_model false \
        # If true, only save model weights (skip optimizer/scheduler state).
        # Saves disk but can't resume training from checkpoint.
    # --save_safetensors true \
        # Save in safetensors format (default: true). Safer and faster than .bin.
    # --resume_from_checkpoint /path/to/checkpoint \
        # Resume training from a checkpoint directory.
    # --resume_only_model false \
        # Resume only model weights, reset optimizer/scheduler/step count.
    \
    # =========================================================================
    # EVALUATION  (trainers/arguments.py)
    # =========================================================================
    # --eval_strategy steps \
        # no | steps | epoch. Defaults to save_strategy.
    --eval_steps 50 \
        # Evaluate every N steps.
    # --eval_on_start false \
        # Run evaluation before training starts.
    # --metric_for_best_model 'loss' \
        # Metric to determine best checkpoint.
    # --greater_is_better false \
        # Whether higher metric = better (false for loss, true for accuracy).
    # --predict_with_generate false \
        # Use model.generate() during eval. Slower but shows actual generations.
    # --acc_strategy token \
        # token | seq. Token-level or sequence-level accuracy calculation.
    \
    # =========================================================================
    # LOGGING & MONITORING  (trainers/arguments.py)
    # =========================================================================
    --logging_steps 5 \
        # Log training metrics every N steps.
    # --logging_first_step true \
        # Log metrics at the very first step.
    --report_to wandb \
        # Logging backends (space-separated): tensorboard | wandb | swanlab | none
        # For wandb: set WANDB_PROJECT, WANDB_NAME env vars above.
    # --run_name '' \
        # Experiment name (shown in wandb/tensorboard).
    # --logging_dir '' \
        # TensorBoard log directory (default: output_dir/runs/).
    \
    # =========================================================================
    # DATA LOADING PERFORMANCE  (trainers/arguments.py)
    # =========================================================================
    --dataloader_num_workers 4 \
        # Number of data loading worker processes. 0 = main process only.
    # --dataloader_pin_memory true \
        # Pin memory for faster GPU transfer.
    # --dataloader_persistent_workers false \
        # Keep workers alive between epochs. Faster but uses more memory.
    # --dataloader_prefetch_factor 2 \
        # Number of batches prefetched per worker.
    \
    # =========================================================================
    # DISTRIBUTED TRAINING  (trainers/arguments.py)
    # =========================================================================
    --deepspeed zero3
        # DeepSpeed config. Built-in presets:
        #   zero0          - no sharding (DDP equivalent)
        #   zero1          - optimizer state sharding
        #   zero2          - optimizer + gradient sharding
        #   zero3          - optimizer + gradient + parameter sharding (for large models)
        #   zero2_offload  - zero2 + CPU offloading
        #   zero3_offload  - zero3 + CPU offloading (slowest but fits biggest models)
        #   /path/to/ds_config.json  - custom DeepSpeed config file
    # --ddp_backend nccl \
        # DDP backend: nccl | gloo | mpi | ccl | hccl | cncl | mccl
    # --ddp_find_unused_parameters false \
        # Set true if you see DDP errors about unused parameters.
    # --ddp_timeout 18000000 \
        # DDP timeout in seconds (default: 18000000).
    # --fsdp '' \
        # FSDP config string (alternative to DeepSpeed).
    # --zero_hpz_partition_size 8 \
        # ZeRO++ hierarchical partition size.
    # --deepspeed_autotp_size 2 \
        # DeepSpeed AutoTP tensor parallelism degree.

    # =========================================================================
    # QUANTIZATION (for QLoRA)  (quant_args.py)
    # =========================================================================
    # --quant_method bnb \
        # bnb | hqq | eetq | quanto | fp8. Quantization method.
    # --quant_bits 4 \
        # 1 | 2 | 3 | 4 | 8. Number of bits.
    # --bnb_4bit_compute_dtype bfloat16 \
        # Compute dtype for 4-bit BNB quantization.
    # --bnb_4bit_quant_type nf4 \
        # nf4 | fp4. Quantization type for BNB.
    # --bnb_4bit_use_double_quant true \
        # Use double quantization to save memory.

    # =========================================================================
    # ADVANCED / EXPERIMENTAL
    # =========================================================================
    # --seed 42 \
        # Global random seed for reproducibility.
    # --full_determinism false \
        # Fully deterministic training (slower).
    # --neftune_noise_alpha 0.0 \
        # NEFTune: add noise to embeddings during training. Try 5.0 to improve.
    # --use_liger_kernel false \
        # Use Liger kernel for fused operations (faster training).
    # --average_tokens_across_devices false \
        # Synchronize token counts across GPUs for more accurate loss.
    # --use_logits_to_keep true \
        # Only compute logits for necessary tokens (saves memory). Auto-detected.
    # --vit_lr 1e-5 \
        # Separate learning rate for vision encoder (multimodal).
    # --aligner_lr 1e-4 \
        # Separate learning rate for vision-language aligner (multimodal).
    # --vit_gradient_checkpointing true \
        # Gradient checkpointing specifically for vision encoder.
    # --lazy_tokenize true \
        # Tokenize samples on-the-fly instead of upfront. Saves RAM for large datasets.
    # --check_model true \
        # Verify model file integrity on load (default: true).
    # --use_galore false \
        # Use GaLore optimizer for memory-efficient full-param training.
    # --galore_rank 128 \
        # GaLore rank.
    # --lisa_activated_layers 0 \
        # LISA: number of randomly activated layers per step (0 = disabled).
    # --use_flash_ckpt false \
        # Use DLRover Flash Checkpoint for faster saving.
    # --callbacks '' \
        # Custom trainer callbacks (space-separated class names).

# =============================================================================
# After training, run inference on the trained model:
#
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# swift infer \
#     --adapters ${OUTPUT_DIR}/vx-xxx/checkpoint-xxx \
#     --stream true \
#     --experts_impl grouped_mm \
#     --enable_thinking false \
#     --max_new_tokens 512 \
#     --load_data_args true
# =============================================================================
