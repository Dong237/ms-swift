# Multi-Teacher GKD Testing Guide

This guide describes how to test the multi-teacher GKD feature in various environments.

## Prerequisites

- Python >= 3.9
- PyTorch >= 2.0 with CUDA support
- ms-swift installed (editable mode: `pip install -e .`)
- pytest (`pip install pytest`)
- At least 1 GPU (24 GB VRAM recommended for Qwen2.5-0.5B tests)
- Internet access for model downloads (first run only)

## 1. Unit Tests (No GPU Required)

Unit tests validate argument parsing, deduplication, and routing logic without loading any models.

```bash
python -m pytest tests/train/test_multi_teacher_gkd.py -v \
    -k "TestTeacherDomainMapParsing or TestGetTeacherIndices"
```

**What is tested:**
- JSON parsing of `--teacher_domain_map`
- Mutual exclusivity with `--teacher_model`
- Path deduplication (same model path for multiple domains)
- Channel-to-teacher-index routing
- Default routing for unknown/missing channels

## 2. Integration Tests (Single GPU)

Integration tests run the full training pipeline with small models.

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/train/test_multi_teacher_gkd.py -v \
    -k "TestMultiTeacherGKDIntegration" --timeout=600
```

**Individual tests:**

| Test | What it validates |
|------|-------------------|
| `test_multi_teacher_basic` | Two-teacher GKD with channel-routed data completes training |
| `test_multi_teacher_with_offload` | Teacher CPU offloading works with multi-teacher |
| `test_multi_teacher_seq_kd` | Sequential KD (teacher generates) with multi-teacher routing |
| `test_multi_teacher_default_routing` | Data without `channel` field defaults to teacher[0] |
| `test_single_teacher_backward_compat` | Single `--teacher_model` still works (regression check) |
| `test_dedup_loads_once` | Two domains pointing to same path load only one model |

## 3. Manual CLI Test (Single GPU)

Create test data with channel fields:

```bash
cat > /tmp/test_multi_teacher.jsonl << 'EOF'
{"messages":[{"role":"user","content":"Solve: 2+3"},{"role":"assistant","content":"5"}],"channel":"math"}
{"messages":[{"role":"user","content":"Solve: 10-4"},{"role":"assistant","content":"6"}],"channel":"math"}
{"messages":[{"role":"user","content":"Write hello world in Python"},{"role":"assistant","content":"print('hello world')"}],"channel":"code"}
{"messages":[{"role":"user","content":"Write a for loop"},{"role":"assistant","content":"for i in range(10): print(i)"}],"channel":"code"}
{"messages":[{"role":"user","content":"What is 7*8?"},{"role":"assistant","content":"56"}],"channel":"math"}
{"messages":[{"role":"user","content":"Write a list comprehension"},{"role":"assistant","content":"[x**2 for x in range(10)]"}],"channel":"code"}
{"messages":[{"role":"user","content":"Solve: x+1=3"},{"role":"assistant","content":"x=2"}],"channel":"math"}
{"messages":[{"role":"user","content":"Reverse a string"},{"role":"assistant","content":"s[::-1]"}],"channel":"code"}
EOF
```

Run multi-teacher GKD:

```bash
CUDA_VISIBLE_DEVICES=0 swift rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen2.5-0.5B \
    --teacher_domain_map '{"math":"Qwen/Qwen2.5-0.5B","code":"Qwen/Qwen2.5-0.5B"}' \
    --dataset /tmp/test_multi_teacher.jsonl \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --save_steps 5
```

**Expected output:** Training completes without errors. Logs should show:
```
Multi-teacher GKD: loaded 1 unique teacher(s) for 2 domain(s)
```
(1 unique teacher because both domains use the same model path.)

For truly different teachers, use different model paths:
```bash
CUDA_VISIBLE_DEVICES=0 swift rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen2.5-0.5B \
    --teacher_domain_map '{"math":"Qwen/Qwen2.5-1.5B-Instruct","code":"Qwen/Qwen2.5-3B-Instruct"}' \
    --dataset /tmp/test_multi_teacher.jsonl \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2
```

## 4. Teacher Offloading Test

For memory-constrained environments:

```bash
CUDA_VISIBLE_DEVICES=0 swift rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen2.5-0.5B \
    --teacher_domain_map '{"math":"Qwen/Qwen2.5-1.5B-Instruct","code":"Qwen/Qwen2.5-1.5B-Instruct"}' \
    --dataset /tmp/test_multi_teacher.jsonl \
    --offload_teacher_model true \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2
```

When `--offload_teacher_model true`, teachers are moved to CPU between forward passes. Peak GPU memory should be similar to single-teacher training.

## 5. Sequential KD + Multi-Teacher Test

```bash
CUDA_VISIBLE_DEVICES=0 swift rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen2.5-0.5B \
    --teacher_domain_map '{"math":"Qwen/Qwen2.5-0.5B","code":"Qwen/Qwen2.5-0.5B"}' \
    --dataset /tmp/test_multi_teacher.jsonl \
    --seq_kd true \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2
```

When `seq_kd=True`, the majority teacher in each batch is used for response generation, while soft labels still route per-sample to the correct teacher.

## 6. vLLM Integration Tests (Multi-GPU)

### Colocate Mode (shared GPU)

```bash
CUDA_VISIBLE_DEVICES=0,1 swift rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen2.5-0.5B \
    --teacher_domain_map '{"math":"Qwen/Qwen2.5-0.5B","code":"Qwen/Qwen2.5-0.5B"}' \
    --dataset /tmp/test_multi_teacher.jsonl \
    --use_vllm true \
    --vllm_mode colocate \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2
```

vLLM is used only for student on-policy generation (not affected by multi-teacher). Teachers are still loaded as standard PyTorch models.

### Server Mode (separate process)

```bash
CUDA_VISIBLE_DEVICES=0,1 swift rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen2.5-0.5B \
    --teacher_domain_map '{"math":"Qwen/Qwen2.5-0.5B","code":"Qwen/Qwen2.5-0.5B"}' \
    --dataset /tmp/test_multi_teacher.jsonl \
    --use_vllm true \
    --vllm_mode server \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2
```

## 7. DeepSpeed Tests

### ZeRO-2

```bash
CUDA_VISIBLE_DEVICES=0 swift rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen2.5-0.5B \
    --teacher_domain_map '{"math":"Qwen/Qwen2.5-0.5B","code":"Qwen/Qwen2.5-0.5B"}' \
    --dataset /tmp/test_multi_teacher.jsonl \
    --deepspeed zero2 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2
```

### Teacher with Separate DeepSpeed Config

```bash
CUDA_VISIBLE_DEVICES=0 swift rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen2.5-0.5B \
    --teacher_domain_map '{"math":"Qwen/Qwen2.5-0.5B","code":"Qwen/Qwen2.5-0.5B"}' \
    --dataset /tmp/test_multi_teacher.jsonl \
    --deepspeed zero2 \
    --teacher_deepspeed '{"zero_optimization":{"stage":2}}' \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2
```

## 8. Regression Check

Run the existing single-teacher GKD test to verify no regressions:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/train/test_gkd.py::test_llm -v --timeout=600
```

## Troubleshooting

### OOM with multiple teachers
- Use `--offload_teacher_model true` to load only one teacher at a time
- Reduce `--per_device_train_batch_size`
- Use DeepSpeed ZeRO-2 or ZeRO-3

### "channel" field not found
- Ensure your JSONL data has a `"channel"` key in each sample
- Channel values must match keys in `--teacher_domain_map`
- Samples without a channel field default to teacher[0]

### Liger kernel warning
- The fused Liger JSD loss does not support mixed-teacher batches
- When a batch has samples from different teachers, it automatically falls back to the standard loss path
- For homogeneous batches (all samples use same teacher), Liger works normally

### Teacher model download failures
- Each unique path in `--teacher_domain_map` is downloaded once
- Ensure network access to ModelScope/HuggingFace
- Use `--use_hf true` for HuggingFace Hub models
