# Multi-Teacher GKD: User Guide

## What is Multi-Teacher GKD?

Generalized Knowledge Distillation (GKD) trains a smaller student model by minimizing the Jensen-Shannon Divergence (JSD) between the student's and a teacher's output distributions. Standard GKD uses a single teacher for all training data.

**Multi-teacher GKD** extends this by assigning different domain-specialized teachers to different data samples based on a `channel` field. For example:

- Math problems are distilled from a math-tuned teacher
- Code tasks are distilled from a code-tuned teacher
- General chat is distilled from a general-purpose teacher

This allows the student to learn domain expertise from specialized teachers while training on a unified dataset.

```
                    ┌─────────────────────┐
                    │   Training Data     │
                    │ (with channel field) │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Channel Router    │
                    │  math → teacher_0   │
                    │  code → teacher_1   │
                    └──┬──────────────┬───┘
                       │              │
              ┌────────▼────┐  ┌──────▼──────┐
              │ Math Teacher│  │ Code Teacher│
              │ (frozen)    │  │ (frozen)    │
              └────────┬────┘  └──────┬──────┘
                       │              │
                  soft labels    soft labels
                       │              │
                    ┌──▼──────────────▼───┐
                    │    JSD Loss         │
                    │  (per-sample)       │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Student Model     │
                    │   (trainable)       │
                    └─────────────────────┘
```

## Key Properties

- **Per-sample routing**: Each sample in a batch is independently routed to its designated teacher
- **JSD loss unchanged**: The same Generalized JSD loss function is used (with temperature, beta, chunked computation)
- **Backward compatibility**: Single-teacher GKD works exactly as before
- **Deduplication**: If multiple domains share the same teacher model path, the model is loaded only once
- **vLLM support**: vLLM is used for student on-policy generation only (unaffected by multi-teacher)

## Data Preparation

Each training sample must include a `"channel"` field matching one of the keys in `--teacher_domain_map`.

### JSONL Format

```jsonl
{"messages":[{"role":"user","content":"Solve: 2x + 3 = 7"},{"role":"assistant","content":"x = 2"}],"channel":"math"}
{"messages":[{"role":"user","content":"Write a Python sort function"},{"role":"assistant","content":"def sort(arr): return sorted(arr)"}],"channel":"code"}
{"messages":[{"role":"user","content":"Tell me about climate change"},{"role":"assistant","content":"Climate change refers to..."}],"channel":"general"}
```

### Important Notes on Data

- The `channel` field is a first-class field in ms-swift's data pipeline — it flows through the preprocessor, template encoder, and data collator automatically
- **Samples without a `channel` field default to teacher[0]** (the first unique teacher in the domain map)
- You can mix channeled and un-channeled samples in the same dataset
- The channel field does not affect the training messages, only teacher routing

## CLI Usage

### Basic Multi-Teacher Training

```bash
swift rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen2.5-7B \
    --teacher_domain_map '{"math":"path/to/math-teacher","code":"path/to/code-teacher"}' \
    --dataset /path/to/channeled_data.jsonl \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-6 \
    --temperature 0.9 \
    --beta 0.5
```

### With Teacher Offloading (Memory-Constrained)

When using large teachers, offload them to CPU between forward passes:

```bash
swift rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen2.5-7B \
    --teacher_domain_map '{"math":"Qwen/Qwen2.5-72B-Instruct","code":"deepseek-ai/DeepSeek-V3"}' \
    --dataset /path/to/data.jsonl \
    --offload_teacher_model true \
    --per_device_train_batch_size 2
```

With offloading, peak GPU memory usage is the same as single-teacher training — only one teacher is on GPU at any time.

### With Sequential KD (Teacher Generates Responses)

```bash
swift rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen2.5-7B \
    --teacher_domain_map '{"math":"path/to/math-teacher","code":"path/to/code-teacher"}' \
    --dataset /path/to/data.jsonl \
    --seq_kd true \
    --max_completion_length 512
```

When `seq_kd=True`, the teacher generates new responses (replacing dataset responses). For multi-teacher, the majority teacher in each batch is used for generation, while soft labels are still routed per-sample.

### With On-Policy Student Generation (vLLM)

```bash
NPROC_PER_NODE=2 swift rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen2.5-7B \
    --teacher_domain_map '{"math":"path/to/math-teacher","code":"path/to/code-teacher"}' \
    --dataset /path/to/data.jsonl \
    --use_vllm true \
    --vllm_mode colocate \
    --lmbda 0.5
```

vLLM handles student on-policy generation only. Teacher models are standard PyTorch models. The `lmbda` parameter controls the probability of using student-generated responses vs dataset responses.

### With DeepSpeed

```bash
swift rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen2.5-7B \
    --teacher_domain_map '{"math":"path/to/math-teacher","code":"path/to/code-teacher"}' \
    --dataset /path/to/data.jsonl \
    --deepspeed zero2
```

You can also set a separate DeepSpeed config for teacher models:

```bash
swift rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen2.5-7B \
    --teacher_domain_map '{"math":"path/to/math-teacher","code":"path/to/code-teacher"}' \
    --dataset /path/to/data.jsonl \
    --deepspeed zero3 \
    --teacher_deepspeed '{"zero_optimization":{"stage":2}}'
```

### Shared Teachers (Deduplication)

If multiple domains should use the same teacher, just point them to the same model path. The model is loaded once and shared:

```bash
swift rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen2.5-7B \
    --teacher_domain_map '{"math":"Qwen/Qwen2.5-72B-Instruct","science":"Qwen/Qwen2.5-72B-Instruct","code":"deepseek-coder-v2"}' \
    --dataset /path/to/data.jsonl
```

This loads 2 unique teacher models (not 3), saving GPU memory.

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--teacher_domain_map` | None | JSON mapping channel names to teacher paths |
| `--teacher_model` | None | Single teacher path (mutually exclusive with domain map) |
| `--offload_teacher_model` | False | CPU offload teachers between forward passes |
| `--seq_kd` | False | Teacher generates responses instead of using dataset |
| `--lmbda` | 0.5 | Probability of on-policy (student-generated) training |
| `--temperature` | 0.9 | Temperature for JSD loss softmax scaling |
| `--beta` | 0.5 | JSD interpolation (0=forward KL, 0.5=symmetric, 1=reverse KL) |
| `--sft_alpha` | 0 | Weight for auxiliary SFT loss (only on dataset/teacher data) |
| `--use_liger_kernel` | False | Fused Liger JSD loss (falls back to standard for mixed-teacher batches) |

## How Multi-Teacher Routing Works

1. **Data loading**: Each sample's `channel` field is preserved through the data pipeline
2. **Batch formation**: Standard batching — no special sorting by channel needed
3. **`compute_loss()`**:
   - The `channel` field is popped from inputs before model forward
   - `_get_teacher_indices()` maps channels to teacher indices
   - If all samples use the same teacher: single teacher forward (fast path)
   - If mixed teachers: `_multi_teacher_forward()` groups by teacher, runs subset forwards, scatters logits back
4. **JSD loss**: Computed on the combined student/teacher logits — same function as single-teacher
5. **Vocab alignment**: If teachers have different vocab sizes, logits are padded to match

## Performance Considerations

### GPU Memory

- **Without offloading**: All teachers are on GPU simultaneously. Memory = student + all teachers.
- **With offloading**: One teacher on GPU at a time. Memory = student + 1 teacher (same as single-teacher).
- **Recommendation**: Use `--offload_teacher_model true` when teachers are large or numerous.

### Batch Efficiency

- **Homogeneous batches** (all samples share one teacher): Fastest — single teacher forward, no grouping overhead.
- **Mixed batches** (samples from multiple teachers): Each teacher does a subset forward. Total computation is the same, but there's a small overhead from tensor indexing and scatter.
- For best efficiency, keep batches homogeneous by sorting data by channel. However, this is not required — random mixing works correctly.

### Liger Kernel

The fused Liger JSD loss (`--use_liger_kernel true`) operates on hidden states and requires both student and teacher base models. It supports single-teacher batches only:
- Homogeneous batches: Liger kernel is used
- Mixed batches: Falls back to standard loss path with a one-time warning

## Common Issues

### "teacher_domain_map and teacher_model are mutually exclusive"
You set both `--teacher_model` and `--teacher_domain_map`. Use only one.

### "GKD requires either --teacher_model or --teacher_domain_map"
You must specify at least one teacher source for GKD training.

### "teacher_domain_map must be a non-empty JSON dict"
The JSON string could not be parsed or is empty. Ensure valid JSON with at least one entry:
```bash
--teacher_domain_map '{"domain":"model/path"}'
```

### Missing channel in data
Samples without a `channel` field are routed to teacher[0] (first unique teacher in the map). This is by design — you can mix channeled and non-channeled data.

### Different teacher vocab sizes
Handled automatically. Teacher logits are padded with zeros to match the largest vocab size, then aligned with the student vocab.

## Comparison: Single-Teacher vs Multi-Teacher

| Aspect | Single-Teacher | Multi-Teacher |
|--------|---------------|---------------|
| CLI | `--teacher_model path` | `--teacher_domain_map '{"ch":"path",...}'` |
| Data | Standard messages | Messages + `channel` field |
| Teacher loading | 1 model | N unique models (deduped) |
| Loss | JSD (all samples vs 1 teacher) | JSD (each sample vs its routed teacher) |
| seq_kd | 1 teacher generates | Majority teacher generates |
| vLLM | Student generation | Student generation (unchanged) |
| Offloading | 1 teacher to/from CPU | Each teacher to/from CPU independently |
