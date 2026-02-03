# GKD (Generalized Knowledge Distillation) in ms-swift — Deep Dive

## Table of Contents

1. [Overview](#1-overview)
2. [System Architecture](#2-system-architecture)
3. [Three Training Modes](#3-three-training-modes)
4. [Training Step Flow](#4-training-step-flow)
5. [Loss Computation](#5-loss-computation)
6. [Synchronous vs Asynchronous — The Core Question](#6-synchronous-vs-asynchronous--the-core-question)
7. [Memory Optimization Strategies](#7-memory-optimization-strategies)
8. [Megatron GKD Implementation](#8-megatron-gkd-implementation)
9. [Configuration Reference](#9-configuration-reference)
10. [Source File Index](#10-source-file-index)

---

## 1. Overview

### What is GKD?

GKD (Generalized Knowledge Distillation), introduced by Agarwal et al. (2024), generalizes traditional knowledge distillation for autoregressive language models. It addresses the **train-inference mismatch** problem: standard KD trains the student on *teacher-generated* or *dataset* responses, but at inference time the student generates its own tokens — leading to distribution shift.

GKD solves this by **mixing on-policy (student-generated) and off-policy (dataset) data** during training, controlled by a probability parameter λ. It also generalizes the divergence measure from KL to the full Jensen-Shannon Divergence (JSD) family via a parameter β.

### GKD in ms-swift

ms-swift integrates GKD as a first-class RLHF training type with two implementations:

| Implementation | File | Use Case |
|---|---|---|
| **HuggingFace-based** | `swift/rlhf_trainers/gkd_trainer.py` | Standard training with DeepSpeed/FSDP |
| **Megatron-based** | `swift/megatron/trainers/gkd_trainer.py` | Large-scale training with Pipeline/Tensor Parallelism |

**Launch command:**
```bash
swift rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen2.5-7B \
    --teacher_model Qwen/Qwen2.5-14B-Instruct \
    --temperature 0.9 \
    --lmbda 0.5 \
    --beta 0.5
```

---

## 2. System Architecture

### Class Hierarchy

```
                         HuggingFace Trainer
                                |
                    +-----------+-----------+
                    |                       |
              SwiftMixin              HFGKDTrainer (trl)
              (mixin.py)              (deleted __init__)
                    |                       |
                    +-----------+-----------+
                                |
                      RLHFTrainerMixin
                                |
                      RolloutTrainerMixin
                       (rollout_mixin.py)
                                |
                         ┌──────┴──────┐
                         │ GKDTrainer  │
                         │(gkd_trainer)│
                         └─────────────┘
```

### System Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GKD Training System                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌───────────────────┐    ┌──────────────────┐ │
│  │  Dataset      │    │  Student Model    │    │  Teacher Model   │ │
│  │  (off-policy) │    │  (trainable)      │    │  (frozen, eval)  │ │
│  │              │    │                   │    │                  │ │
│  │  messages[]  │    │  .forward()       │    │  .forward()      │ │
│  │  images[]    │    │  .generate()      │    │  (no_grad)       │ │
│  └──────┬───────┘    └────┬──────┬───────┘    └────────┬─────────┘ │
│         │                 │      │                      │           │
│         │        ┌────────┘      │                      │           │
│         ▼        ▼               ▼                      ▼           │
│  ┌─────────────────┐   ┌────────────────────────────────────────┐  │
│  │  DataSource      │   │           compute_loss()               │  │
│  │  Selection       │   │                                        │  │
│  │  (probabilistic) │   │  student_logits ──┐                    │  │
│  │                  │   │                    ├─► JSD Loss         │  │
│  │  λ → STUDENT    │   │  teacher_logits ──┘    (β, temp)       │  │
│  │  seq_kd→TEACHER │   │                                        │  │
│  │  else → DATASET │   │  + optional SFT loss (α)               │  │
│  └──────┬──────────┘   └───────────────┬────────────────────────┘  │
│         │                              │                            │
│         │         ┌────────────────────┘                            │
│         │         ▼                                                 │
│         │  ┌──────────────┐                                        │
│         │  │   Backward   │                                        │
│         │  │   + Step     │                                        │
│         │  └──────────────┘                                        │
│         │                                                          │
│    ┌────┴────────────────────────────────┐                         │
│    │  vLLM Engine (optional)             │                         │
│    │  - Server mode (external process)   │                         │
│    │  - Colocate mode (shared GPU)       │                         │
│    │  Used for fast on-policy generation │                         │
│    └─────────────────────────────────────┘                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Three Training Modes

GKD defines three data sources via the `DataSource` enum (`gkd_trainer.py:45-48`):

```python
class DataSource(str, Enum):
    STUDENT = 'student'   # On-policy: student model generates responses
    TEACHER = 'teacher'   # Sequential KD: teacher model generates responses
    DATASET = 'dataset'   # Off-policy: use dataset responses
```

### Mode Selection Logic

```
                    training_step() called
                            │
                            ▼
                ┌───────────────────────┐
                │  _get_random_num()    │
                │  seed = args.seed +   │
                │         global_step   │
                └───────────┬───────────┘
                            │
                            ▼
                   ┌────────────────┐
                   │ rand ≤ λ ?     │──── Yes ──► DataSource.STUDENT
                   └────────┬───────┘             (on-policy generation)
                            │ No
                            ▼
                   ┌────────────────┐
                   │ seq_kd = True? │──── Yes ──► DataSource.TEACHER
                   └────────┬───────┘             (teacher generation)
                            │ No
                            ▼
                    DataSource.DATASET
                    (use dataset as-is)
```

**Key parameter:** `lmbda` (λ, default 0.5) controls how often on-policy student generation is used. At λ=0 training is purely off-policy; at λ=1 it is purely on-policy.

### Mode Details

| Mode | When | Generation Source | Use Case |
|---|---|---|---|
| **STUDENT** | `rand ≤ λ` | Student via vLLM or `model.generate()` | Reduces train-inference mismatch |
| **TEACHER** | `rand > λ` and `seq_kd=True` | Teacher via `model.generate()` | Traditional SeqKD |
| **DATASET** | `rand > λ` and `seq_kd=False` | None (use dataset responses) | Standard off-policy KD |

---

## 4. Training Step Flow

### HuggingFace Implementation (`gkd_trainer.py:306-390`)

```
training_step(model, inputs)
│
├── Phase 1: Data Source Selection & Generation
│   │
│   ├── [STUDENT mode] ─────────────────────────────────────────────┐
│   │   │                                                           │
│   │   ├── if use_vllm:                                            │
│   │   │   ├── _preprocess_inputs(inputs)   # remove responses     │
│   │   │   ├── _fast_infer(inputs)          # vLLM generation      │
│   │   │   │   ├── _move_model_to_vllm()    # sync weights         │
│   │   │   │   └── _infer_single_or_multi_turn()  # generate       │
│   │   │   └── _prepare_batch_inputs(encode_prompt_only=False)      │
│   │   │                                                           │
│   │   └── else (transformers generate):                           │
│   │       ├── _prepare_batch_inputs(encode_prompt_only=True)      │
│   │       ├── unwrap_model_for_generation()                       │
│   │       ├── generate_on_policy_outputs(model, ...)  ◄── SYNC    │
│   │       └── override input_ids, attention_mask, labels          │
│   │                                                               │
│   ├── [TEACHER mode] ────────────────────────────────────────────┐│
│   │   ├── _prepare_batch_inputs(encode_prompt_only=True)         ││
│   │   ├── unwrap_model_for_generation(teacher_model)             ││
│   │   ├── generate_on_policy_outputs(teacher, ...)  ◄── SYNC    ││
│   │   └── override input_ids, attention_mask, labels             ││
│   │                                                              ││
│   └── [DATASET mode] ───────────────────────────────────────────┐││
│       └── _prepare_batch_inputs(encode_prompt_only=False)       │││
│                                                                  │││
├── Phase 2: Mark data source ◄────────────────────────────────────┘││
│   encoded_inputs['_data_source'] = data_source                    ││
│                                                 ◄─────────────────┘│
│                                                 ◄──────────────────┘
├── Phase 3: Loss Computation (HFSFTTrainer.training_step → compute_loss)
│   │
│   ├── Student forward:  outputs_student = model(**inputs)    ◄── WITH grad
│   ├── Teacher forward:  outputs_teacher = teacher(**inputs)  ◄── NO grad
│   ├── Align vocab sizes (pad if different)
│   ├── Compute generalized_jsd_loss(student_logits, teacher_logits)
│   └── Add SFT loss if sft_alpha > 0 and not STUDENT mode
│
└── Phase 4: Backward pass (handled by HF Trainer)
```

### Detailed `compute_loss()` Flow (`gkd_trainer.py:152-270`)

```
compute_loss(model, inputs)
│
├── Extract _data_source from inputs
│
├── [Liger kernel path] ──────────────────────────────────────┐
│   ├── Get base models (exclude lm_head)                     │
│   ├── student_outputs = base_student(**inputs)               │
│   ├── teacher_outputs = base_teacher(**inputs)  [no_grad]    │
│   ├── Get hidden states (shifted by 1)                       │
│   ├── liger_jsd_loss(student_hidden, teacher_hidden)         │
│   └── Normalize by sequence length                           │
│                                                              │
├── [Standard path] ──────────────────────────────────────────┐│
│   ├── outputs_student = model(**inputs)          [with grad]││
│   │                                                         ││
│   ├── with torch.no_grad():                                 ││
│   │   └── outputs_teacher = teacher_model(**inputs)         ││
│   │                                                         ││
│   ├── Shift labels by 1, create mask (labels != -100)       ││
│   ├── Extract masked logits from both models                ││
│   │                                                         ││
│   ├── Vocab size alignment:                                 ││
│   │   if stu_dim < tea_dim:                                 ││
│   │       pad student, copy teacher extra logits            ││
│   │   elif stu_dim > tea_dim:                               ││
│   │       pad teacher, copy student extra logits            ││
│   │                                                         ││
│   ├── loss = generalized_jsd_loss(s_logits, t_logits, β)   ││
│   │                                                         ││
│   └── if sft_alpha > 0 and source != STUDENT:              ││
│       loss += sft_alpha * outputs_student.loss              ││
│                                                              ││
└── return loss ◄──────────────────────────────────────────────┘│
                ◄──────────────────────────────────────────────┘
```

---

## 5. Loss Computation

### Generalized JSD Loss (`gkd_trainer.py:468-537`)

The loss function generalizes KL divergence through a parameter β:

```
              ┌─────────────────────────────────────────────────┐
              │  Generalized JSD Loss                           │
              │                                                 │
              │  β = 0  →  KL(teacher ‖ student)  [Forward KL] │
              │  β = 0.5 → Symmetric JSD                       │
              │  β = 1  →  KL(student ‖ teacher)  [Reverse KL] │
              │                                                 │
              │  General case (0 < β < 1):                      │
              │                                                 │
              │  m = β·P_teacher + (1-β)·P_student              │
              │                                                 │
              │  JSD = β·KL(teacher ‖ m)                        │
              │      + (1-β)·KL(student ‖ m)                    │
              └─────────────────────────────────────────────────┘
```

**Implementation details:**

1. **Temperature scaling**: Both logit tensors divided by `temperature` (default 0.9)
2. **Chunked computation**: Processes in chunks of 512 tokens to reduce peak memory
3. **Log-space computation**: Uses `F.log_softmax` and `torch.logsumexp` for numerical stability
4. **Masking**: Only computes loss on non-padding tokens (labels != -100)

### Combined Loss

```
Total Loss = JSD_loss + sft_alpha × SFT_loss
                         ▲
                         │
                 (only if sft_alpha > 0 AND
                  data_source != STUDENT)
```

The SFT loss is **excluded for student-generated data** to avoid conflicting optimization objectives — the student's own generations are not ground truth.

---

## 6. Synchronous vs Asynchronous — The Core Question

### Answer: The GKD process is FULLY SYNCHRONOUS

All three phases — student inference, teacher soft label generation, and training backward pass — execute **sequentially within each training step** with no overlap or pipelining.

### Evidence from Code

#### Evidence 1: `training_step()` is a single sequential function

In `gkd_trainer.py:306-390`, the entire training step is a blocking, sequential function:

```python
def training_step(self, model, inputs, num_items_in_batch=None):
    # PHASE 1: Generation (BLOCKING)
    if self._get_random_num() <= self.lmbda:
        if args.use_vllm:
            generated_inputs = self._fast_infer(inputs)      # ◄── BLOCKS until done
            encoded_inputs = self._prepare_batch_inputs(...)
        else:
            new_ids, mask, labels = self.generate_on_policy_outputs(...)  # ◄── BLOCKS

    # PHASE 2 + 3: Loss + Backward (BLOCKING)
    loss = HFSFTTrainer.training_step(self, model, encoded_inputs)  # ◄── BLOCKS
    return loss
```

There is **no threading, no async/await, no Future** between these phases.

#### Evidence 2: `compute_loss()` runs teacher forward synchronously

In `gkd_trainer.py:230-236`:

```python
# Student forward (with gradients)
outputs_student = model(**model_inputs)              # ◄── BLOCKS

# Teacher forward (without gradients, IMMEDIATELY after student)
with torch.no_grad():
    outputs_teacher = self.teacher_model(**model_inputs)  # ◄── BLOCKS
```

Both forward passes execute on the same GPU sequentially. The teacher forward **waits** for the student forward to complete.

#### Evidence 3: `async_generate` is GRPO-only

The `RolloutTrainerMixin` (which GKD inherits) has async generation capability, but it is explicitly documented as GRPO-only:

```python
class RolloutTrainerMixin(RLHFTrainerMixin):
    """
    This mixin provides vLLM integration and rollout infrastructure.
    It should be used for trainers that require:
    - Policy rollout with vLLM engine (server or colocate mode)
    - Multi-turn dialogue support (GRPO only)
    - Async generation capabilities (GRPO only)     ◄── GRPO ONLY
    """
```

GKD does **not** set `self.async_generate = True`, so the async code path is never activated.

#### Evidence 4: Megatron GKD is also synchronous

In `megatron/trainers/gkd_trainer.py:449-492`, the `_replace_data_iterator()` method runs **all** operations synchronously before the training step:

```python
def _replace_data_iterator(self, data_iterator, model):
    data_source = self._determine_data_source()           # 1. Choose mode

    for _ in range(num_microbatches):
        raw_batch = next(data_iterator)                    # 2. Get data
        global_batch.extend(raw_batch)

    if data_source == DataSource.STUDENT:
        local_batch = self._generate_completions(batch)    # 3. Generate (SYNC)
        global_batch = self._gather_rollout_results(batch)

    for i in range(num_microbatches):
        encoded_batch = self._encode_batch(raw_batch)      # 4. Encode
        encoded_batches.append(encoded_batch)

    self._compute_teacher_logits(encoded_batches)          # 5. Teacher fwd (SYNC)

    return RerunDataIterator(iter(encoded_batches))         # 6. Return for training
```

Everything happens before `forward_step()` is called.

### Timeline Visualization

```
            ┌──────── One Training Step ─────────────────────────────────────┐
            │                                                                 │
Time ──────►│                                                                 │
            │                                                                 │
Phase 1     │  ████████████████████                                          │
(Generate)  │  Student generates   │                                          │
            │  via vLLM/generate() │                                          │
            │  (if on-policy)      │                                          │
            │                      │                                          │
Phase 2a    │                      ████████████                               │
(Student    │                      │ Student    │                              │
 Forward)   │                      │ forward()  │                              │
            │                      │ (with grad)│                              │
            │                                   │                              │
Phase 2b    │                                   █████████████                  │
(Teacher    │                                   │ Teacher     │                │
 Forward)   │                                   │ forward()   │                │
            │                                   │ (no_grad)   │                │
            │                                                 │                │
Phase 3     │                                                 █████████████   │
(JSD Loss   │                                                 │ JSD + Back │   │
 +Backward) │                                                 │ propagation│   │
            │                                                              │   │
            └──────────────────────────────────────────────────────────────────┘

              ◄─────────── ALL SEQUENTIAL, NO OVERLAP ──────────────────────►
```

### Why Synchronous?

1. **Data dependency**: The JSD loss requires **both** student and teacher logits on the **same input**. Teacher forward cannot start until the input is prepared (which may include student-generated text).

2. **Memory constraint**: Running student and teacher forwards simultaneously would require storing both sets of activations, roughly doubling peak GPU memory.

3. **Simplicity**: The sequential design avoids complex synchronization logic, race conditions, and non-determinism.

### Contrast with GRPO (which IS partially asynchronous)

For reference, GRPO in ms-swift **does** support async generation:

```
GRPO (async=True):
  Step N generation ──►  overlaps with  ──► Step N-1 training
  (prefetch next batch)                     (backward + optimizer step)

GKD:
  Step N generation ──► Step N training ──► Step N+1 generation ──► ...
  (no overlap, fully sequential)
```

---

## 7. Memory Optimization Strategies

### Teacher Model CPU Offloading

When `offload_teacher_model=True`, the teacher is moved to CPU except during its forward pass:

```
┌───────────────────────────────────────────────────┐
│ Training Step Timeline (with teacher offloading)  │
│                                                   │
│  [CPU]  Teacher on CPU ████████          █████████│
│  [GPU]                         ██████████         │
│                                ▲        ▲         │
│                          load_model  offload      │
│                          to GPU      to CPU       │
│                                                   │
│  [GPU]  Student ████████████████████████████████  │
│         (always on GPU)                           │
└───────────────────────────────────────────────────┘
```

Implemented via `load_teacher_model_context()` (`gkd_trainer.py:437-448`).

### Student/Optimizer Offloading During vLLM Inference

When using vLLM colocate mode, the student model and optimizer can be offloaded to CPU to free GPU memory for vLLM:

```python
@contextmanager
def offload_context(self):       # gkd_trainer.py:398-420
    if self.args.offload_model:
        self.offload_model(self.model)
    if self.args.offload_optimizer:
        self.offload_optimizer()
    yield                        # vLLM uses GPU here
    if self.args.offload_model:
        self.load_model(self.model)
    if self.args.offload_optimizer:
        self.load_optimizer()
```

### Liger Kernel Fused JSD Loss

When `use_liger_kernel=True`, the JSD loss is computed using LinkedIn's Liger kernel (`LigerFusedLinearJSDLoss`), which:
- Operates on hidden states instead of logits (avoids materializing full logit tensor)
- Uses chunked computation with fused linear + loss
- Reduces peak memory significantly

Limitation: Incompatible with `sft_alpha > 0`.

---

## 8. Megatron GKD Implementation

The Megatron implementation (`swift/megatron/trainers/gkd_trainer.py`) supports distributed training with Pipeline Parallelism (PP), Tensor Parallelism (TP), and Context Parallelism (CP).

### Key Differences from HF Implementation

| Aspect | HF GKD | Megatron GKD |
|---|---|---|
| Teacher loading | Via HF `AutoModel` + DeepSpeed/FSDP | Via Megatron `get_model()` with PP/TP |
| Teacher forward | Direct `teacher_model(**inputs)` | `forward_step_helper()` with args override |
| JSD loss | Standard `F.log_softmax` | `vocab_parallel_log_softmax` (TP-aware) |
| KL computation | `F.kl_div` | `vocab_parallel_kl_div` (TP-aware) |
| On-policy gen | vLLM or `model.generate()` | vLLM only (via `MegatronRolloutMixin`) |
| Seq KD | Supported | Not yet implemented (falls back to DATASET) |

### Megatron Training Step Flow

```
_replace_data_iterator(data_iterator, model)
│
├── _determine_data_source()               # Choose mode
├── Collect all micro-batches
├── [If STUDENT]: _generate_completions()  # vLLM generation
├── _encode_batch() for each micro-batch
├── _compute_teacher_logits()              # Teacher forward (ALL micro-batches)
│   │
│   └── for each encoded_batch:
│       ├── _teacher_args_context()        # Override Megatron args for teacher arch
│       ├── forward_step_helper(teacher_model, teacher_data)  [no_grad]
│       └── store teacher_logits in batch
│
└── Return RerunDataIterator(encoded_batches)
    │
    ▼
forward_step(data_iterator, model)
    ├── Pop teacher_logits and data_source from batch
    ├── student_output = model(**data)      # Student forward
    └── return (student_output, partial(loss_func, ...))
        │
        ▼
    loss_func()
        ├── generalized_jsd_loss()          # Vocab-parallel JSD
        ├── Optional SFT loss
        └── All-reduce across CP group
```

### Teacher Architecture Override

When student and teacher have different architectures (e.g., student is Dense, teacher is MoE), Megatron's global args must be temporarily overridden:

```python
@contextmanager
def _teacher_args_context(self):
    # Save student config → Apply teacher config → yield → Restore student config
```

This handles differences in `hidden_size`, `num_layers`, `num_experts`, etc.

---

## 9. Configuration Reference

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--rlhf_type` | — | Must be `gkd` |
| `--model` | — | Student model path |
| `--teacher_model` | — | Teacher model path (required) |
| `--temperature` | `0.9` | Softmax temperature for logit scaling |
| `--lmbda` | `0.5` | On-policy probability (0=off-policy, 1=on-policy) |
| `--beta` | `0.5` | JSD interpolation (0=fwd KL, 0.5=JSD, 1=rev KL) |
| `--seq_kd` | `False` | Enable teacher generation mode |
| `--sft_alpha` | `0` | Weight for auxiliary SFT loss |
| `--offload_teacher_model` | `False` | Offload teacher to CPU between forward passes |
| `--use_vllm` | `False` | Use vLLM for on-policy student generation |
| `--vllm_mode` | `server` | vLLM mode: `server` or `colocate` |
| `--max_completion_length` | `512` | Max tokens for on-policy generation |
| `--teacher_deepspeed` | `None` | Separate DeepSpeed config for teacher |
| `--use_liger_kernel` | `False` | Use Liger fused JSD loss |

### Argument Sources

- `swift/arguments/rlhf_args.py` — `TeacherModelArguments`, `RLHFArguments`
- `swift/rlhf_trainers/arguments.py` — `GKDConfig`

---

## 10. Source File Index

| File | Lines | Role |
|---|---|---|
| `swift/rlhf_trainers/gkd_trainer.py` | 606 | Main GKD trainer (HF-based) |
| `swift/rlhf_trainers/rollout_mixin.py` | 1622 | vLLM integration, rollout infrastructure |
| `swift/megatron/trainers/gkd_trainer.py` | 713 | Megatron GKD trainer |
| `swift/pipelines/train/rlhf.py` | 241 | Model loading, pipeline orchestration |
| `swift/arguments/rlhf_args.py` | ~240 | RLHF/GKD argument definitions |
| `swift/rlhf_trainers/arguments.py` | ~80 | GKDConfig training arguments |
| `swift/trainers/trainer_factory.py` | ~75 | Maps `gkd` → `GKDTrainer` |
| `swift/megatron/trainers/vocab_parallel_utils.py` | — | TP-aware JSD/KL operations |
| `examples/train/rlhf/gkd/` | — | Example training scripts |

---

## Summary

GKD in ms-swift is a **synchronous, single-step knowledge distillation** framework that:

1. **Mixes data sources** probabilistically (on-policy student generation vs. off-policy dataset)
2. **Computes JSD loss** between student and teacher logits with configurable β
3. Executes **all three phases sequentially** within each training step: generation → dual forward pass → backward
4. Supports **two backends**: HuggingFace (DeepSpeed/FSDP) and Megatron (PP/TP/CP)
5. Provides **memory optimizations**: teacher offloading, student offloading during vLLM, Liger kernel fusion

The fully synchronous design is a deliberate architectural choice driven by data dependencies (JSD requires both logit tensors on the same input) and memory constraints (avoiding simultaneous activation storage). This contrasts with GRPO, which supports asynchronous generation pipelining.
