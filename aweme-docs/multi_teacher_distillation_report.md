# Multi-Teacher GKD: Feasibility Report & Implementation Guide

> Upgrade ms-swift's GKD trainer to support multiple domain-specialized teachers with per-sample routing, keeping the existing JSD loss.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current GKD Architecture](#2-current-gkd-architecture)
3. [Data Flow: How `channel` Propagates](#3-data-flow-how-channel-propagates)
4. [Gap Analysis](#4-gap-analysis)
5. [Step-by-Step Implementation Guide](#5-step-by-step-implementation-guide)
6. [Per-Sample Routing Within a Batch](#6-per-sample-routing-within-a-batch)
7. [Example Multi-Teacher Training Script](#7-example-multi-teacher-training-script)
8. [Challenges & Mitigations](#8-challenges--mitigations)

---

## 1. Executive Summary

### Goal

Support **N domain-specialized teacher models** in GKD, where each training sample is routed to the correct teacher based on a `channel` field in the data. Within a single batch containing mixed-domain samples, each sample gets its soft label from its assigned teacher. The JSD loss is then computed over the gathered teacher logits as if they came from one teacher.

### Verdict: FEASIBLE with ~225 lines of changes across 3 files

| What | Status |
|---|---|
| `channel` field flows through the full pipeline | Confirmed safe (used only for metric logging, not loss computation) |
| Multi-model loading loop blueprint exists | `rlhf.py:149` iterates over reward models — same pattern for teachers |
| Teacher CPU offloading exists | `gkd_trainer.py:437-448` — extend to indexed multi-teacher offloading |
| vLLM colocate/server mode | Unaffected (student generates, not teacher) |
| `--teacher_domain_map` maps channel→path directly | No positional index coupling; model types auto-detected |

### What stays the same

- JSD loss (no MOPD RL loss, no ORM, no importance weighting)
- `generalized_jsd_loss()` function — unchanged
- vLLM rollout infrastructure — student generates on-policy as before
- All three training modes: on-policy (student), seq_kd (teacher), off-policy (dataset)
- Liger kernel support (with routing to one teacher per batch for liger path)

---

## 2. Current GKD Architecture

### 2.1 Training Flow

```
CLI (single-teacher):  swift rlhf --rlhf_type gkd --teacher_model <path> --dataset <name>
CLI (multi-teacher):   swift rlhf --rlhf_type gkd --teacher_domain_map '{"math":"/path/a","code":"/path/b"}' --dataset <name>
  │
  ├─ RLHFArguments.__post_init__()          [rlhf_args.py:245]
  │   ├─ _check_gkd(): validates mutual exclusivity, parses JSON,
  │   │   deduplicates paths → builds _teacher_paths + _channel_to_teacher_idx
  │   └─ TeacherModelArguments.teacher_model: Optional[str]  ← SINGLE model (backward compat)
  │
  ├─ SwiftRLHF._load_teacher_models()       [rlhf.py — NEW]
  │   ├─ Single-teacher: loads via teacher_model, wraps in list
  │   └─ Multi-teacher: iterates _teacher_paths, loads each unique teacher once
  │
  ├─ SwiftRLHF._get_trainer_kwargs()        [rlhf.py:219]
  │   └─ Passes teacher_model (list) + channel_to_teacher_idx + teacher_deepspeed_config
  │
  ├─ TrainerFactory: 'gkd' → GKDTrainer     [trainer_factory.py:27]
  │
  └─ GKDTrainer                             [gkd_trainer.py:51]
      ├─ __init__: wraps ALL teachers with DS/FSDP, stores channel_to_teacher_idx
      ├─ training_step: decides data source → _prepare_batch_inputs → HFSFTTrainer.training_step
      ├─ compute_loss: student forward + PER-SAMPLE teacher routing → JSD
      ├─ _multi_teacher_forward: groups by teacher, runs subsets, scatters logits
      └─ generalized_jsd_loss: β-parameterized JSD between logits (UNCHANGED)
```

### 2.2 Key Code Paths

**Teacher initialization** (`gkd_trainer.py:53-91`):
```python
teacher_model = kwargs.pop('teacher_model')  # ONE model
# Wrap with DeepSpeed/FSDP/accelerator
if self.is_deepspeed_enabled:
    self.teacher_model = prepare_deepspeed(teacher_model, ...)
elif self.is_fsdp_enabled:
    self.teacher_model = prepare_fsdp(teacher_model, ...)
else:
    self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)
self.teacher_model.eval()
```

**Loss computation** (`gkd_trainer.py:225-262`, standard path):
```python
# Student forward (full batch)
outputs_student = model(**model_inputs)

# Teacher forward (full batch, no grad)
with torch.no_grad():
    outputs_teacher = self.teacher_model(**model_inputs)  # ONE teacher

# Mask and flatten
shifted_labels = torch.roll(inputs['labels'], shifts=-1, dims=1)
mask = shifted_labels != -100
shifted_student_logits = outputs_student.logits[mask][None]   # [1, N_valid, V]
shifted_teacher_logits = outputs_teacher.logits[mask][None]   # [1, N_valid, V]

# Vocab alignment + JSD
loss = self.generalized_jsd_loss(student_logits=..., teacher_logits=..., beta=self.beta)
```

**Critical insight**: The mask flattens across the batch dimension (`outputs.logits[mask]` produces `[N_valid_tokens, vocab]`). This means if we scatter the correct teacher logits into the full `[batch, seq, vocab]` tensor BEFORE masking, the flattening naturally produces correct per-sample teacher logits.

---

## 3. Data Flow: How `channel` Propagates

The `channel` field is an **existing first-class field** in ms-swift, designed for per-channel loss logging. It is safe to reuse for teacher routing.

### 3.1 End-to-End Flow

| Stage | File | What happens to `channel` |
|---|---|---|
| JSONL data | User data | `{"prompt": "...", "response": "...", "channel": "math"}` |
| Dataset loading | `preprocessor/core.py:32` | `channel` IS in `standard_keys` → **preserved** |
| `StdTemplateInputs.from_dict()` | `template_inputs.py:61-63` | Explicitly extracted: `kwargs['channel'] = inputs['channel']` |
| `template.encode()` | `base.py:563-564` | Re-exported: `if chosen.channel is not None: encoded['channel'] = chosen.channel` |
| `template.data_collator()` | `base.py:1746-1753` | Collated into list: `channel = [b.get('channel') for b in batch]` → `res['channel'] = channel` |
| GKD `_prepare_batch_inputs` | `gkd_trainer.py:298-301` | Calls `template.encode()` + `template.data_collator()` → `channel` is in batch dict |
| GKD `compute_loss` | `gkd_trainer.py:156` | Available in `inputs['channel']` → pop before model forward |

### 3.2 Evidence: `channel` Does Not Affect Loss

In `seq2seq_trainer.py:129-177`:
```python
channels = inputs.pop('channel', None)  # Popped, NOT passed to model

# Only used for METRIC logging when enable_channel_loss=True:
if self.args.enable_channel_loss and channels is not None:
    # Logs per-channel loss breakdown — does NOT modify loss value
    metrics[channel_name] = channel_loss.item()
```

**Conclusion**: `channel` is purely a metadata/metric field. Using it for teacher routing has zero side effects.

### 3.3 What the User Needs in Their Data

Each JSONL row should include a `channel` field:
```jsonl
{"prompt": "Solve x^2 + 3x - 4 = 0", "response": "...", "channel": "math"}
{"prompt": "Write a Python quicksort", "response": "...", "channel": "code"}
{"prompt": "Summarize this article", "response": "...", "channel": "general"}
```

The `custom_dataset_info.json` stays the same — no changes needed there.

---

## 4. Gap Analysis

### What Exists

| Capability | Location | Evidence |
|---|---|---|
| `channel` field end-to-end | `core.py:32`, `template_inputs.py:24`, `base.py:563`, `base.py:1746` | Flows from JSONL → template → batch dict |
| List-based model args | `rlhf_args.py:29` | `reward_model: Optional[List[str]]` |
| Multi-model loading loop | `rlhf.py:138-164` | Iterates over reward models |
| Teacher DeepSpeed/FSDP wrapping | `gkd_trainer.py:75-88` | Works per-model |
| Teacher CPU offloading | `gkd_trainer.py:90-91, 437-448` | `offload_model` / `load_model` |
| vLLM colocate/server | `rollout_mixin.py` via `RolloutTrainerMixin` | Student-only generation |
| Vocab size alignment | `gkd_trainer.py:243-251` | Handles different vocab sizes |

### What's Missing

| Capability | Estimated Lines |
|---|---|
| `teacher_domain_map` CLI arg + validation + dedup | ~30 lines new in `rlhf_args.py` |
| Multi-teacher loading in pipeline | ~40 lines new in `rlhf.py` |
| Multi-teacher init in GKDTrainer | ~30 lines modified in `gkd_trainer.py` |
| Per-sample routing in `compute_loss` | ~60 lines new in `gkd_trainer.py` |
| Indexed teacher offload context | ~15 lines new in `gkd_trainer.py` |
| `_get_teacher_indices` + `_multi_teacher_forward` | ~50 lines new in `gkd_trainer.py` |
| **Total** | **~225 lines** |

**Key design decision**: `--teacher_domain_map` maps channel names directly to model paths (not integer indices). This eliminates positional coupling with `--teacher_model` and makes the CLI self-documenting. `--teacher_model` (single teacher) and `--teacher_domain_map` (multi-teacher) are mutually exclusive.

---

## 5. Step-by-Step Implementation Guide

### Phase 1: Arguments

#### File: `swift/arguments/rlhf_args.py`

**Step 1.1**: `TeacherModelArguments` — NO changes needed

The existing `teacher_model: Optional[str]` stays as-is for **single-teacher backward compatibility**. Multi-teacher model paths are specified entirely via `teacher_domain_map`.

```python
# UNCHANGED — single-teacher backward compat
@dataclass
class TeacherModelArguments:
    teacher_model: Optional[str] = None
    teacher_adapters: List[str] = field(default_factory=list)
    teacher_model_type: Optional[str] = field(...)
    teacher_model_revision: Optional[str] = None
    teacher_deepspeed: Optional[str] = field(...)
```

**Step 1.2**: Add `teacher_domain_map` to `RLHFArguments` (after line 237)

The value is a JSON string mapping channel names **directly to model paths** — no positional index coupling:

```python
    # Multi-teacher GKD routing
    teacher_domain_map: Optional[str] = field(
        default=None,
        metadata={'help': 'JSON string mapping data channel names to teacher model paths. '
                  'e.g. \'{"math": "/path/to/math_teacher", "code": "/path/to/code_teacher"}\'. '
                  'Mutually exclusive with --teacher_model (single-teacher).'})
```

**Step 1.3**: Add validation in `_check_gkd` (line 545)

```python
def _check_gkd(self):
    if self.rlhf_type != 'gkd':
        return
    # ... existing checks ...

    # Parse and validate teacher_domain_map
    if self.teacher_domain_map is not None:
        import json
        if isinstance(self.teacher_domain_map, str):
            self.teacher_domain_map = json.loads(self.teacher_domain_map)
        assert isinstance(self.teacher_domain_map, dict), (
            f'teacher_domain_map must be a JSON dict, got {type(self.teacher_domain_map)}')
        for domain, model_path in self.teacher_domain_map.items():
            assert isinstance(domain, str) and isinstance(model_path, str), (
                f'teacher_domain_map values must be strings (model paths), '
                f'got {domain!r}: {model_path!r}')

    # Mutual exclusivity
    if self.teacher_domain_map and self.teacher_model:
        raise ValueError(
            '--teacher_domain_map and --teacher_model are mutually exclusive. '
            'Use --teacher_domain_map for multi-teacher, --teacher_model for single-teacher.')

    if self.teacher_domain_map is None and self.teacher_model is None:
        raise ValueError('GKD requires either --teacher_model (single) or --teacher_domain_map (multi).')

    # Deduplicate: extract unique teacher paths and build channel→index mapping
    if self.teacher_domain_map is not None:
        unique_paths = []
        path_to_idx = {}
        for domain, model_path in self.teacher_domain_map.items():
            if model_path not in path_to_idx:
                path_to_idx[model_path] = len(unique_paths)
                unique_paths.append(model_path)
        self._teacher_paths = unique_paths
        self._channel_to_teacher_idx = {
            domain: path_to_idx[path] for domain, path in self.teacher_domain_map.items()
        }
        logger.info(f'Multi-teacher GKD: {len(unique_paths)} unique teacher(s) '
                    f'for {len(self.teacher_domain_map)} domain(s)')
        logger.info(f'  Channel→teacher mapping: {self._channel_to_teacher_idx}')
```

**Design rationale**: By mapping `channel → model_path` directly, users never need to worry about matching positional indices between `--teacher_model` order and `--teacher_domain_map` integers. Multiple domains can share the same teacher model path — deduplication ensures it's loaded only once.

#### File: `swift/rlhf_trainers/arguments.py`

**Step 1.4**: No changes needed to `GKDConfig` — the `teacher_domain_map` is consumed at the `RLHFArguments` level and does not need to be in HuggingFace `TrainingArguments`.

---

### Phase 2: Pipeline — Multi-Teacher Loading

#### File: `swift/pipelines/train/rlhf.py`

**Step 2.1**: Replace teacher loading in `_prepare_model_tokenizer` (line 112-134)

The existing code handles teacher as a single model in the `for key in ['ref', 'value', 'teacher']` loop. Replace the teacher branch with a new `_load_teacher_models` method that supports both single-teacher (`--teacher_model`) and multi-teacher (`--teacher_domain_map`):

```python
def _prepare_model_tokenizer(self):
    args = self.args
    for key in ['ref', 'value', 'teacher']:
        setattr(self, f'{key}_model', None)
        if key == 'ref' and args.rlhf_type == 'gkd':
            continue
        if key == 'value' and args.rlhf_type != 'ppo':
            continue
        if key == 'teacher' and args.rlhf_type != 'gkd':
            continue

        if key == 'teacher':
            # Single-teacher or multi-teacher loading
            self._load_teacher_models()
            continue

        # Existing single-model loading for ref/value...
        model_key = 'reward' if key == 'value' else key
        model_type = getattr(args, f'{model_key}_model_type')
        model_revision = getattr(args, f'{model_key}_model_revision')
        if key == 'value':
            model_type = model_type[0] if model_type else None
            model_revision = model_revision[0] if model_revision else None
        result = self._prepare_single_model(model_key, key, model_type, model_revision)
        if result is not None:
            model, _ = result
            setattr(self, f'{key}_model', model)

    # Reward model handling unchanged...
    self.reward_model = None
    # ... existing code ...
    super()._prepare_model_tokenizer()

def _load_teacher_models(self):
    """Load teacher model(s). Supports two modes:

    1. Single-teacher (--teacher_model): loads one teacher, wraps in list
    2. Multi-teacher (--teacher_domain_map): parses domain→path JSON,
       deduplicates paths, loads each unique teacher once
    """
    args = self.args

    if args.teacher_domain_map is not None:
        # Multi-teacher mode: paths come from teacher_domain_map
        # _teacher_paths and _channel_to_teacher_idx were built in _check_gkd()
        teacher_paths = args._teacher_paths
        teacher_models = []
        original_teacher_model = args.teacher_model

        for teacher_path in teacher_paths:
            args.teacher_model = teacher_path  # Temporarily set for _prepare_single_model
            # model_type=None triggers auto-detection in _prepare_single_model
            result = self._prepare_single_model('teacher', 'teacher', None, None)
            if result is not None:
                model, _ = result
                teacher_models.append(model)

        args.teacher_model = original_teacher_model  # Restore
        self.teacher_model = teacher_models
        logger.info(f'Multi-teacher GKD: loaded {len(teacher_models)} unique teacher(s) '
                    f'for {len(args.teacher_domain_map)} domain(s)')
    else:
        # Single-teacher mode (backward compat): load via existing path
        result = self._prepare_single_model(
            'teacher', 'teacher', args.teacher_model_type, args.teacher_model_revision)
        if result is not None:
            model, _ = result
            self.teacher_model = [model]  # Wrap in list for uniform handling
        else:
            self.teacher_model = None
```

**Step 2.2**: Update `_get_trainer_kwargs` (line 219-236)

Pass teacher models as a list + the resolved `_channel_to_teacher_idx` mapping:

```python
def _get_trainer_kwargs(self):
    trainer_kwargs = {}
    for key in ['ref', 'reward', 'value', 'teacher']:
        key_name = f'{key}_model'
        model = getattr(self, key_name, None)
        if key == 'teacher' and model is not None:
            # Always pass as list
            trainer_kwargs['teacher_model'] = model if isinstance(model, list) else [model]
            # Pass the resolved channel→index mapping (or None for single teacher)
            trainer_kwargs['channel_to_teacher_idx'] = getattr(
                self.args, '_channel_to_teacher_idx', None)
            continue
        if model or self.args.rlhf_type == 'ppo' and key_name != 'teacher_model':
            trainer_kwargs[key_name] = model
    # ... rest unchanged ...
    if self.args.rlhf_type in ['grpo', 'gkd']:
        trainer_kwargs['vllm_client'] = self.args.vllm_client
    if self.args.rlhf_type == 'gkd' and self.args.teacher_deepspeed:
        trainer_kwargs['teacher_deepspeed_config'] = self.args.teacher_deepspeed
    return trainer_kwargs
```

---

### Phase 3: GKD Trainer — Multi-Teacher Init + Routing

#### File: `swift/rlhf_trainers/gkd_trainer.py`

**Step 3.1**: Modify `__init__` (line 53-107) to accept and initialize multiple teachers

The pipeline now always passes `teacher_model` as a list and `channel_to_teacher_idx` as the resolved mapping:

```python
def __init__(self, model=None, *_args, **kwargs):
    teacher_models_input = kwargs.pop('teacher_model')  # Always a list now
    teacher_deepspeed_config = kwargs.pop('teacher_deepspeed_config', None)
    # channel_to_teacher_idx: Dict[str, int] or None
    self.channel_to_teacher_idx = kwargs.pop('channel_to_teacher_idx', None)
    self.vllm_client = kwargs.pop('vllm_client', None)
    super().__init__(model, None, *_args, **kwargs)
    args = kwargs['args']

    # Existing params
    self.lmbda = args.lmbda
    self.temperature = args.temperature
    self.seq_kd = args.seq_kd
    self.generation_config = model.generation_config
    self._metrics = {'train': defaultdict(list), 'eval': defaultdict(list)}
    self._total_train_tokens = 0

    self._prepare_logging()
    self._prepare_liger_loss()

    self.teacher_ds3_gather_for_generation = args.ds3_gather_for_generation

    # Initialize ALL teacher models
    if not isinstance(teacher_models_input, list):
        teacher_models_input = [teacher_models_input]

    self.teacher_models = []
    self.is_teacher_ds3_list = []

    for tm in teacher_models_input:
        if self.is_deepspeed_enabled:
            if teacher_deepspeed_config is not None:
                is_ds3 = teacher_deepspeed_config.get('zero_optimization', {}).get('stage') == 3
                self.is_teacher_ds3_list.append(is_ds3)
                prepared = prepare_deepspeed(
                    tm, self.accelerator,
                    deepspeed_config=teacher_deepspeed_config, training_args=args)
            else:
                self.is_teacher_ds3_list.append(None)
                prepared = prepare_deepspeed(tm, self.accelerator)
        elif self.is_fsdp_enabled:
            from .utils import prepare_fsdp
            self.is_teacher_ds3_list.append(False)
            prepared = prepare_fsdp(tm, self.accelerator)
        else:
            self.is_teacher_ds3_list.append(False)
            prepared = self.accelerator.prepare_model(tm, evaluation_mode=True)
        prepared.eval()
        self.teacher_models.append(prepared)

    # Backward compat aliases (single-teacher code paths still use self.teacher_model)
    self.teacher_model = self.teacher_models[0]
    self.is_teacher_ds3 = self.is_teacher_ds3_list[0] if self.is_teacher_ds3_list else None

    logger.info(f'Initialized {len(self.teacher_models)} teacher model(s)')
    if self.channel_to_teacher_idx:
        logger.info(f'Channel→teacher routing: {self.channel_to_teacher_idx}')

    if args.offload_teacher_model:
        for tm in self.teacher_models:
            self.offload_model(self.accelerator.unwrap_model(tm))

    # ... rest of existing init unchanged ...
```

**Step 3.2**: Add indexed teacher offload context manager (new method)

```python
@contextmanager
def _load_single_teacher_context(self, teacher_idx):
    """Load/offload a specific teacher. Ensures only ONE teacher on GPU at a time."""
    if not self.args.offload_teacher_model:
        yield
        return
    teacher = self.accelerator.unwrap_model(self.teacher_models[teacher_idx])
    self.load_model(teacher)
    try:
        yield
    finally:
        self.offload_model(teacher)

@contextmanager
def load_teacher_model_context(self):
    """Backward-compatible: loads the first teacher."""
    with self._load_single_teacher_context(0):
        yield
```

**Step 3.3**: Add channel-to-teacher routing helper (new method)

Uses `self.channel_to_teacher_idx` — the resolved `Dict[str, int]` mapping that was built during argument validation (deduplication) and passed from the pipeline:

```python
def _get_teacher_indices(self, channels):
    """Map channel values to teacher indices.

    Args:
        channels: list of channel strings (one per sample in batch), or None

    Returns:
        list of int teacher indices, or None if no routing needed
    """
    if channels is None or self.channel_to_teacher_idx is None or len(self.teacher_models) == 1:
        return None  # Use first/only teacher for all

    indices = []
    for ch in channels:
        if ch is not None and ch in self.channel_to_teacher_idx:
            indices.append(self.channel_to_teacher_idx[ch])
        else:
            indices.append(0)  # Default to first teacher
    return indices
```

---

### Phase 4: Per-Sample Routing in `compute_loss`

This is the core change. See [Section 6](#6-per-sample-routing-within-a-batch) for the detailed algorithm.

#### File: `swift/rlhf_trainers/gkd_trainer.py`

**Step 4.1**: Modify `compute_loss` (line 152-270) — standard path

```python
@patch_profiling_decorator
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    data_source = inputs.pop('_data_source', DataSource.DATASET)
    channels = inputs.pop('channel', None)  # NEW: extract channel for routing
    model_inputs = {k: v for k, v in inputs.items() if k not in {'prompt', 'labels'}}

    use_logits_to_keep = self.get_use_logits_to_keep(True)
    if use_logits_to_keep and not self.use_liger_gkd_loss:
        self.prepare_logits_to_keep(inputs)
        model_inputs['logits_to_keep'] = inputs['logits_to_keep']

    # Determine per-sample teacher routing
    teacher_indices = self._get_teacher_indices(channels)

    if self.use_liger_gkd_loss:
        # Liger path: use first teacher or majority teacher for the batch
        # (Liger kernel is monolithic — can't easily split by teacher)
        teacher_idx = 0
        if teacher_indices is not None:
            from collections import Counter
            teacher_idx = Counter(teacher_indices).most_common(1)[0][0]
        teacher_model = self.teacher_models[teacher_idx]

        # ... existing liger code, replacing self.teacher_model with teacher_model ...
        # (same logic, just use the selected teacher_model variable)
        # [existing lines 163-224 with self.teacher_model → teacher_model]

    else:
        # Standard path
        if self.args.sft_alpha > 0:
            model_inputs['labels'] = inputs['labels']
        outputs_student = model(**model_inputs)
        model_inputs.pop('labels', None)

        if teacher_indices is None or len(set(teacher_indices)) == 1:
            # Single teacher for all samples (original behavior)
            teacher_idx = teacher_indices[0] if teacher_indices else 0
            teacher_model = self.teacher_models[teacher_idx]
            load_ctx = (self._load_single_teacher_context(teacher_idx)
                       if self.args.offload_teacher_model else nullcontext())
            with torch.no_grad(), load_ctx, disable_gradient_checkpointing(
                    teacher_model, self.args.gradient_checkpointing_kwargs):
                outputs_teacher = teacher_model(**model_inputs)
            teacher_logits = outputs_teacher.logits
        else:
            # PER-SAMPLE ROUTING: multiple teachers in one batch
            teacher_logits = self._multi_teacher_forward(
                model_inputs, teacher_indices)

        shifted_labels = torch.roll(inputs['labels'], shifts=-1, dims=1)
        mask = shifted_labels != -100
        shifted_student_logits = outputs_student.logits[mask][None]
        shifted_teacher_logits = teacher_logits[mask][None]

        # Vocab alignment (existing code)
        stu_dim = shifted_student_logits.shape[-1]
        tea_dim = shifted_teacher_logits.shape[-1]
        if stu_dim < tea_dim:
            shifted_student_logits = F.pad(
                shifted_student_logits, (0, tea_dim - stu_dim), 'constant', 0)
            shifted_student_logits[..., stu_dim:] = shifted_teacher_logits[..., stu_dim:]
        elif stu_dim > tea_dim:
            shifted_teacher_logits = F.pad(
                shifted_teacher_logits, (0, stu_dim - tea_dim), 'constant', 0)
            shifted_teacher_logits[..., tea_dim:] = shifted_student_logits[..., tea_dim:]

        loss = self.generalized_jsd_loss(
            student_logits=shifted_student_logits,
            teacher_logits=shifted_teacher_logits,
            beta=self.beta,
        )
        if self.args.sft_alpha > 0 and data_source != DataSource.STUDENT:
            loss = loss + self.args.sft_alpha * outputs_student.loss

    if return_outputs:
        return (loss, outputs_student if not self.use_liger_gkd_loss else None)
    return loss
```

**Step 4.2**: Add `_multi_teacher_forward` (new method)

```python
def _multi_teacher_forward(self, model_inputs, teacher_indices):
    """Run per-sample teacher routing: each sample's logits come from its assigned teacher.

    Args:
        model_inputs: dict with tensors of shape [batch, seq, ...]
        teacher_indices: list[int] of length batch_size

    Returns:
        teacher_logits: tensor [batch, seq, vocab] with per-sample teacher logits
    """
    batch_size = model_inputs['input_ids'].shape[0]
    teacher_logits = None

    # Group samples by teacher
    from collections import defaultdict
    teacher_to_samples = defaultdict(list)
    for sample_idx, teacher_idx in enumerate(teacher_indices):
        teacher_to_samples[teacher_idx].append(sample_idx)

    for teacher_idx, sample_idxs in teacher_to_samples.items():
        sample_idxs_tensor = torch.tensor(sample_idxs, device=model_inputs['input_ids'].device)

        # Extract subset of batch for this teacher
        subset_inputs = {}
        for k, v in model_inputs.items():
            if isinstance(v, torch.Tensor) and v.shape[0] == batch_size:
                subset_inputs[k] = v[sample_idxs_tensor]
            else:
                subset_inputs[k] = v

        teacher_model = self.teacher_models[teacher_idx]
        load_ctx = (self._load_single_teacher_context(teacher_idx)
                   if self.args.offload_teacher_model else nullcontext())
        with torch.no_grad(), load_ctx, disable_gradient_checkpointing(
                teacher_model, self.args.gradient_checkpointing_kwargs):
            subset_outputs = teacher_model(**subset_inputs)

        # Initialize output tensor on first iteration
        if teacher_logits is None:
            seq_len = subset_outputs.logits.shape[1]
            vocab_size = subset_outputs.logits.shape[2]
            teacher_logits = torch.zeros(
                batch_size, seq_len, vocab_size,
                dtype=subset_outputs.logits.dtype,
                device=subset_outputs.logits.device)

        # Scatter subset logits back into full batch
        teacher_logits[sample_idxs_tensor] = subset_outputs.logits

    return teacher_logits
```

**Step 4.3**: Update `training_step` for seq_kd with multi-teacher (line 356-376)

When `seq_kd=True`, the teacher generates responses. With multi-teacher, route to the correct teacher per batch. Since `inputs` are raw dicts at this point, we can read `channel` from them:

```python
elif self.seq_kd:
    data_source = DataSource.TEACHER
    if self.template.truncation_strategy == 'raise':
        inputs = self.resample_encode_failed_inputs(inputs)

    # Determine which teacher to use for generation
    # (use majority channel in batch for seq_kd generation)
    channels = [inp.get('channel') for inp in inputs]
    teacher_indices = self._get_teacher_indices(channels)
    if teacher_indices and len(set(teacher_indices)) == 1:
        teacher_idx = teacher_indices[0]
    else:
        teacher_idx = 0  # Default for mixed batch or no routing

    encoded_inputs = self._prepare_batch_inputs(inputs, encode_prompt_only=True)
    teacher_model = self.teacher_models[teacher_idx]
    load_context = (self._load_single_teacher_context(teacher_idx)
                   if self.args.offload_teacher_model else nullcontext())
    with load_context, unwrap_model_for_generation(
            teacher_model, self.accelerator,
            gather_deepspeed3_params=self.teacher_ds3_gather_for_generation) as unwrapped_model:
        unwrapped_model.eval()
        new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
            unwrapped_model, encoded_inputs, self.generation_config,
            self.processing_class.pad_token_id)
    encoded_inputs['input_ids'] = new_input_ids
    encoded_inputs['attention_mask'] = new_attention_mask
    encoded_inputs['labels'] = new_labels
```

---

## 6. Per-Sample Routing Within a Batch

### Algorithm

Given a batch of 4 samples with channels `["math", "code", "math", "code"]` and resolved `channel_to_teacher_idx = {"math": 0, "code": 1}` (derived from `teacher_domain_map = {"math": "/path/to/math_teacher", "code": "/path/to/code_teacher"}`):

```
Step 1: Student forward on FULL batch
  student_logits = model(input_ids)  # [4, seq, vocab]

Step 2: Group samples by teacher
  teacher_0_samples = [0, 2]  (math)
  teacher_1_samples = [1, 3]  (code)

Step 3: Teacher 0 forward on subset [0, 2]
  subset_inputs = input_ids[[0, 2]]  # [2, seq]
  subset_logits_0 = teacher_0(subset_inputs)  # [2, seq, vocab]

Step 4: Teacher 1 forward on subset [1, 3]
  subset_inputs = input_ids[[1, 3]]  # [2, seq]
  subset_logits_1 = teacher_1(subset_inputs)  # [2, seq, vocab]

Step 5: Scatter into full batch tensor
  teacher_logits = zeros(4, seq, vocab)
  teacher_logits[[0, 2]] = subset_logits_0
  teacher_logits[[1, 3]] = subset_logits_1
  # Now teacher_logits[i] has logits from the CORRECT teacher for sample i

Step 6: Apply mask and compute JSD (UNCHANGED from single-teacher)
  mask = shifted_labels != -100
  shifted_student = student_logits[mask][None]    # [1, N_valid, vocab]
  shifted_teacher = teacher_logits[mask][None]    # [1, N_valid, vocab]
  loss = generalized_jsd_loss(shifted_student, shifted_teacher, beta=beta)
```

### Why This Works

The mask operation `logits[mask]` flattens valid tokens across the batch. Since each position `teacher_logits[i, j, :]` contains the correct teacher's output for sample `i`, the flattening preserves correctness — each valid token in the flattened tensor has its own teacher's distribution.

### Memory Profile

With `offload_teacher_model=True`:
```
GPU at peak: student model + 1 teacher model (same as single-teacher)
CPU: (N-1) other teacher models
```

Teachers are loaded one at a time. After each teacher's forward pass on its subset, it's offloaded before loading the next.

---

## 7. Example Multi-Teacher Training Script

### 7.1 Data Preparation

**`dataset_info_ms_swift.json`** (unchanged format):
```json
[
    {
        "dataset_name": "math_data",
        "dataset_path": "/mnt/bn/youxiang-lf/data/math/competition_math.jsonl",
        "columns": {"input": "prompt", "output": "response"}
    },
    {
        "dataset_name": "code_data",
        "dataset_path": "/mnt/bn/youxiang-lf/data/code/code_problems.jsonl",
        "columns": {"input": "prompt", "output": "response"}
    },
    {
        "dataset_name": "general_data",
        "dataset_path": "/mnt/bn/youxiang-lf/data/general/general_qa.jsonl",
        "columns": {"input": "prompt", "output": "response"}
    }
]
```

**Each JSONL file** — add `channel` field to each row:
```jsonl
{"prompt": "Solve x^2 + 3x - 4 = 0", "response": "The solutions are x=1 and x=-4.", "channel": "math"}
{"prompt": "Find the derivative of x^3", "response": "3x^2", "channel": "math"}
```

```jsonl
{"prompt": "Write a binary search in Python", "response": "def binary_search(arr, target): ...", "channel": "code"}
```

```jsonl
{"prompt": "Summarize the French Revolution", "response": "The French Revolution was...", "channel": "general"}
```

### 7.2 Training Script

```bash
#!/bin/bash

# Multi-node setup
nnodes=${ARNOLD_NUM:-1}
nproc_per_node=${ARNOLD_WORKER_GPU:-8}
export NNODES=$nnodes
export NPROC_PER_NODE=${nproc_per_node}
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export MASTER_ADDR=${MASTER_ADDR:=$ARNOLD_WORKER_0_HOST}
export MASTER_PORT=${MASTER_PORT:=$(echo "$ARNOLD_WORKER_0_PORT" | cut -d "," -f 1)}
export WANDB_PROJECT="ms-swift-multi-teacher-gkd"

# Models
STUDENT_MODEL="/mnt/bn/youxiang-lf/models/AwemeLM6-1.7B-v4.7.0-live-lm-v1/checkpoint-45200"
MATH_TEACHER="/mnt/bn/youxiang-lf/models/qwen3_8b_math_teacher/checkpoint-5936"
CODE_TEACHER="/mnt/bn/youxiang-lf/models/qwen3_8b_code_teacher/checkpoint-3000"
GENERAL_TEACHER="/mnt/bn/youxiang-lf/models/qwen3_8b_instruct-anchor-memory-teacher-40w/checkpoint-5936"

CUSTOM_DATASET_INFO="/mnt/bn/youxiang-lf/data/dataset_info_ms_swift.json"
DATASET="math_data code_data general_data"

# teacher_domain_map: maps channel names DIRECTLY to teacher model paths
# No positional index needed — the mapping is self-documenting
TEACHER_DOMAIN_MAP="{
  \"math\": \"$MATH_TEACHER\",
  \"code\": \"$CODE_TEACHER\",
  \"general\": \"$GENERAL_TEACHER\"
}"

run_name="Livelm-1.7B-multi-teacher-gkd-3teachers"
OUTPUT_DIR="/mnt/bn/youxiang-lf/models/$run_name"
export WANDB_NAME="$run_name"

mkdir -p $OUTPUT_DIR

NNODES=$nnodes \
NODE_RANK="${NODE_RANK:=$ARNOLD_ID}" \
MASTER_ADDR=${MASTER_ADDR} \
MASTER_PORT=${MASTER_PORT} \
NPROC_PER_NODE=$nproc_per_node \
swift rlhf \
    --rlhf_type gkd \
    --model $STUDENT_MODEL \
    --model_type qwen3 \
    --teacher_domain_map "$TEACHER_DOMAIN_MAP" \
    --train_type full \
    --custom_dataset_info $CUSTOM_DATASET_INFO \
    --dataset $DATASET \
    --split_dataset_ratio 0.01 \
    --seq_kd false \
    --lmbda 1 \
    --beta 0.9 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 4 \
    --eval_steps 495 \
    --save_steps 495 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --max_length 4096 \
    --max_completion_length 1024 \
    --output_dir $OUTPUT_DIR \
    --warmup_ratio 0.05 \
    --save_only_model true \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --deepspeed zero3 \
    --attn_impl sdpa \
    --teacher_deepspeed zero3 \
    --offload_teacher_model true \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --report_to wandb \
    --log_completions true \
    --seed 42 \
    --data_seed 42 \
    --run_name ${WANDB_NAME} \
    --sleep_level 1 | tee -a $OUTPUT_DIR/log.txt
```

**Key UX improvement**: No `--teacher_model` or `--teacher_model_type` needed! The `--teacher_domain_map` JSON directly maps each domain to its teacher path. Model types are auto-detected. If two domains share the same teacher path, the model is loaded only once.

### 7.3 Key Differences from Single-Teacher Script

| Parameter | Single-Teacher | Multi-Teacher |
|---|---|---|
| `--teacher_model` | One path | **Not used** |
| `--teacher_model_type` | One type (optional) | **Not used** (auto-detected) |
| `--teacher_domain_map` | Not used | JSON mapping channel → teacher **path** |
| `--offload_teacher_model` | Optional | Recommended (saves GPU memory) |
| Data format | No `channel` field needed | Each row needs `"channel": "domain_name"` |
| Everything else | Same | Same |

### 7.4 Backward Compatibility

A single-teacher invocation works identically to before:

```bash
swift rlhf \
    --rlhf_type gkd \
    --teacher_model /path/to/single/teacher \
    # ... all other args same as before
```

No `teacher_domain_map` + existing `teacher_model` → single-teacher mode. The model is auto-wrapped in a list internally. Zero behavior change.

### 7.5 Deduplication Example

If math and physics share the same STEM teacher:
```bash
TEACHER_DOMAIN_MAP='{
  "math": "/path/to/stem_teacher",
  "physics": "/path/to/stem_teacher",
  "code": "/path/to/code_teacher"
}'
```

Internally: 2 unique teachers loaded (not 3). Both "math" and "physics" channels route to the same model object. The deduplication is handled automatically during argument validation.

---

## 8. Challenges & Mitigations

### 8.1 GPU Memory with Multiple Teachers

**Challenge**: N teachers on GPU simultaneously.

**Mitigation**: `--offload_teacher_model true`. Teachers live on CPU. During `_multi_teacher_forward`, each teacher is loaded to GPU one at a time, runs its subset, then offloads. Peak GPU memory = student + 1 teacher (same as single-teacher).

### 8.2 Mixed-Domain Batches

**Challenge**: A batch may contain samples from different domains, requiring multiple teacher forward passes.

**Mitigation**: The `_multi_teacher_forward` method groups samples by teacher, runs each teacher on its subset, and scatters results back. This is mathematically equivalent to running each sample through its own teacher.

**Performance note**: If a batch has K distinct domains, K teacher forward passes are needed (each on a smaller subset). Sorting the dataset by domain before training would minimize this by creating homogeneous batches.

### 8.3 Vocab Size Mismatch Across Teachers

**Challenge**: Different teachers may have different vocab sizes.

**Mitigation**: The existing vocab alignment code at `gkd_trainer.py:243-251` handles this. It pads the smaller vocab and copies values from the larger. Since alignment happens AFTER teacher logits are gathered into the full batch tensor, it works correctly regardless of which teacher produced which sample's logits — all teachers of the same architecture will have the same vocab size.

**Note**: If different teachers have different vocab sizes (e.g., math teacher is Qwen with 150K vocab, code teacher is CodeLlama with 32K vocab), the per-sample scatter approach needs per-teacher vocab alignment. This is an edge case — in practice, domain teachers are usually fine-tuned from the same base model and share vocab size.

### 8.4 Liger Kernel with Multi-Teacher

**Challenge**: Liger fused JSD operates on hidden states and is monolithic — can't split by teacher.

**Mitigation**: For the liger path, use majority-teacher routing (the most common teacher in the batch). This is approximate but avoids liger kernel modifications. A warning is logged when liger is used with multi-teacher and mixed batches.

### 8.5 `seq_kd` Mode with Multi-Teacher

**Challenge**: In `seq_kd` mode, the teacher generates responses. With multiple teachers, which one generates?

**Mitigation**: Use the majority channel in the batch to select the generation teacher. Since `seq_kd` generates responses that are then used for the full batch, using a single teacher for generation is reasonable. The soft label computation in `compute_loss` still routes per-sample correctly.

### 8.6 vLLM Colocate/Server Mode

**No challenge**: vLLM is used for **student** on-policy generation, not teacher generation. Multi-teacher routing only affects `compute_loss` (soft label computation). The vLLM rollout infrastructure is completely unaffected.
