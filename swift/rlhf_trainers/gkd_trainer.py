# Copyright (c) ModelScope Contributors. All rights reserved.
import inspect
import os
import random
from collections import defaultdict, deque
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from enum import Enum
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import trl
from accelerate.utils import gather_object, is_peft_model
from packaging import version
from transformers import PreTrainedModel
from trl import GKDTrainer as HFGKDTrainer
from trl import SFTTrainer as HFSFTTrainer

from swift.template import TemplateInputs
from swift.trainers import SwiftMixin, disable_gradient_checkpointing
from swift.utils import (JsonlWriter, get_logger, is_swanlab_available, is_wandb_available, remove_response, to_device,
                         unwrap_model_for_generation)
from .rollout_mixin import DataType, RolloutTrainerMixin
from .utils import (get_gather_if_zero3_context, identity_data_collator, patch_profiling_context,
                    patch_profiling_decorator, prepare_deepspeed)

try:
    from liger_kernel.chunked_loss import LigerFusedLinearJSDLoss
    _liger_kernel_available = True
except ImportError:
    _liger_kernel_available = False

del HFGKDTrainer.__init__
del HFSFTTrainer.__init__

logger = get_logger()
if is_wandb_available():
    import wandb
if is_swanlab_available():
    import swanlab


class DataSource(str, Enum):
    STUDENT = 'student'  # On-policy: student model generates responses
    TEACHER = 'teacher'  # Sequential KD: teacher model generates responses
    DATASET = 'dataset'  # Off-policy: use dataset responses


class GKDTrainer(RolloutTrainerMixin, SwiftMixin, HFGKDTrainer):

    def __init__(self, model: Optional[Union[PreTrainedModel, nn.Module, str]] = None, *_args, **kwargs):
        teacher_models_input = kwargs.pop('teacher_model')  # Always a list from pipeline
        teacher_deepspeed_config = kwargs.pop('teacher_deepspeed_config', None)
        self.channel_to_teacher_idx = kwargs.pop('channel_to_teacher_idx', None)
        self.vllm_client = kwargs.pop('vllm_client', None)
        super().__init__(model, None, *_args, **kwargs)
        args = kwargs['args']
        self.lmbda = args.lmbda
        self.temperature = args.temperature
        self.seq_kd = args.seq_kd
        # Per-teacher hyperparameter maps (None means use global values)
        self.channel_to_beta = getattr(args, '_channel_to_beta', None)
        self.channel_to_temperature = getattr(args, '_channel_to_temperature', None)
        self.generation_config = model.generation_config
        self._metrics = {'train': defaultdict(list), 'eval': defaultdict(list)}
        self._total_train_tokens = 0
        # Multi-teacher observability and loss balance
        self.log_domain_routing = getattr(args, 'log_domain_routing', True)
        self.enable_weighted_domain_loss = getattr(args, 'enable_weighted_domain_loss', True)
        # Accumulated per-domain losses for WandB logging (flushed in log())
        self._domain_loss_accum: Dict[str, list] = defaultdict(list)

        # Initialize logging components
        self._prepare_logging()

        # Initialize liger loss
        self._prepare_liger_loss()

        self.teacher_ds3_gather_for_generation = args.ds3_gather_for_generation

        # Initialize teacher model(s) — supports both single and multi-teacher
        if not isinstance(teacher_models_input, list):
            teacher_models_input = [teacher_models_input]

        self.teacher_models = []
        self.is_teacher_ds3_list = []

        for tm in teacher_models_input:
            if self.is_deepspeed_enabled:
                if teacher_deepspeed_config is not None:
                    is_ds3 = teacher_deepspeed_config.get('zero_optimization', {}).get('stage') == 3
                    self.is_teacher_ds3_list.append(is_ds3)
                    if not is_ds3:
                        self.teacher_ds3_gather_for_generation = False
                    prepared = prepare_deepspeed(
                        tm, self.accelerator, deepspeed_config=teacher_deepspeed_config, training_args=args)
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

        # Backward compat aliases for single-teacher code paths
        self.teacher_model = self.teacher_models[0]
        self.is_teacher_ds3 = self.is_teacher_ds3_list[0] if self.is_teacher_ds3_list else None

        if len(self.teacher_models) > 1:
            logger.info(f'Multi-teacher GKD: initialized {len(self.teacher_models)} teacher(s)')
            if self.channel_to_teacher_idx:
                logger.info(f'  Channel->teacher routing: {self.channel_to_teacher_idx}')
            if not self.args.offload_teacher_model:
                logger.warning(
                    f'Multi-teacher GKD: {len(self.teacher_models)} teachers loaded to GPU simultaneously. '
                    f'Consider --offload_teacher_model true to reduce peak GPU memory.')

        if self.args.offload_teacher_model:
            for tm in self.teacher_models:
                self.offload_model(self.accelerator.unwrap_model(tm))

        # Initialize rollout infrastructure for vLLM support
        self.prepare_rollout()

        # Initialize activation offloading context
        args.activation_offloading = False  # TODO: remove
        if args.activation_offloading:
            from trl.models import get_act_offloading_ctx_manager
            self.maybe_activation_offload_context = get_act_offloading_ctx_manager(model=self.model)
        else:
            self.maybe_activation_offload_context = nullcontext()
        self._trl_version_gte_0_24 = version.parse(trl.__version__) >= version.parse('0.24')

        # Initialize resample data iterator for truncation_strategy 'raise'('delete')
        if self.template.truncation_strategy == 'raise':
            self._prepare_resample_data_iterator()

    def _get_data_collator(self, args, template):
        return identity_data_collator

    # Multi-teacher routing helpers

    def _get_teacher_indices(self, channels):
        """Map channel values to teacher indices for per-sample routing.

        Args:
            channels: list of channel strings (one per sample in batch), or None

        Returns:
            list of int teacher indices, or None if single-teacher or no routing needed
        """
        if channels is None or self.channel_to_teacher_idx is None or len(self.teacher_models) <= 1:
            return None

        indices = []
        for ch in channels:
            if ch is not None and ch in self.channel_to_teacher_idx:
                indices.append(self.channel_to_teacher_idx[ch])
            else:
                if ch is not None:
                    logger.warning_once(
                        f'Channel "{ch}" not found in teacher_domain_map. '
                        f'Defaulting to teacher[0]. Valid channels: {list(self.channel_to_teacher_idx.keys())}')
                indices.append(0)  # Default to first teacher
        return indices

    def _multi_teacher_forward(self, model_inputs, teacher_indices):
        """Run per-sample teacher routing: groups by teacher, runs subset forwards, scatters back.

        Args:
            model_inputs: dict with tensors of shape [batch, seq, ...]
            teacher_indices: list[int] of length batch_size

        Returns:
            teacher_logits: tensor [batch, seq, vocab] with per-sample teacher logits

        Note on vocab mismatch handling:
            When teachers have different vocab sizes, smaller teachers' logits are padded
            with -1e9 (softmax ~ 0) for missing token positions. This interacts with the
            student-teacher vocab padding in compute_loss() (which copies logits from the
            larger model). For best results, use teachers from the same model family with
            identical vocab sizes.
        """
        batch_size = model_inputs['input_ids'].shape[0]
        device = model_inputs['input_ids'].device
        teacher_logits = None
        max_vocab_size = 0

        # Group samples by teacher
        teacher_to_samples = defaultdict(list)
        for sample_idx, teacher_idx in enumerate(teacher_indices):
            teacher_to_samples[teacher_idx].append(sample_idx)

        observed_vocab_sizes = set()
        for teacher_idx, sample_idxs in teacher_to_samples.items():
            sample_idxs_tensor = torch.tensor(sample_idxs, device=device)

            # Extract subset of batch for this teacher
            subset_inputs = {}
            for k, v in model_inputs.items():
                if isinstance(v, torch.Tensor) and v.shape[0] == batch_size:
                    subset_inputs[k] = v[sample_idxs_tensor]
                else:
                    subset_inputs[k] = v

            teacher_model = self.teacher_models[teacher_idx]
            is_ds3 = self.is_teacher_ds3_list[teacher_idx] if teacher_idx < len(self.is_teacher_ds3_list) else None
            load_ctx = self._load_single_teacher_context(teacher_idx) if self.args.offload_teacher_model else nullcontext()
            with torch.no_grad(), load_ctx, disable_gradient_checkpointing(
                    teacher_model, self.args.gradient_checkpointing_kwargs):
                subset_outputs = teacher_model(**subset_inputs)

            subset_logits = subset_outputs.logits
            vocab_size = subset_logits.shape[-1]
            observed_vocab_sizes.add(vocab_size)

            # Initialize or expand output tensor
            if teacher_logits is None:
                seq_len = subset_logits.shape[1]
                max_vocab_size = vocab_size
                teacher_logits = torch.zeros(
                    batch_size, seq_len, vocab_size,
                    dtype=subset_logits.dtype, device=device)
            elif vocab_size > max_vocab_size:
                # Expand to accommodate larger vocab; use -1e9 so softmax gives ~0 probability
                # for tokens the smaller teacher doesn't have (avoids spurious probability mass)
                teacher_logits = F.pad(teacher_logits, (0, vocab_size - max_vocab_size), 'constant', -1e9)
                max_vocab_size = vocab_size
            elif vocab_size < max_vocab_size:
                # Pad this teacher's logits to match; -1e9 for tokens it doesn't have
                subset_logits = F.pad(subset_logits, (0, max_vocab_size - vocab_size), 'constant', -1e9)

            # Scatter subset logits back into full batch
            teacher_logits[sample_idxs_tensor] = subset_logits

        if len(observed_vocab_sizes) > 1:
            logger.warning_once(
                f'Multi-teacher vocab size mismatch detected: {observed_vocab_sizes}. '
                f'Smaller teachers are padded with -1e9 logits. For best results, '
                f'use teachers from the same model family with identical vocab sizes.')

        return teacher_logits

    @contextmanager
    def _load_single_teacher_context(self, teacher_idx):
        """Load/offload a specific teacher by index."""
        if not self.args.offload_teacher_model:
            yield
            return
        teacher = self.accelerator.unwrap_model(self.teacher_models[teacher_idx])
        self.load_model(teacher)
        try:
            yield
        finally:
            self.offload_model(teacher)

    def _seq_kd_per_teacher_generate(self, inputs, channels, teacher_indices):
        """Generate seq_kd responses per-teacher so each sample gets text from its assigned teacher.

        Args:
            inputs: raw list of input dicts
            channels: list of channel strings (one per sample)
            teacher_indices: list of int teacher indices (one per sample)

        Returns:
            encoded_inputs dict with input_ids, attention_mask, labels, and channel
        """
        pad_token_id = self.processing_class.pad_token_id

        # Group samples by teacher
        teacher_to_samples = defaultdict(list)
        for i, tidx in enumerate(teacher_indices):
            teacher_to_samples[tidx].append(i)

        # Per-sample results: (input_ids_1d, attention_mask_1d, labels_1d)
        results = {}
        max_seq_len = 0

        for tidx, sample_idxs in teacher_to_samples.items():
            subset_inputs = [inputs[i] for i in sample_idxs]
            subset_encoded = self._prepare_batch_inputs(subset_inputs, encode_prompt_only=True)

            load_context = (self._load_single_teacher_context(tidx)
                            if self.args.offload_teacher_model else nullcontext())
            with load_context, unwrap_model_for_generation(
                    self.teacher_models[tidx],
                    self.accelerator,
                    gather_deepspeed3_params=self.teacher_ds3_gather_for_generation) as unwrapped_model:
                unwrapped_model.eval()
                new_ids, new_mask, new_labels = self.generate_on_policy_outputs(
                    unwrapped_model, subset_encoded, self.generation_config, pad_token_id)

            max_seq_len = max(max_seq_len, new_ids.shape[1])
            for local_i, global_i in enumerate(sample_idxs):
                results[global_i] = (new_ids[local_i], new_mask[local_i], new_labels[local_i])

        # Pad all samples to max_seq_len and stack in original order
        batch_ids, batch_mask, batch_labels = [], [], []
        for i in range(len(inputs)):
            ids, mask, labels = results[i]
            pad_len = max_seq_len - ids.shape[0]
            if pad_len > 0:
                ids = F.pad(ids, (0, pad_len), value=pad_token_id)
                mask = F.pad(mask, (0, pad_len), value=0)
                labels = F.pad(labels, (0, pad_len), value=-100)
            batch_ids.append(ids)
            batch_mask.append(mask)
            batch_labels.append(labels)

        encoded_inputs = {
            'input_ids': torch.stack(batch_ids),
            'attention_mask': torch.stack(batch_mask),
            'labels': torch.stack(batch_labels),
        }
        # Preserve channels for routing in compute_loss
        if any(ch is not None for ch in channels):
            encoded_inputs['channel'] = channels

        # Compute position_ids from attention_mask
        position_ids = encoded_inputs['attention_mask'].cumsum(dim=1) - 1
        position_ids[position_ids < 0] = 0
        encoded_inputs['position_ids'] = position_ids

        return encoded_inputs

    # Code borrowed from huggingface/trl
    def generate_on_policy_outputs(self, model, inputs, generation_config, pad_token_id=None):
        """Generate on-policy outputs using the model.

        When encode_prompt_only=True, inputs['input_ids'] already contains only the prompt part.
        """
        assert not self.template.padding_free, 'generate not support padding_free/packing.'
        prompt_input_ids = inputs['input_ids']
        model_inputs = {k: v for k, v in inputs.items() if k != 'labels'}
        model_inputs.pop('position_ids', None)
        model_inputs.pop('text_position_ids', None)
        model_inputs.pop('channel', None)  # metadata field, not a model input
        kwargs = {}
        base_model = self.template.get_base_model(model)
        parameters = inspect.signature(base_model.generate).parameters
        if 'use_model_defaults' in parameters:
            kwargs['use_model_defaults'] = False
        with self.template.generate_context():
            if self.model.model_meta.is_multimodal:
                _, model_inputs = self.template.pre_forward_hook(model, None, model_inputs)
            generated_outputs = model.generate(
                **model_inputs, generation_config=generation_config, return_dict_in_generate=True, **kwargs)
        # Get the generated token IDs
        generated_tokens = generated_outputs.sequences
        if not self.template.skip_prompt:
            generated_tokens = torch.concat([prompt_input_ids, generated_tokens], dim=1)
        # Calculate new attention mask
        new_attention_mask = torch.ones_like(generated_tokens)
        new_labels = generated_tokens.clone()
        new_labels[:, :prompt_input_ids.shape[1]] = -100

        # If there's pad_token_id, set attention mask to 0 for padding tokens
        if pad_token_id is not None:
            new_labels[new_labels == pad_token_id] = -100
            new_attention_mask[generated_tokens == pad_token_id] = 0

        new_position_ids = new_attention_mask.cumsum(dim=1) - 1
        new_position_ids[new_position_ids < 0] = 0
        inputs['position_ids'] = new_position_ids
        return generated_tokens, new_attention_mask, new_labels

    def _compute_grouped_jsd_loss(self, flat_student_logits, flat_teacher_logits, channels, mask):
        """Compute JSD loss with per-teacher beta and temperature.

        Groups flattened masked tokens by their (beta, temperature) pair, computes JSD loss
        per group, and returns the weighted average (weighted by token count).

        Args:
            flat_student_logits: [1, num_valid_tokens, vocab] already masked/flattened student logits
            flat_teacher_logits: [1, num_valid_tokens, vocab] already masked/flattened teacher logits
            channels: list of channel names per sample (length = batch_size)
            mask: [batch, seq_len] boolean mask (used to map flat token index -> sample index)
        """
        global_beta = self.beta
        global_temp = self.temperature

        # Build per-token (beta, temp) by expanding sample-level channels using mask token counts
        token_params = []
        for i in range(mask.shape[0]):
            n_tokens = mask[i].sum().item()
            ch = channels[i] if channels and i < len(channels) else None
            beta = self.channel_to_beta.get(ch, global_beta) if self.channel_to_beta and ch is not None else global_beta
            temp = (self.channel_to_temperature.get(ch, global_temp)
                    if self.channel_to_temperature and ch is not None else global_temp)
            token_params.extend([(beta, temp)] * n_tokens)

        # Group flat token indices by (beta, temp)
        groups = defaultdict(list)
        for tok_idx, (beta, temp) in enumerate(token_params):
            groups[(beta, temp)].append(tok_idx)

        # Squeeze batch dim: [1, N, V] -> [N, V]
        student_logits_2d = flat_student_logits.squeeze(0)
        teacher_logits_2d = flat_teacher_logits.squeeze(0)

        total_loss = student_logits_2d.new_zeros(())
        total_tokens = 0

        for (beta, temp), indices in groups.items():
            idx_tensor = torch.tensor(indices, device=student_logits_2d.device, dtype=torch.long)
            group_student = student_logits_2d[idx_tensor]  # [num_tokens, vocab]
            group_teacher = teacher_logits_2d[idx_tensor]  # [num_tokens, vocab]

            group_loss = self.generalized_jsd_loss(
                student_logits=group_student.unsqueeze(0),
                teacher_logits=group_teacher.unsqueeze(0),
                beta=beta,
                temperature=temp,
            )
            num_tokens = len(indices)
            total_loss = total_loss + group_loss * num_tokens
            total_tokens += num_tokens

        if total_tokens == 0:
            return total_loss
        return total_loss / total_tokens

    def _compute_domain_weighted_jsd_loss(self, flat_student_logits, flat_teacher_logits, channels, mask):
        """Compute JSD loss with equal per-domain weighting to prevent high-KL domains from dominating.

        Groups flattened masked tokens by channel name, computes token-averaged JSD loss per domain,
        then returns the mean of per-domain losses (equal domain weight regardless of token count).
        Also populates self._domain_loss_accum for per-domain WandB logging.

        Uses per-teacher beta/temperature from channel_to_beta/channel_to_temperature if set.

        Args:
            flat_student_logits: [1, num_valid_tokens, vocab] masked/flattened student logits
            flat_teacher_logits: [1, num_valid_tokens, vocab] masked/flattened teacher logits
            channels: list of channel names per sample (length = batch_size)
            mask: [batch, seq_len] boolean mask (maps sample index -> token count)
        """
        global_beta = self.beta
        global_temp = self.temperature

        # Build per-token channel labels by expanding sample-level channels using mask token counts
        token_channels = []
        for i in range(mask.shape[0]):
            n_tokens = mask[i].sum().item()
            ch = channels[i] if channels and i < len(channels) else None
            token_channels.extend([ch] * int(n_tokens))

        # Group flat token indices by channel name
        channel_to_indices = defaultdict(list)
        for tok_idx, ch in enumerate(token_channels):
            channel_to_indices[ch].append(tok_idx)

        unique_channels = list(channel_to_indices.keys())

        # Single channel in batch: fall back to standard loss (no domain weighting needed)
        if len(unique_channels) <= 1:
            ch = unique_channels[0] if unique_channels else None
            beta = self.channel_to_beta.get(ch, global_beta) if self.channel_to_beta and ch is not None else global_beta
            temp = self.channel_to_temperature.get(ch, global_temp) if self.channel_to_temperature and ch is not None else global_temp
            loss = self.generalized_jsd_loss(
                student_logits=flat_student_logits,
                teacher_logits=flat_teacher_logits,
                beta=beta,
                temperature=temp,
            )
            if ch is not None:
                self._domain_loss_accum[ch].append(loss.item())
            return loss

        # Squeeze batch dim: [1, N, V] -> [N, V]
        student_logits_2d = flat_student_logits.squeeze(0)
        teacher_logits_2d = flat_teacher_logits.squeeze(0)

        per_domain_losses = []
        for ch, indices in channel_to_indices.items():
            idx_tensor = torch.tensor(indices, device=student_logits_2d.device, dtype=torch.long)
            group_student = student_logits_2d[idx_tensor]
            group_teacher = teacher_logits_2d[idx_tensor]
            beta = self.channel_to_beta.get(ch, global_beta) if self.channel_to_beta and ch is not None else global_beta
            temp = self.channel_to_temperature.get(ch, global_temp) if self.channel_to_temperature and ch is not None else global_temp
            domain_loss = self.generalized_jsd_loss(
                student_logits=group_student.unsqueeze(0),
                teacher_logits=group_teacher.unsqueeze(0),
                beta=beta,
                temperature=temp,
            )
            per_domain_losses.append(domain_loss)
            if ch is not None:
                self._domain_loss_accum[ch].append(domain_loss.item())

        # Equal-weight average across domains
        return sum(per_domain_losses) / len(per_domain_losses)

    @patch_profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Get data source: DataSource.STUDENT, DataSource.TEACHER, or DataSource.DATASET
        data_source = inputs.pop('_data_source', DataSource.DATASET)
        # Pop channel for multi-teacher routing (also prevents leaking to model forward)
        channels = inputs.pop('channel', None)
        model_inputs = {k: v for k, v in inputs.items() if k not in {'prompt', 'labels'}}
        # If generate is used, then use_logits_to_keep must be set to False.
        use_logits_to_keep = self.get_use_logits_to_keep(True)
        if use_logits_to_keep and not self.use_liger_gkd_loss:
            self.prepare_logits_to_keep(inputs)
            model_inputs['logits_to_keep'] = inputs['logits_to_keep']

        # Determine per-sample teacher routing
        teacher_indices = self._get_teacher_indices(channels)
        use_multi_teacher = teacher_indices is not None and len(set(teacher_indices)) > 1

        # Log per-batch routing distribution to stdout (rank 0, gated by log_domain_routing).
        # Each entry shows: channel->teacher_idx=count (unknown channels show ->0 to clarify fallback).
        if self.log_domain_routing and channels and self.accelerator.is_main_process:
            from collections import Counter
            routing_counts = Counter(channels)
            # Use explicit sort key: None sorts last, all other keys coerced to str for safe comparison
            def _teacher_label(ch):
                if self.channel_to_teacher_idx and ch is not None and ch in self.channel_to_teacher_idx:
                    return f'{ch}->[{self.channel_to_teacher_idx[ch]}]'
                elif ch is None:
                    return 'None->[0]'
                else:
                    return f'{ch}->[0]?'  # unknown channel falls back to teacher[0]

            routing_str = ', '.join(
                f'{_teacher_label(ch)}={cnt}' for ch, cnt in
                sorted(routing_counts.items(), key=lambda x: (x[0] is None, str(x[0]) if x[0] is not None else '')))
            print(f'[Step {self.state.global_step}] Routing: {routing_str}', flush=True)

        has_per_teacher_params = (self.channel_to_beta or self.channel_to_temperature) and channels
        if self.use_liger_gkd_loss and not use_multi_teacher and not has_per_teacher_params:
            # Liger fused JSD loss for memory efficiency (single teacher only)
            teacher_idx = 0 if not teacher_indices else teacher_indices[0]
            teacher_model_selected = self.teacher_models[teacher_idx]
            is_ds3_selected = self.is_teacher_ds3_list[teacher_idx] if teacher_idx < len(
                self.is_teacher_ds3_list) else None

            # Get base models (exclude lm_head to save memory)
            unwrapped_student = self.accelerator.unwrap_model(model)
            if is_peft_model(unwrapped_student):
                unwrapped_student = unwrapped_student.base_model.model
            base_student = getattr(unwrapped_student, getattr(unwrapped_student, 'base_model_prefix', 'model'),
                                   unwrapped_student)

            unwrapped_teacher = self.accelerator.unwrap_model(teacher_model_selected)
            base_teacher = getattr(unwrapped_teacher, getattr(unwrapped_teacher, 'base_model_prefix', 'model'),
                                   unwrapped_teacher)

            # Forward through base models
            student_outputs = base_student(**model_inputs, use_cache=False)

            load_context = (self._load_single_teacher_context(teacher_idx)
                            if self.args.offload_teacher_model else nullcontext())
            with load_context:
                with torch.no_grad(), disable_gradient_checkpointing(teacher_model_selected,
                                                                     self.args.gradient_checkpointing_kwargs):
                    teacher_outputs = base_teacher(**model_inputs, use_cache=False)

                # Get hidden states (shifted)
                student_hidden = student_outputs.last_hidden_state[:, :-1]
                teacher_hidden = teacher_outputs.last_hidden_state[:, :-1]

                # Release full outputs to free memory
                del student_outputs, teacher_outputs

                # Prepare labels (shifted)
                labels_mask = inputs['labels'] != -100
                masked_input_ids = torch.where(labels_mask, inputs['input_ids'],
                                               torch.full_like(inputs['input_ids'], -100))
                true_labels = masked_input_ids[:, 1:].contiguous()

                # Release intermediate tensors
                del labels_mask, masked_input_ids

                # Get output heads
                student_head = unwrapped_student.get_output_embeddings()
                teacher_head = unwrapped_teacher.get_output_embeddings()

                # Prepare context managers for gathering parameters in zero3
                teacher_context = get_gather_if_zero3_context(self, is_zero3=is_ds3_selected)(teacher_head.weight)
                student_context = get_gather_if_zero3_context(self)(student_head.weight)

                with teacher_context, student_context:
                    # Compute liger fused JSD loss
                    loss = self.liger_jsd_loss(
                        student_input=student_hidden,
                        student_weight=student_head.weight,
                        teacher_input=teacher_hidden,
                        teacher_weight=teacher_head.weight,
                        true_labels=true_labels,
                        student_bias=getattr(student_head, 'bias', None),
                        teacher_bias=getattr(teacher_head, 'bias', None),
                    )
                    # loss / grad norm is unexpectedly large, normalize by sequence length
                    # https://github.com/linkedin/Liger-Kernel/blob/v0.6.3/src/liger_kernel/chunked_loss/jsd_loss.py#L9-L39
                    loss /= student_hidden.shape[1]
                # Release hidden states after loss computation
                del student_hidden, teacher_hidden, true_labels
        else:
            if self.use_liger_gkd_loss and use_multi_teacher:
                logger.warning_once(
                    'Liger GKD loss does not support mixed-teacher batches. Falling back to standard JSD, '
                    'which uses more GPU memory. To avoid this: (1) set --use_liger_kernel false, or '
                    '(2) sort dataset by channel so each batch uses a single teacher.')
            elif self.use_liger_gkd_loss and has_per_teacher_params:
                logger.warning_once(
                    'Liger GKD loss does not support per-teacher beta/temperature. '
                    'Falling back to standard JSD with grouped loss.')

            # Standard loss computation
            if self.args.sft_alpha > 0:
                model_inputs['labels'] = inputs['labels']
            # compute student output
            outputs_student = model(**model_inputs)

            model_inputs.pop('labels', None)

            if not use_multi_teacher:
                # Single teacher for all samples (original behavior)
                teacher_idx = teacher_indices[0] if teacher_indices else 0
                teacher_model_selected = self.teacher_models[teacher_idx]
                load_context = (self._load_single_teacher_context(teacher_idx)
                                if self.args.offload_teacher_model else nullcontext())
                with torch.no_grad(), load_context, disable_gradient_checkpointing(
                        teacher_model_selected, self.args.gradient_checkpointing_kwargs):
                    outputs_teacher = teacher_model_selected(**model_inputs)
                teacher_logits = outputs_teacher.logits
            else:
                # Per-sample routing: multiple teachers in one batch
                teacher_logits = self._multi_teacher_forward(model_inputs, teacher_indices)

            shifted_labels = torch.roll(inputs['labels'], shifts=-1, dims=1)
            mask = shifted_labels != -100
            shifted_student_logits = outputs_student.logits[mask][None]
            shifted_teacher_logits = teacher_logits[mask][None]

            # Fix the vocab_size mismatch between Qwen2.5-VL-3B-Instruct and Qwen2.5-VL-7B-Instruct.
            stu_dim = shifted_student_logits.shape[-1]
            tea_dim = shifted_teacher_logits.shape[-1]
            if stu_dim < tea_dim:
                shifted_student_logits = F.pad(shifted_student_logits, (0, tea_dim - stu_dim), 'constant', 0)
                shifted_student_logits[..., stu_dim:] = shifted_teacher_logits[..., stu_dim:]
            elif stu_dim > tea_dim:
                shifted_teacher_logits = F.pad(shifted_teacher_logits, (0, stu_dim - tea_dim), 'constant', 0)
                shifted_teacher_logits[..., tea_dim:] = shifted_student_logits[..., tea_dim:]

            # compute loss
            if self.enable_weighted_domain_loss and use_multi_teacher:
                # New: equal per-domain weighting (also handles per-teacher beta/temp)
                loss = self._compute_domain_weighted_jsd_loss(
                    shifted_student_logits, shifted_teacher_logits, channels, mask)
            elif (self.channel_to_beta or self.channel_to_temperature) and channels:
                # Per-teacher hyperparameters only: group tokens by (beta, temp), token-weighted
                loss = self._compute_grouped_jsd_loss(
                    shifted_student_logits, shifted_teacher_logits, channels, mask)
            else:
                loss = self.generalized_jsd_loss(
                    student_logits=shifted_student_logits,
                    teacher_logits=shifted_teacher_logits,
                    beta=self.beta,
                    temperature=self.temperature,
                )
            # Add SFT loss if enabled (skip for student-generated responses)
            if self.args.sft_alpha > 0 and data_source != DataSource.STUDENT:
                loss = loss + self.args.sft_alpha * outputs_student.loss

        # Return loss
        if return_outputs:
            if self.use_liger_gkd_loss:
                # outputs has been released in liger loss computation to reduce peak memory
                outputs_student = None
            return (loss, outputs_student)
        else:
            return loss

    def _prepare_batch_inputs(self, inputs: list, encode_prompt_only: bool = False) -> Dict[str, torch.Tensor]:
        """Prepare batch inputs for training.

        Args:
            inputs: List of input data dictionaries
            encode_prompt_only: If True, only encode the prompt part (for on-policy/seq_kd generation).
                               If False, encode the full messages including response (for offline dataset).
        """
        from .utils import replace_assistant_response_with_ids

        template = self.template
        batch_encoded_inputs = []

        # Use 'transformers' mode for prompt-only encoding, 'train' mode for full encoding
        mode = 'transformers' if encode_prompt_only else 'train'
        with self._template_context(template, mode=mode):
            for data in inputs:
                if 'response_token_ids' in data and data['response_token_ids']:
                    data['messages'] = replace_assistant_response_with_ids(data['messages'], data['response_token_ids'])

                if encode_prompt_only:
                    # Remove response content for prompt-only encoding.
                    # Create a shallow copy to avoid mutating the original data dict
                    # (which would corrupt responses for multi-epoch training).
                    messages = data.get('messages', [])
                    if messages and messages[-1].get('role') == 'assistant':
                        data = {**data, 'messages': [*messages[:-1], {**messages[-1], 'content': None}]}

                encoded = template.encode(data, return_length=True)
                batch_encoded_inputs.append(encoded)

            batch_encoded = to_device(template.data_collator(batch_encoded_inputs), self.model.device)

        return batch_encoded

    # Code borrowed from huggingface/trl
    @patch_profiling_decorator
    def training_step(self,
                      model: nn.Module,
                      inputs: DataType,
                      num_items_in_batch: Optional[int] = None) -> torch.Tensor:
        """
        Perform a training step for the Generalized Knowledge Distillation (GKD) model.

        This method implements the on-policy learning approach described in the GKD paper.
        With probability `self.lmbda`, it generates new responses using the student model,
        which are then used for training instead of the original inputs.

        When use_vllm is enabled, vLLM engine is used for faster generation.
        """
        args = self.args
        with patch_profiling_context(self, 'get_completions'):
            if self._get_random_num() <= self.lmbda:
                # On-policy: student model generates responses
                data_source = DataSource.STUDENT
                # Resample inputs that fail encoding when truncation_strategy is 'raise'('delete')
                if self.template.truncation_strategy == 'raise':
                    inputs = self.resample_encode_failed_inputs(inputs)
                if args.use_vllm:
                    processed_inputs = self._preprocess_inputs(inputs)
                    # Preserve channels before vLLM inference — vLLM drops metadata fields
                    original_channels = [inp.get('channel') for inp in inputs]
                    generated_inputs = self._fast_infer(processed_inputs)
                    # Reattach channels for multi-teacher routing in compute_loss
                    for gen_inp, ch in zip(generated_inputs, original_channels):
                        if ch is not None:
                            gen_inp['channel'] = ch
                    if self.log_completions:
                        messages = [inp['messages'][:-1] for inp in generated_inputs]
                        completions = [deepcopy(inp['messages'][-1]['content']) for inp in generated_inputs]
                        valid_messages = gather_object(messages)
                        valid_completions = gather_object(completions)
                        self._logs['prompt'].extend(self._apply_chat_template_to_messages_list(valid_messages))
                        self._logs['completion'].extend(valid_completions)
                    with self._template_context(self.template):
                        # vLLM already generated response, encode full messages
                        encoded_inputs = self._prepare_batch_inputs(generated_inputs, encode_prompt_only=False)
                else:
                    # Need prompt-only encoding for on-policy generation
                    encoded_inputs = self._prepare_batch_inputs(inputs, encode_prompt_only=True)
                    with unwrap_model_for_generation(
                            model, self.accelerator,
                            gather_deepspeed3_params=args.ds3_gather_for_generation) as unwrapped_model:
                        unwrapped_model.eval()
                        new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                            unwrapped_model, encoded_inputs, self.generation_config, self.processing_class.pad_token_id)
                        unwrapped_model.train()
                    # override with generated inputs
                    encoded_inputs['input_ids'] = new_input_ids
                    encoded_inputs['attention_mask'] = new_attention_mask
                    encoded_inputs['labels'] = new_labels

            elif self.seq_kd:
                # Sequential KD: teacher model generates responses
                data_source = DataSource.TEACHER

                # Resample inputs that fail encoding when truncation_strategy is 'raise'('delete')
                if self.template.truncation_strategy == 'raise':
                    inputs = self.resample_encode_failed_inputs(inputs)

                channels = [inp.get('channel') for inp in inputs] if isinstance(inputs, list) else None
                teacher_indices = self._get_teacher_indices(channels)

                if teacher_indices is not None and len(set(teacher_indices)) > 1:
                    # Multi-teacher seq_kd: generate per-teacher so each sample gets
                    # text from its assigned teacher (not just the majority teacher)
                    encoded_inputs = self._seq_kd_per_teacher_generate(inputs, channels, teacher_indices)
                else:
                    # Single teacher (original path)
                    teacher_idx = teacher_indices[0] if teacher_indices else 0
                    encoded_inputs = self._prepare_batch_inputs(inputs, encode_prompt_only=True)
                    load_context = (self._load_single_teacher_context(teacher_idx)
                                    if self.args.offload_teacher_model else nullcontext())
                    with load_context, unwrap_model_for_generation(
                            self.teacher_models[teacher_idx],
                            self.accelerator,
                            gather_deepspeed3_params=self.teacher_ds3_gather_for_generation) as unwrapped_model:
                        unwrapped_model.eval()
                        new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                            unwrapped_model, encoded_inputs, self.generation_config,
                            self.processing_class.pad_token_id)
                    encoded_inputs['input_ids'] = new_input_ids
                    encoded_inputs['attention_mask'] = new_attention_mask
                    encoded_inputs['labels'] = new_labels

            else:
                # Off-policy: use dataset responses, encode full messages
                data_source = DataSource.DATASET
                total_length = self.template.max_length + self.max_completion_length
                with self._template_context(self.template, max_length=total_length):
                    encoded_inputs = self._prepare_batch_inputs(inputs, encode_prompt_only=False)

            # Mark data source for downstream processing (e.g., conditional SFT loss)
            encoded_inputs['_data_source'] = data_source

        with self.template.forward_context(self.model, encoded_inputs):
            loss = HFSFTTrainer.training_step(self, model, encoded_inputs, num_items_in_batch)
        return loss

    def prediction_step(self, model, inputs, *args, **kwargs):
        # Prediction uses full messages
        encoded_inputs = self._prepare_batch_inputs(inputs, encode_prompt_only=False)
        with self.template.forward_context(self.model, encoded_inputs):
            return super().prediction_step(model, encoded_inputs, *args, **kwargs)

    @contextmanager
    def offload_context(self):
        """Context manager for offloading model and optimizer during vLLM inference

        This offloads:
        - Student model (self.model)
        - Optimizer states

        to CPU to free up GPU memory for vLLM engine.
        """
        if self.args.offload_model:
            self.offload_model(self.accelerator.unwrap_model(self.model))
        if getattr(self, 'optimizer', None) and self.args.offload_optimizer:
            self.offload_optimizer()

        try:
            yield
        finally:
            # reload (load back) model when exiting context
            if self.args.offload_model:
                self.load_model(self.accelerator.unwrap_model(self.model))
            if getattr(self, 'optimizer', None) and self.args.offload_optimizer:
                self.load_optimizer()

    def _get_random_num(self) -> float:
        """
        Generate a deterministic random number.

        Uses an isolated Random instance to avoid interfering with the global
        random state, ensuring thread-safety and consistent behavior across processes.

        Returns:
            float: A random number in the range [0.0, 1.0).
        """
        seed = int(getattr(self.args, 'seed', 0))
        seed += int(self.state.global_step)
        rng = random.Random(seed)
        return rng.random()

    @contextmanager
    def load_teacher_model_context(self):
        """Backward-compatible: loads the first (or only) teacher."""
        with self._load_single_teacher_context(0):
            yield

    def _prepare_liger_loss(self):
        """Initialize liger loss if enabled."""
        args = self.args
        self.use_liger_gkd_loss = False
        if getattr(args, 'use_liger_kernel', False):
            if not _liger_kernel_available:
                raise ImportError(
                    'Liger kernel is not installed. Please install liger-kernel by running: pip install liger-kernel')
            assert self.args.sft_alpha == 0, 'SFT loss is not supported with liger loss'

            self.liger_jsd_loss = LigerFusedLinearJSDLoss(
                beta=self.beta,
                ignore_index=-100,
                temperature=self.temperature,
                compiled=False,
            )
            self.use_liger_gkd_loss = True

    @staticmethod
    def generalized_jsd_loss(
        student_logits,
        teacher_logits,
        labels=None,
        beta=0.5,
        temperature=1.0,
        chunk_size=512,
    ):
        # Apply temperature scaling
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

        # Apply masking if labels provided
        if labels is not None:
            mask = labels != -100
            student_logits = student_logits[mask]
            teacher_logits = teacher_logits[mask]
            num_valid = mask.sum()
        else:
            # Flatten to [num_tokens, vocab_size]
            student_logits = student_logits.view(-1, student_logits.size(-1))
            teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))
            num_valid = student_logits.size(0)

        if num_valid == 0:
            return student_logits.new_zeros(())

        num_valid_int = num_valid if isinstance(num_valid, int) else num_valid.item()
        total_loss = student_logits.new_zeros(())

        # Precompute beta tensor once if needed
        if beta != 0 and beta != 1:
            beta_t = torch.tensor(beta, dtype=student_logits.dtype, device=student_logits.device)
            log_beta = torch.log(beta_t)
            log_1_minus_beta = torch.log1p(-beta_t)
        else:
            beta_t = log_beta = log_1_minus_beta = None

        # Process in chunks to reduce peak memory
        for start_idx in range(0, num_valid_int, chunk_size):
            end_idx = min(start_idx + chunk_size, num_valid_int)
            s_chunk = student_logits[start_idx:end_idx]
            t_chunk = teacher_logits[start_idx:end_idx]

            s_log_probs = F.log_softmax(s_chunk, dim=-1)
            t_log_probs = F.log_softmax(t_chunk, dim=-1)
            del s_chunk, t_chunk

            if beta == 0:
                jsd_chunk = F.kl_div(s_log_probs, t_log_probs, reduction='none', log_target=True)
            elif beta == 1:
                jsd_chunk = F.kl_div(t_log_probs, s_log_probs, reduction='none', log_target=True)
            else:
                mixture_log_probs = torch.logsumexp(
                    torch.stack([s_log_probs + log_1_minus_beta, t_log_probs + log_beta]),
                    dim=0,
                )

                kl_teacher = F.kl_div(mixture_log_probs, t_log_probs, reduction='none', log_target=True)
                kl_student = F.kl_div(mixture_log_probs, s_log_probs, reduction='none', log_target=True)
                del mixture_log_probs

                jsd_chunk = beta_t * kl_teacher + (1 - beta_t) * kl_student
                del kl_teacher, kl_student

            total_loss = total_loss + jsd_chunk.sum()
            del jsd_chunk, s_log_probs, t_log_probs

        return total_loss / num_valid

    def _prepare_logging(self):
        """Initialize logging components for on-policy rollout tracking."""
        args = self.args
        self.log_completions = args.log_completions
        self.wandb_log_unique_prompts = getattr(args, 'wandb_log_unique_prompts', False)
        self.jsonl_writer = JsonlWriter(os.path.join(self.args.output_dir, 'completions.jsonl'))

        # Initialize logs deque for storing rollout data (aligned with GRPO)
        self._logs = {
            'prompt': deque(),
            'completion': deque(),
        }

    def _apply_chat_template_to_messages_list(self, messages_list: DataType):
        """Convert messages list to prompt text list using template (aligned with GRPO)."""
        prompts_text = []
        for messages in messages_list:
            remove_response(messages)
            template_inputs = TemplateInputs.from_dict({'messages': messages})
            res = self.template.encode(template_inputs)
            prompts_text.append(self.template.safe_decode(res['input_ids']))
        return prompts_text

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """Override log method to include completion table logging (aligned with GRPO)."""
        import sys
        # Inject per-domain losses into logs before calling super (so WandB gets them).
        # Only main process logs/reports; ALL ranks clear to prevent unbounded memory growth.
        if self.accelerator.is_main_process and self._domain_loss_accum:
            for channel, losses in self._domain_loss_accum.items():
                avg_loss = sum(losses) / len(losses)
                logs[f'domain_loss/{channel}'] = round(avg_loss, 6)
                print(
                    f'[Step {self.state.global_step}] domain_loss/{channel} = {avg_loss:.4f}',
                    file=sys.stderr,
                    flush=True)
        # Always clear on every rank — accum is filled on all ranks in _compute_domain_weighted_jsd_loss
        self._domain_loss_accum.clear()

        # Call parent log method
        import transformers
        from packaging import version
        if version.parse(transformers.__version__) >= version.parse('4.47.0.dev0'):
            super().log(logs, start_time)
        else:
            super().log(logs)

        # Log completions table if we have data (only for on-policy generations)
        if self.accelerator.is_main_process and self.log_completions and len(self._logs['prompt']) > 0:
            seen_nums = len(self._logs['prompt'])
            table = {
                'step': [str(self.state.global_step)] * seen_nums,
                'prompt': list(self._logs['prompt'])[:seen_nums],
                'completion': list(self._logs['completion'])[:seen_nums],
            }

            # Write to jsonl
            self.jsonl_writer.append(table)

            self._logs['prompt'].clear()
            self._logs['completion'].clear()
            # Log to wandb if enabled
            report_to_wandb = self.args.report_to and 'wandb' in self.args.report_to and wandb.run is not None
            if report_to_wandb:
                wandb_table = table.copy()
                import pandas as pd
                df = pd.DataFrame(wandb_table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=['prompt'])
                wandb.log({'completions': wandb.Table(dataframe=df)})

            # Log to swanlab if enabled
            report_to_swanlab = self.args.report_to and 'swanlab' in self.args.report_to and swanlab.get_run(
            ) is not None
            if report_to_swanlab:
                headers = list(table.keys())
                rows = []
                for i in range(len(table['step'])):
                    row = [table[header][i] for header in headers]
                    rows.append(row)
                swanlab.log({'completions': swanlab.echarts.Table().add(headers, rows)})
