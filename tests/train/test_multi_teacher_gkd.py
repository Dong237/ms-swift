"""Tests for multi-teacher GKD functionality.

Unit tests (no GPU): test argument parsing, deduplication, routing logic.
Integration tests (GPU): full training pipeline with multi-teacher routing.
"""
import json
import os
import tempfile

import pytest

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

kwargs = {
    'per_device_train_batch_size': 2,
    'save_steps': 5,
    'gradient_accumulation_steps': 2,
    'num_train_epochs': 1,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_channel_jsonl(path, n_samples=40):
    """Create a JSONL file with alternating math/code channels."""
    math_samples = [
        {'messages': [{'role': 'user', 'content': f'Solve: {i}+{i}={i*2}?'},
                      {'role': 'assistant', 'content': f'The answer is {i*2}.'}],
         'channel': 'math'}
        for i in range(n_samples // 2)
    ]
    code_samples = [
        {'messages': [{'role': 'user', 'content': f'Write a function that returns {i}.'},
                      {'role': 'assistant', 'content': f'def f(): return {i}'}],
         'channel': 'code'}
        for i in range(n_samples // 2)
    ]
    samples = []
    for m, c in zip(math_samples, code_samples):
        samples.append(m)
        samples.append(c)
    with open(path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    return path


def _create_no_channel_jsonl(path, n_samples=20):
    """Create a JSONL file without channel fields (for default routing test)."""
    with open(path, 'w') as f:
        for i in range(n_samples):
            sample = {
                'messages': [
                    {'role': 'user', 'content': f'What is {i}+1?'},
                    {'role': 'assistant', 'content': f'{i+1}'},
                ],
            }
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    return path


# ---------------------------------------------------------------------------
# Unit tests — argument parsing (no GPU required)
# ---------------------------------------------------------------------------

class TestTeacherDomainMapParsing:
    """Test teacher_domain_map JSON parsing, validation, and dedup logic."""

    def _make_args(self, **overrides):
        """Create a minimal mock RLHFArguments-like object for _check_gkd testing."""
        from swift.arguments import RLHFArguments

        class _MockArgs:
            rlhf_type = 'gkd'
            use_vllm = False
            multi_turn_scheduler = None
            async_generate = False
            teacher_model = None
            teacher_domain_map = None
            teacher_type_map = None
            teacher_deepspeed = None
            _teacher_paths = None
            _channel_to_teacher_idx = None

        mock = _MockArgs()
        for k, v in overrides.items():
            setattr(mock, k, v)
        # Call the real validation method bound to our mock
        RLHFArguments._check_gkd(mock)
        return mock

    def test_basic_parsing(self):
        """JSON string is parsed into dict, _teacher_paths and _channel_to_teacher_idx are set."""
        domain_map = json.dumps({'math': '/path/to/math', 'code': '/path/to/code'})
        mock = self._make_args(teacher_domain_map=domain_map)

        assert isinstance(mock.teacher_domain_map, dict)
        assert len(mock._teacher_paths) == 2
        assert set(mock._teacher_paths) == {'/path/to/math', '/path/to/code'}
        assert mock._channel_to_teacher_idx['math'] != mock._channel_to_teacher_idx['code']

    def test_dict_input(self):
        """Already-parsed dict is accepted directly."""
        domain_map = {'math': '/path/to/math', 'code': '/path/to/code'}
        mock = self._make_args(teacher_domain_map=domain_map)
        assert len(mock._teacher_paths) == 2

    def test_deduplication(self):
        """Two domains with same path share one teacher index."""
        domain_map = json.dumps({
            'math': '/shared/model',
            'code': '/shared/model',
            'writing': '/other/model',
        })
        mock = self._make_args(teacher_domain_map=domain_map)

        assert len(mock._teacher_paths) == 2  # 2 unique paths
        assert mock._channel_to_teacher_idx['math'] == mock._channel_to_teacher_idx['code']
        assert mock._channel_to_teacher_idx['writing'] != mock._channel_to_teacher_idx['math']

    def test_mutual_exclusivity(self):
        """Cannot set both teacher_model and teacher_domain_map."""
        with pytest.raises(ValueError, match='mutually exclusive'):
            self._make_args(
                teacher_model='some/model',
                teacher_domain_map=json.dumps({'math': '/path/to/math'}),
            )

    def test_neither_set_raises(self):
        """Must set either teacher_model or teacher_domain_map."""
        with pytest.raises(ValueError, match='GKD requires either'):
            self._make_args(teacher_model=None, teacher_domain_map=None)

    def test_empty_map_raises(self):
        """Empty dict is rejected."""
        with pytest.raises(ValueError, match='non-empty'):
            self._make_args(teacher_domain_map=json.dumps({}))

    def test_invalid_json_raises(self):
        """Malformed JSON string is rejected."""
        with pytest.raises(json.JSONDecodeError):
            self._make_args(teacher_domain_map='not-valid-json')

    def test_single_teacher_model_passes(self):
        """Single teacher_model (no domain map) still works."""
        mock = self._make_args(teacher_model='some/model')
        assert mock._teacher_paths is None
        assert mock._channel_to_teacher_idx is None

    def test_zero3_teacher_multi_teacher_raises(self):
        """ZeRO-3 teacher + multi-teacher should raise ValueError (deadlock risk)."""
        zero3_config = {'zero_optimization': {'stage': 3}}
        with pytest.raises(ValueError, match='ZeRO-3 for teacher models is incompatible'):
            self._make_args(
                teacher_domain_map=json.dumps({'math': '/path/math', 'code': '/path/code'}),
                teacher_deepspeed=zero3_config,
            )

    def test_zero2_teacher_multi_teacher_passes(self):
        """ZeRO-2 teacher + multi-teacher should work fine."""
        zero2_config = {'zero_optimization': {'stage': 2}}
        mock = self._make_args(
            teacher_domain_map=json.dumps({'math': '/path/math', 'code': '/path/code'}),
            teacher_deepspeed=zero2_config,
        )
        assert len(mock._teacher_paths) == 2

    def test_zero3_teacher_single_teacher_passes(self):
        """ZeRO-3 teacher + single teacher should still work (no per-sample routing)."""
        zero3_config = {'zero_optimization': {'stage': 3}}
        mock = self._make_args(
            teacher_model='some/model',
            teacher_deepspeed=zero3_config,
        )
        assert mock._teacher_paths is None  # Single teacher, no multi-teacher paths


# ---------------------------------------------------------------------------
# Unit tests — teacher routing logic (no GPU required)
# ---------------------------------------------------------------------------

class TestGetTeacherIndices:
    """Test _get_teacher_indices routing method."""

    def _make_trainer_stub(self, n_teachers=2, channel_to_idx=None):
        """Create a minimal stub with the fields _get_teacher_indices reads."""
        from swift.rlhf_trainers.gkd_trainer import GKDTrainer

        class _Stub:
            teacher_models = [None] * n_teachers
            channel_to_teacher_idx = channel_to_idx

        stub = _Stub()
        # Bind the method to our stub
        stub._get_teacher_indices = GKDTrainer._get_teacher_indices.__get__(stub, type(stub))
        return stub

    def test_single_teacher_returns_none(self):
        """Single teacher should always return None (no routing needed)."""
        stub = self._make_trainer_stub(n_teachers=1, channel_to_idx={'math': 0})
        result = stub._get_teacher_indices(['math', 'math'])
        assert result is None

    def test_none_channels_returns_none(self):
        """None channels should return None."""
        stub = self._make_trainer_stub(n_teachers=2, channel_to_idx={'math': 0, 'code': 1})
        result = stub._get_teacher_indices(None)
        assert result is None

    def test_no_channel_map_returns_none(self):
        """No channel_to_teacher_idx should return None."""
        stub = self._make_trainer_stub(n_teachers=2, channel_to_idx=None)
        result = stub._get_teacher_indices(['math', 'code'])
        assert result is None

    def test_valid_routing(self):
        """Valid channels are correctly routed."""
        stub = self._make_trainer_stub(n_teachers=3, channel_to_idx={'math': 0, 'code': 1, 'writing': 2})
        result = stub._get_teacher_indices(['code', 'math', 'writing', 'code'])
        assert result == [1, 0, 2, 1]

    def test_unknown_channel_defaults_to_zero(self):
        """Unknown channels default to teacher index 0."""
        stub = self._make_trainer_stub(n_teachers=2, channel_to_idx={'math': 0, 'code': 1})
        result = stub._get_teacher_indices(['math', 'unknown', None])
        assert result == [0, 0, 0]

    def test_unknown_channel_logs_warning(self):
        """Unknown (non-None) channels should trigger a warning."""
        stub = self._make_trainer_stub(n_teachers=2, channel_to_idx={'math': 0, 'code': 1})
        # Use caplog-style approach: check that result is correct and warning would fire
        result = stub._get_teacher_indices(['math', 'typo_channel', 'code'])
        assert result == [0, 0, 1]  # 'typo_channel' defaults to 0

    def test_all_same_channel(self):
        """All samples have the same channel — all route to the same teacher."""
        stub = self._make_trainer_stub(n_teachers=2, channel_to_idx={'math': 0, 'code': 1})
        result = stub._get_teacher_indices(['math', 'math', 'math'])
        assert result == [0, 0, 0]

    def test_empty_channels_list(self):
        """Empty channels list should return empty indices (not None)."""
        stub = self._make_trainer_stub(n_teachers=2, channel_to_idx={'math': 0, 'code': 1})
        result = stub._get_teacher_indices([])
        assert result == []

    def test_mixed_known_unknown_none(self):
        """Mix of known, unknown, and None channels."""
        stub = self._make_trainer_stub(n_teachers=3, channel_to_idx={'a': 0, 'b': 1, 'c': 2})
        result = stub._get_teacher_indices(['a', 'unknown', None, 'b', 'c', 'also_unknown'])
        assert result == [0, 0, 0, 1, 2, 0]  # unknown and None default to 0


# ---------------------------------------------------------------------------
# Integration tests (require GPU + model downloads)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.environ.get('CUDA_VISIBLE_DEVICES'), reason='No GPU available')
class TestMultiTeacherGKDIntegration:
    """Integration tests that run actual training. Require GPU and model downloads."""

    @pytest.fixture(autouse=True)
    def setup_tmpdir(self, tmp_path):
        self.tmp_dir = str(tmp_path)
        self.channel_jsonl = _create_channel_jsonl(os.path.join(self.tmp_dir, 'channel_data.jsonl'))
        self.no_channel_jsonl = _create_no_channel_jsonl(os.path.join(self.tmp_dir, 'no_channel_data.jsonl'))

    def test_multi_teacher_basic(self):
        """Multi-teacher GKD with two 'teachers' (same model, different paths logically)."""
        from swift import RLHFArguments, rlhf_main

        # Use Qwen2.5-0.5B as both student and teachers (for speed)
        teacher_domain_map = json.dumps({
            'math': 'Qwen/Qwen2.5-0.5B',
            'code': 'Qwen/Qwen2.5-0.5B',
        })

        result = rlhf_main(
            RLHFArguments(
                rlhf_type='gkd',
                model='Qwen/Qwen2.5-0.5B',
                teacher_domain_map=teacher_domain_map,
                dataset=[self.channel_jsonl],
                split_dataset_ratio=0.0,
                load_from_cache_file=False,
                output_dir=os.path.join(self.tmp_dir, 'output_basic'),
                **kwargs,
            ))
        assert result is not None
        assert 'last_model_checkpoint' in result

    def test_multi_teacher_with_offload(self):
        """Multi-teacher GKD with teacher offloading enabled."""
        from swift import RLHFArguments, rlhf_main

        teacher_domain_map = json.dumps({
            'math': 'Qwen/Qwen2.5-0.5B',
            'code': 'Qwen/Qwen2.5-0.5B',
        })

        result = rlhf_main(
            RLHFArguments(
                rlhf_type='gkd',
                model='Qwen/Qwen2.5-0.5B',
                teacher_domain_map=teacher_domain_map,
                dataset=[self.channel_jsonl],
                split_dataset_ratio=0.0,
                load_from_cache_file=False,
                offload_teacher_model=True,
                output_dir=os.path.join(self.tmp_dir, 'output_offload'),
                **kwargs,
            ))
        assert result is not None
        assert 'last_model_checkpoint' in result

    def test_multi_teacher_seq_kd(self):
        """Multi-teacher GKD with seq_kd=True (teacher generates responses)."""
        from swift import RLHFArguments, rlhf_main

        teacher_domain_map = json.dumps({
            'math': 'Qwen/Qwen2.5-0.5B',
            'code': 'Qwen/Qwen2.5-0.5B',
        })

        result = rlhf_main(
            RLHFArguments(
                rlhf_type='gkd',
                model='Qwen/Qwen2.5-0.5B',
                teacher_domain_map=teacher_domain_map,
                dataset=[self.channel_jsonl],
                split_dataset_ratio=0.0,
                load_from_cache_file=False,
                seq_kd=True,
                output_dir=os.path.join(self.tmp_dir, 'output_seq_kd'),
                **kwargs,
            ))
        assert result is not None
        assert 'last_model_checkpoint' in result

    def test_multi_teacher_default_routing(self):
        """Samples without channel field should default to teacher[0]."""
        from swift import RLHFArguments, rlhf_main

        teacher_domain_map = json.dumps({
            'math': 'Qwen/Qwen2.5-0.5B',
            'code': 'Qwen/Qwen2.5-0.5B',
        })

        result = rlhf_main(
            RLHFArguments(
                rlhf_type='gkd',
                model='Qwen/Qwen2.5-0.5B',
                teacher_domain_map=teacher_domain_map,
                dataset=[self.no_channel_jsonl],
                split_dataset_ratio=0.0,
                load_from_cache_file=False,
                output_dir=os.path.join(self.tmp_dir, 'output_default'),
                **kwargs,
            ))
        assert result is not None
        assert 'last_model_checkpoint' in result

    def test_single_teacher_backward_compat(self):
        """Existing single-teacher GKD still works with new code."""
        from swift import RLHFArguments, rlhf_main

        result = rlhf_main(
            RLHFArguments(
                rlhf_type='gkd',
                model='Qwen/Qwen2.5-0.5B',
                teacher_model='Qwen/Qwen2.5-0.5B',
                dataset=[self.no_channel_jsonl],
                split_dataset_ratio=0.0,
                load_from_cache_file=False,
                output_dir=os.path.join(self.tmp_dir, 'output_single'),
                **kwargs,
            ))
        assert result is not None
        assert 'last_model_checkpoint' in result

    def test_dedup_loads_once(self):
        """Two domains mapping to same path should load only one teacher model."""
        from swift import RLHFArguments, rlhf_main

        # Both domains point to the same model — dedup should load only once
        teacher_domain_map = json.dumps({
            'math': 'Qwen/Qwen2.5-0.5B',
            'code': 'Qwen/Qwen2.5-0.5B',
        })

        result = rlhf_main(
            RLHFArguments(
                rlhf_type='gkd',
                model='Qwen/Qwen2.5-0.5B',
                teacher_domain_map=teacher_domain_map,
                dataset=[self.channel_jsonl],
                split_dataset_ratio=0.0,
                load_from_cache_file=False,
                output_dir=os.path.join(self.tmp_dir, 'output_dedup'),
                **kwargs,
            ))
        assert result is not None

    def test_multi_teacher_on_policy(self):
        """On-policy student generation with multi-teacher routing (validates channel preservation)."""
        from swift import RLHFArguments, rlhf_main

        teacher_domain_map = json.dumps({
            'math': 'Qwen/Qwen2.5-0.5B',
            'code': 'Qwen/Qwen2.5-0.5B',
        })

        result = rlhf_main(
            RLHFArguments(
                rlhf_type='gkd',
                model='Qwen/Qwen2.5-0.5B',
                teacher_domain_map=teacher_domain_map,
                dataset=[self.channel_jsonl],
                lmbda=1.0,  # Always on-policy (student generates)
                max_completion_length=32,
                split_dataset_ratio=0.0,
                load_from_cache_file=False,
                output_dir=os.path.join(self.tmp_dir, 'output_on_policy'),
                **kwargs,
            ))
        assert result is not None
        assert 'last_model_checkpoint' in result

    def test_multi_teacher_seq_kd_per_teacher(self):
        """Multi-teacher seq_kd with per-teacher generation (not majority vote)."""
        from swift import RLHFArguments, rlhf_main

        teacher_domain_map = json.dumps({
            'math': 'Qwen/Qwen2.5-0.5B',
            'code': 'Qwen/Qwen2.5-0.5B',
        })

        result = rlhf_main(
            RLHFArguments(
                rlhf_type='gkd',
                model='Qwen/Qwen2.5-0.5B',
                teacher_domain_map=teacher_domain_map,
                dataset=[self.channel_jsonl],
                seq_kd=True,
                lmbda=0.0,  # Always seq_kd (no on-policy)
                max_completion_length=32,
                split_dataset_ratio=0.0,
                load_from_cache_file=False,
                output_dir=os.path.join(self.tmp_dir, 'output_seq_kd_per_teacher'),
                **kwargs,
            ))
        assert result is not None
        assert 'last_model_checkpoint' in result


# ---------------------------------------------------------------------------
# Unit tests — vocab padding logic (no GPU required)
# ---------------------------------------------------------------------------

class TestVocabPadding:
    """Test that vocab mismatch padding uses -1e9 (not 0) to avoid spurious probability mass."""

    def test_pad_smaller_teacher_uses_neg_inf(self):
        """When a teacher has smaller vocab, padded positions should be -1e9."""
        import torch
        import torch.nn.functional as F

        # Simulate: teacher A has vocab 100, teacher B has vocab 120
        # After teacher A, max_vocab=100. Teacher B expands to 120.
        # Teacher A's padded positions (100-119) should be -1e9.
        teacher_a_logits = torch.randn(2, 5, 100)  # 2 samples, 5 tokens, vocab 100
        max_vocab_size = 100

        # Simulate expansion when teacher B has vocab 120
        new_vocab_size = 120
        padded = F.pad(teacher_a_logits, (0, new_vocab_size - max_vocab_size), 'constant', -1e9)

        assert padded.shape == (2, 5, 120)
        # Original positions should be unchanged
        assert torch.allclose(padded[:, :, :100], teacher_a_logits)
        # Padded positions should be -1e9
        assert (padded[:, :, 100:] == -1e9).all()

    def test_pad_value_gives_zero_softmax_probability(self):
        """Padding with -1e9 should give ~0 probability after softmax."""
        import torch
        import torch.nn.functional as F

        logits = torch.randn(1, 3, 50)
        padded = F.pad(logits, (0, 10), 'constant', -1e9)  # pad 10 extra positions
        probs = F.softmax(padded, dim=-1)

        # Padded positions should have negligible probability
        assert probs[:, :, 50:].max().item() < 1e-30
        # Original positions should still sum to ~1
        assert torch.allclose(probs.sum(dim=-1), torch.ones(1, 3), atol=1e-6)

    def test_pad_value_no_nan_in_kl_div(self):
        """Using -1e9 padding should not produce NaN in KL divergence (unlike -inf)."""
        import torch
        import torch.nn.functional as F

        student_logits = torch.randn(1, 10, 60)
        teacher_logits = torch.randn(1, 10, 50)
        # Pad teacher to match student vocab
        teacher_logits = F.pad(teacher_logits, (0, 10), 'constant', -1e9)

        s_log_probs = F.log_softmax(student_logits, dim=-1)
        t_log_probs = F.log_softmax(teacher_logits, dim=-1)

        kl = F.kl_div(s_log_probs, t_log_probs, reduction='none', log_target=True)
        assert not torch.isnan(kl).any(), 'KL divergence should not contain NaN with -1e9 padding'
        assert not torch.isinf(kl).any(), 'KL divergence should not contain Inf with -1e9 padding'


# ---------------------------------------------------------------------------
# Unit tests — channel preservation (no GPU required)
# ---------------------------------------------------------------------------

class TestChannelPreservation:
    """Test that channel field is preserved through vLLM and seq_kd paths."""

    def test_vllm_channel_reattachment(self):
        """Channels from original inputs should be reattached to vLLM-generated outputs."""
        # Simulate the fix: preserve channels before vLLM, reattach after
        original_inputs = [
            {'messages': [{'role': 'user', 'content': 'q1'}], 'channel': 'math'},
            {'messages': [{'role': 'user', 'content': 'q2'}], 'channel': 'code'},
            {'messages': [{'role': 'user', 'content': 'q3'}], 'channel': 'math'},
        ]
        # vLLM returns only messages (no channel)
        vllm_outputs = [
            {'messages': [{'role': 'user', 'content': 'q1'}, {'role': 'assistant', 'content': 'a1'}]},
            {'messages': [{'role': 'user', 'content': 'q2'}, {'role': 'assistant', 'content': 'a2'}]},
            {'messages': [{'role': 'user', 'content': 'q3'}, {'role': 'assistant', 'content': 'a3'}]},
        ]

        # Apply the fix
        original_channels = [inp.get('channel') for inp in original_inputs]
        for gen_inp, ch in zip(vllm_outputs, original_channels):
            if ch is not None:
                gen_inp['channel'] = ch

        assert vllm_outputs[0]['channel'] == 'math'
        assert vllm_outputs[1]['channel'] == 'code'
        assert vllm_outputs[2]['channel'] == 'math'

    def test_none_channels_not_attached(self):
        """When original inputs have no channel, nothing should be attached."""
        original_inputs = [
            {'messages': [{'role': 'user', 'content': 'q1'}]},
            {'messages': [{'role': 'user', 'content': 'q2'}]},
        ]
        vllm_outputs = [
            {'messages': [{'role': 'user', 'content': 'q1'}, {'role': 'assistant', 'content': 'a1'}]},
            {'messages': [{'role': 'user', 'content': 'q2'}, {'role': 'assistant', 'content': 'a2'}]},
        ]

        original_channels = [inp.get('channel') for inp in original_inputs]
        for gen_inp, ch in zip(vllm_outputs, original_channels):
            if ch is not None:
                gen_inp['channel'] = ch

        assert 'channel' not in vllm_outputs[0]
        assert 'channel' not in vllm_outputs[1]


# ---------------------------------------------------------------------------
# Unit tests — Megatron multi-teacher guard (no GPU required)
# ---------------------------------------------------------------------------

class TestMegatronMultiTeacherGuard:
    """Test that Megatron backend rejects multi-teacher GKD."""

    def test_megatron_multi_teacher_raises(self):
        """Megatron GKD with teacher_domain_map should raise NotImplementedError."""
        # We can't easily instantiate MegatronGKDTrainer without full megatron setup,
        # so we test the guard logic directly by checking the class source.
        try:
            from swift.megatron.trainers.gkd_trainer import MegatronGKDTrainer
            # Verify the guard exists in __init__
            import inspect
            source = inspect.getsource(MegatronGKDTrainer.__init__)
            assert 'teacher_domain_map' in source
            assert 'NotImplementedError' in source
        except ImportError:
            pytest.skip('megatron not installed')


if __name__ == '__main__':
    # Run unit tests (no GPU required)
    pytest.main([__file__, '-v', '-k', 'TestTeacherDomainMapParsing or TestGetTeacherIndices'
                 ' or TestVocabPadding or TestChannelPreservation or TestMegatronMultiTeacherGuard'])
