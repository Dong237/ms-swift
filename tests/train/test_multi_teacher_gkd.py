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


if __name__ == '__main__':
    # Run unit tests (no GPU required)
    pytest.main([__file__, '-v', '-k', 'TestTeacherDomainMapParsing or TestGetTeacherIndices'])
