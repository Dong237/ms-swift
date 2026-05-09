"""Tests for embedding ip_id metadata plumbing.

Patch 2 only verifies that ip_id can travel from raw JSON rows through template
encoding/collation and into the embedding trainer loss side-channel. The current
losses do not consume ip_ids yet.
"""

import unittest
from types import MethodType, SimpleNamespace

import torch

from swift.template.base import Template
from swift.template.template_inputs import TemplateInputs
from swift.trainers.embedding_trainer import EmbeddingTrainer


class TestEmbeddingIpIdEncode(unittest.TestCase):

    def _make_template_stub(self, multi_positive=False):
        stub = object.__new__(Template)
        stub.is_training = True
        stub.multi_positive_embedding = multi_positive

        def _encode_truncated(self, inputs):
            value = len(inputs.images[0]) if inputs.images else 0
            return {
                'input_ids': [101, value, 102],
                'attention_mask': [1, 1, 1],
            }

        stub._encode_truncated = MethodType(_encode_truncated, stub)
        return stub

    def test_template_inputs_keeps_ip_id_in_extra_kwargs(self):
        inputs = TemplateInputs.from_dict({
            'messages': [{'role': 'user', 'content': '<image>'}],
            'images': ['anchor.jpg'],
            'positive_messages': [[{'role': 'user', 'content': '<image>'}]],
            'positive_images': [['pos.jpg']],
            'negative_messages': [[{'role': 'user', 'content': '<image>'}]],
            'negative_images': [['neg.jpg']],
            'ip_id': 42,
            'ip_name': '蜡笔小新_樱田妮妮',
        })

        self.assertEqual(inputs.chosen.extra_kwargs['ip_id'], 42)
        self.assertEqual(inputs.chosen.extra_kwargs['ip_name'], '蜡笔小新_樱田妮妮')

    def test_embedding_encode_writes_ip_id_when_present(self):
        stub = self._make_template_stub(multi_positive=True)
        inputs = TemplateInputs.from_dict({
            'messages': [{'role': 'user', 'content': '<image>'}],
            'images': ['anchor.jpg'],
            'positive_messages': [
                [{'role': 'user', 'content': '<image>'}],
                [{'role': 'user', 'content': '<image>'}],
            ],
            'positive_images': [['pos1.jpg'], ['pos2.jpg']],
            'negative_messages': [[{'role': 'user', 'content': '<image>'}]],
            'negative_images': [['neg.jpg']],
            'ip_id': '7',
        })

        encoded = stub._embedding_encode(inputs)

        self.assertEqual(encoded['ip_id'], 7)
        self.assertEqual(encoded['labels'], [2.0, 1.0, 0.0])

    def test_embedding_encode_omits_ip_id_when_missing(self):
        stub = self._make_template_stub()
        inputs = TemplateInputs.from_dict({
            'messages': [{'role': 'user', 'content': '<image>'}],
            'images': ['anchor.jpg'],
            'positive_messages': [[{'role': 'user', 'content': '<image>'}]],
            'positive_images': [['pos.jpg']],
            'negative_messages': [],
            'negative_images': [],
        })

        encoded = stub._embedding_encode(inputs)

        self.assertNotIn('ip_id', encoded)
        self.assertEqual(encoded['labels'], [1.0])


class TestEmbeddingIpIdCollator(unittest.TestCase):

    def _make_template_stub(self, multi_positive=False):
        stub = object.__new__(Template)
        stub.multi_positive_embedding = multi_positive

        def _data_collator(batch, padding_to=None):
            max_len = max(len(b.get('input_ids', [])) for b in batch)
            all_ids = []
            all_masks = []
            for b in batch:
                ids = b.get('input_ids', [])
                mask = b.get('attention_mask', [])
                all_ids.append(ids + [0] * (max_len - len(ids)))
                all_masks.append(mask + [0] * (max_len - len(mask)))
            return {
                'input_ids': torch.tensor(all_ids),
                'attention_mask': torch.tensor(all_masks),
            }

        stub._data_collator = _data_collator
        stub._fetch_inputs_startswith = Template._fetch_inputs_startswith
        return stub

    def test_collator_outputs_batch_level_ip_ids(self):
        stub = self._make_template_stub(multi_positive=True)
        batch = [{
            'anchor_input_ids': [101, 1, 102],
            'anchor_attention_mask': [1, 1, 1],
            'positive_input_ids': [[101, 2, 102], [101, 3, 102]],
            'positive_attention_mask': [[1, 1, 1], [1, 1, 1]],
            'negative_input_ids': [[101, 4, 102]],
            'negative_attention_mask': [[1, 1, 1]],
            'labels': [2.0, 1.0, 0.0],
            'ip_id': 42,
        }, {
            'anchor_input_ids': [101, 5, 102],
            'anchor_attention_mask': [1, 1, 1],
            'positive_input_ids': [[101, 6, 102]],
            'positive_attention_mask': [[1, 1, 1]],
            'negative_input_ids': [[101, 7, 102]],
            'negative_attention_mask': [[1, 1, 1]],
            'labels': [2.0, 0.0],
            'ip_id': 99,
        }]

        result = stub._embedding_data_collator(batch)

        self.assertEqual(result['num_samples'], 7)
        self.assertEqual(result['labels'].tolist(), [2.0, 1.0, 0.0, 2.0, 0.0])
        self.assertEqual(result['ip_ids'].tolist(), [42, 99])

    def test_collator_omits_ip_ids_for_old_data(self):
        stub = self._make_template_stub()
        batch = [{
            'anchor_input_ids': [101, 1, 102],
            'anchor_attention_mask': [1, 1, 1],
            'positive_input_ids': [101, 2, 102],
            'positive_attention_mask': [1, 1, 1],
            'labels': [1.0],
        }]

        result = stub._embedding_data_collator(batch)

        self.assertNotIn('ip_ids', result)
        self.assertEqual(result['num_samples'], 2)

    def test_collator_uses_minus_one_sentinel_for_mixed_old_rows(self):
        stub = self._make_template_stub()
        batch = [{
            'anchor_input_ids': [101, 1, 102],
            'anchor_attention_mask': [1, 1, 1],
            'positive_input_ids': [101, 2, 102],
            'positive_attention_mask': [1, 1, 1],
            'labels': [1.0],
            'ip_id': 42,
        }, {
            'anchor_input_ids': [101, 3, 102],
            'anchor_attention_mask': [1, 1, 1],
            'positive_input_ids': [101, 4, 102],
            'positive_attention_mask': [1, 1, 1],
            'labels': [1.0],
        }]

        result = stub._embedding_data_collator(batch)

        self.assertEqual(result['ip_ids'].tolist(), [42, -1])
        self.assertEqual(result['labels'].tolist(), [1.0, 1.0])


class TestEmbeddingTrainerIpIdSideChannel(unittest.TestCase):

    def _make_trainer_stub(self, loss_capture):
        trainer = object.__new__(EmbeddingTrainer)
        trainer.compute_loss_func = loss_capture
        trainer.model_accepts_loss_kwargs = False
        trainer.args = SimpleNamespace(gradient_accumulation_steps=1)
        trainer._compute_acc = lambda outputs, labels: None
        return trainer

    def test_compute_loss_pops_ip_ids_before_model_forward_and_passes_to_loss(self):
        captured = {}

        def loss_capture(outputs, labels, **kwargs):
            captured['labels'] = labels
            captured['kwargs'] = kwargs
            return torch.tensor(1.25)

        class Model:
            def __call__(self, **kwargs):
                captured['model_kwargs'] = kwargs
                return SimpleNamespace(loss=None, last_hidden_state=kwargs['input_ids'].float())

        trainer = self._make_trainer_stub(loss_capture)
        inputs = {
            'input_ids': torch.ones(3, 2),
            'labels': torch.tensor([2.0, 1.0]),
            'ip_ids': torch.tensor([42]),
        }

        loss = trainer.compute_loss(Model(), inputs)

        self.assertEqual(float(loss), 1.25)
        self.assertNotIn('ip_ids', captured['model_kwargs'])
        self.assertTrue(torch.equal(captured['kwargs']['ip_ids'], torch.tensor([42])))
        self.assertTrue(torch.equal(captured['labels'], torch.tensor([2.0, 1.0])))

    def test_compute_loss_omits_ip_ids_kwarg_when_missing(self):
        captured = {}

        def loss_capture(outputs, labels, **kwargs):
            captured['kwargs'] = kwargs
            return torch.tensor(0.5)

        class Model:
            def __call__(self, **kwargs):
                return SimpleNamespace(loss=None, last_hidden_state=kwargs['input_ids'].float())

        trainer = self._make_trainer_stub(loss_capture)
        inputs = {
            'input_ids': torch.ones(2, 2),
            'labels': torch.tensor([1.0]),
        }

        loss = trainer.compute_loss(Model(), inputs)

        self.assertEqual(float(loss), 0.5)
        self.assertNotIn('ip_ids', captured['kwargs'])


if __name__ == '__main__':
    unittest.main()
