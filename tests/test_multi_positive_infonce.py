"""Unit tests for MultiPositiveInfonceLoss and _parse_multi_positive_sentences.

These tests use synthetic embeddings and do not require GPU or model loading.
Integration tests for _embedding_encode/_embedding_data_collator use mock encoded dicts.
"""
import os
import unittest

import numpy as np
import torch
import torch.nn.functional as F

os.environ['INFONCE_TEMPERATURE'] = '0.1'
os.environ['INFONCE_USE_BATCH'] = 'False'


class TestParseMultiPositiveSentences(unittest.TestCase):

    def test_basic_two_groups(self):
        """Two groups: first has 2 positives + 2 negatives, second has 1 positive + 3 negatives."""
        from swift.loss.embedding import _parse_multi_positive_sentences

        D = 8
        # Group 1: anchor, pos0, pos1, neg0, neg1
        # Group 2: anchor, pos0, neg0, neg1, neg2
        # Labels (no anchor labels): [2, 1, 0, 0, 2, 0, 0, 0]
        sentences = torch.randn(10, D)  # 5 + 5 sentences
        labels = torch.tensor([2.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0])

        groups = _parse_multi_positive_sentences(sentences, labels)

        self.assertEqual(len(groups), 2)

        anchor0, pos0, neg0 = groups[0]
        self.assertEqual(pos0.shape[0], 2)
        self.assertEqual(neg0.shape[0], 2)
        self.assertTrue(torch.equal(anchor0, sentences[0]))
        self.assertTrue(torch.equal(pos0[0], sentences[1]))
        self.assertTrue(torch.equal(pos0[1], sentences[2]))
        self.assertTrue(torch.equal(neg0[0], sentences[3]))
        self.assertTrue(torch.equal(neg0[1], sentences[4]))

        anchor1, pos1, neg1 = groups[1]
        self.assertEqual(pos1.shape[0], 1)
        self.assertEqual(neg1.shape[0], 3)
        self.assertTrue(torch.equal(anchor1, sentences[5]))
        self.assertTrue(torch.equal(pos1[0], sentences[6]))

    def test_single_group_single_positive(self):
        """Single group with 1 positive + 2 negatives (should behave like standard InfoNCE)."""
        from swift.loss.embedding import _parse_multi_positive_sentences

        D = 8
        sentences = torch.randn(4, D)  # anchor, pos, neg0, neg1
        labels = torch.tensor([2.0, 0.0, 0.0])

        groups = _parse_multi_positive_sentences(sentences, labels)

        self.assertEqual(len(groups), 1)
        anchor, pos, neg = groups[0]
        self.assertEqual(pos.shape[0], 1)
        self.assertEqual(neg.shape[0], 2)

    def test_all_positives_no_negatives(self):
        """Group with only positives and no negatives."""
        from swift.loss.embedding import _parse_multi_positive_sentences

        D = 8
        sentences = torch.randn(4, D)  # anchor, pos0, pos1, pos2
        labels = torch.tensor([2.0, 1.0, 1.0])

        groups = _parse_multi_positive_sentences(sentences, labels)

        self.assertEqual(len(groups), 1)
        anchor, pos, neg = groups[0]
        self.assertEqual(pos.shape[0], 3)
        self.assertEqual(neg.shape[0], 0)


class TestMultiPositiveInfonceLoss(unittest.TestCase):

    def _make_loss(self):
        from swift.loss.embedding import MultiPositiveInfonceLoss
        # Create a minimal mock for args and trainer
        loss = MultiPositiveInfonceLoss.__new__(MultiPositiveInfonceLoss)
        loss.args = None
        loss.trainer = None
        loss.is_megatron = False
        return loss

    def test_loss_decreases_with_closer_positives(self):
        """Loss should be lower when positives are more similar to anchor."""
        loss_fn = self._make_loss()

        D = 16
        anchor = F.normalize(torch.randn(1, D), dim=1)

        # Scenario A: positives are close to anchor
        close_pos = F.normalize(anchor + 0.1 * torch.randn(2, D), dim=1)
        far_neg = F.normalize(torch.randn(3, D), dim=1)
        sentences_close = torch.cat([anchor, close_pos, far_neg], dim=0)
        labels = torch.tensor([2.0, 1.0, 0.0, 0.0, 0.0])

        loss_close = loss_fn({'last_hidden_state': sentences_close}, labels)

        # Scenario B: positives are far from anchor
        far_pos = F.normalize(torch.randn(2, D), dim=1)
        sentences_far = torch.cat([anchor, far_pos, far_neg], dim=0)

        loss_far = loss_fn({'last_hidden_state': sentences_far}, labels)

        self.assertLess(loss_close.item(), loss_far.item(),
                        'Loss should be lower when positives are closer to anchor')

    def test_all_positives_contribute_gradient(self):
        """All positive embeddings should receive non-zero gradients."""
        loss_fn = self._make_loss()

        D = 16
        anchor = F.normalize(torch.randn(1, D), dim=1)
        pos0 = F.normalize(torch.randn(1, D), dim=1, ).requires_grad_(True)
        pos1 = F.normalize(torch.randn(1, D), dim=1).requires_grad_(True)
        neg0 = F.normalize(torch.randn(1, D), dim=1)

        sentences = torch.cat([anchor, pos0, pos1, neg0], dim=0)
        labels = torch.tensor([2.0, 1.0, 0.0])

        loss = loss_fn({'last_hidden_state': sentences}, labels)
        loss.backward()

        self.assertIsNotNone(pos0.grad, 'First positive should have gradient')
        self.assertIsNotNone(pos1.grad, 'Second positive should have gradient')
        self.assertGreater(pos0.grad.abs().sum().item(), 0, 'First positive gradient should be non-zero')
        self.assertGreater(pos1.grad.abs().sum().item(), 0, 'Second positive gradient should be non-zero')

    def test_single_positive_equivalence(self):
        """With 1 positive, multi-positive loss should match standard InfoNCE (use_batch=False)."""
        from swift.loss.embedding import InfonceLoss

        multi_loss_fn = self._make_loss()
        single_loss_fn = InfonceLoss.__new__(InfonceLoss)
        single_loss_fn.args = None
        single_loss_fn.trainer = None
        single_loss_fn.is_megatron = False

        D = 16
        torch.manual_seed(42)
        anchor = F.normalize(torch.randn(1, D), dim=1)
        pos = F.normalize(torch.randn(1, D), dim=1)
        neg0 = F.normalize(torch.randn(1, D), dim=1)
        neg1 = F.normalize(torch.randn(1, D), dim=1)

        sentences = torch.cat([anchor, pos, neg0, neg1], dim=0)

        # Multi-positive labels (2.0 boundary, 0.0 negatives)
        mp_labels = torch.tensor([2.0, 0.0, 0.0])
        mp_loss = multi_loss_fn({'last_hidden_state': sentences}, mp_labels)

        # Standard infonce labels (1.0 boundary, 0.0 negatives)
        std_labels = torch.tensor([1.0, 0.0, 0.0])
        std_loss = single_loss_fn({'last_hidden_state': sentences}, std_labels)

        self.assertAlmostEqual(mp_loss.item(), std_loss.item(), places=5,
                               msg='Single-positive multi_positive_infonce should match standard infonce')

    def test_variable_counts_across_batch(self):
        """Different samples with different positive/negative counts should work."""
        loss_fn = self._make_loss()

        D = 16
        # Sample 1: 1 anchor + 3 positives + 1 negative = 5 sentences
        # Sample 2: 1 anchor + 1 positive + 4 negatives = 6 sentences
        s1_anchor = F.normalize(torch.randn(1, D), dim=1)
        s1_pos = F.normalize(torch.randn(3, D), dim=1)
        s1_neg = F.normalize(torch.randn(1, D), dim=1)

        s2_anchor = F.normalize(torch.randn(1, D), dim=1)
        s2_pos = F.normalize(torch.randn(1, D), dim=1)
        s2_neg = F.normalize(torch.randn(4, D), dim=1)

        sentences = torch.cat([s1_anchor, s1_pos, s1_neg, s2_anchor, s2_pos, s2_neg], dim=0)
        # Labels: sample1=[2,1,1,0], sample2=[2,0,0,0,0]
        labels = torch.tensor([2.0, 1.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0])

        loss = loss_fn({'last_hidden_state': sentences}, labels)

        self.assertFalse(torch.isnan(loss), 'Loss should not be NaN')
        self.assertFalse(torch.isinf(loss), 'Loss should not be Inf')
        self.assertGreater(loss.item(), 0, 'Loss should be positive')

    def test_no_negatives(self):
        """With only positives and no negatives, loss should be 0 (no contrastive signal)."""
        loss_fn = self._make_loss()

        D = 16
        anchor = F.normalize(torch.randn(1, D), dim=1)
        pos = F.normalize(torch.randn(2, D), dim=1)

        sentences = torch.cat([anchor, pos], dim=0)
        labels = torch.tensor([2.0, 1.0])

        loss = loss_fn({'last_hidden_state': sentences}, labels)
        # -log(sum_pos / sum_pos) = -log(1) = 0
        self.assertAlmostEqual(loss.item(), 0.0, places=5,
                               msg='Loss with no negatives should be 0')


class TestMultiPositiveInfonceMetrics(unittest.TestCase):

    def test_metrics_basic(self):
        """Metrics should compute margin, mean_pos, mean_neg correctly."""
        from swift.metrics.embedding import MultiPositiveInfonceMetrics

        metrics = MultiPositiveInfonceMetrics.__new__(MultiPositiveInfonceMetrics)

        D = 8
        anchor = F.normalize(torch.tensor([[1.0, 0, 0, 0, 0, 0, 0, 0]]), dim=1)
        pos0 = F.normalize(torch.tensor([[0.9, 0.1, 0, 0, 0, 0, 0, 0]]), dim=1)
        pos1 = F.normalize(torch.tensor([[0.8, 0.2, 0, 0, 0, 0, 0, 0]]), dim=1)
        neg0 = F.normalize(torch.tensor([[0, 0, 1.0, 0, 0, 0, 0, 0]]), dim=1)

        sentences = torch.cat([anchor, pos0, pos1, neg0], dim=0)
        labels = torch.tensor([2.0, 1.0, 0.0])

        result = metrics._calculate_metrics(sentences.numpy(), labels.numpy())

        self.assertIn('margin', result)
        self.assertIn('mean_pos', result)
        self.assertIn('mean_neg', result)
        self.assertGreater(result['mean_pos'], result['mean_neg'],
                           'Positive similarity should be higher than negative')
        self.assertGreater(result['margin'], 0, 'Margin should be positive')


class TestParserValidation(unittest.TestCase):

    def test_no_boundary_raises(self):
        """Parser should raise ValueError when no 2.0 boundary exists in labels."""
        from swift.loss.embedding import _parse_multi_positive_sentences

        sentences = torch.randn(3, 8)
        labels = torch.tensor([1.0, 0.0])  # no 2.0

        with self.assertRaises(ValueError, msg='Should reject labels without 2.0 boundary'):
            _parse_multi_positive_sentences(sentences, labels)

    def test_sentence_count_mismatch_raises(self):
        """Parser should raise ValueError when sentence/label counts are inconsistent."""
        from swift.loss.embedding import _parse_multi_positive_sentences

        sentences = torch.randn(5, 8)  # 5 sentences
        labels = torch.tensor([2.0, 0.0, 0.0])  # expects 1 group = 3 labels + 1 anchor = 4 sentences

        with self.assertRaises(ValueError, msg='Should reject mismatched sentence/label counts'):
            _parse_multi_positive_sentences(sentences, labels)

    def test_invalid_label_values_raises(self):
        """Parser should raise ValueError for label values outside {0, 1, 2}."""
        from swift.loss.embedding import _parse_multi_positive_sentences

        sentences = torch.randn(4, 8)
        labels = torch.tensor([2.0, 3.0, 0.0])  # 3.0 is invalid

        with self.assertRaises(ValueError, msg='Should reject invalid label values'):
            _parse_multi_positive_sentences(sentences, labels)


class TestCollatorIntegration(unittest.TestCase):
    """Integration tests that actually call _embedding_data_collator.

    Creates a minimal Template stub to test the real collator code path.
    """

    def _make_template_stub(self, multi_positive):
        """Create a minimal Template-like object with just enough to run the collator."""
        from swift.template.base import Template

        stub = object.__new__(Template)
        stub.multi_positive_embedding = multi_positive
        # _data_collator is called by _embedding_data_collator to pad and batch.
        # We stub it to just return the batch as-is with input_ids stacked.
        def _data_collator(batch, padding_to=None):
            # Minimal stub: collect input_ids from each sub-sample
            max_len = max(len(b.get('input_ids', [])) for b in batch)
            all_ids = []
            all_masks = []
            for b in batch:
                ids = b.get('input_ids', [])
                mask = b.get('attention_mask', [])
                # Pad to max_len
                all_ids.append(ids + [0] * (max_len - len(ids)))
                all_masks.append(mask + [0] * (max_len - len(mask)))
            return {
                'input_ids': torch.tensor(all_ids),
                'attention_mask': torch.tensor(all_masks),
            }
        stub._data_collator = _data_collator
        stub._fetch_inputs_startswith = Template._fetch_inputs_startswith
        return stub

    def test_old_infonce_collator_preserves_single_positive(self):
        """With multi_positive_embedding=False, collator uses 'positive_' prefix (no expansion)."""
        stub = self._make_template_stub(multi_positive=False)

        batch = [{
            'anchor_input_ids': [101, 1, 102],
            'anchor_attention_mask': [1, 1, 1],
            'positive_input_ids': [101, 2, 102],  # flat list[int]
            'positive_attention_mask': [1, 1, 1],
            'negative_input_ids': [[101, 3, 102]],
            'negative_attention_mask': [[1, 1, 1]],
            'labels': [1.0, 0.0],
        }]

        result = stub._embedding_data_collator(batch)

        # Should produce 3 sub-samples: anchor, positive, negative0
        self.assertEqual(result['input_ids'].shape[0], 3)
        self.assertEqual(result['num_samples'], 3)
        # Labels should be [1.0, 0.0] (old convention)
        self.assertEqual(result['labels'].tolist(), [1.0, 0.0])

    def test_multi_positive_collator_expands_two_positives(self):
        """With multi_positive_embedding=True, collator expands positive0_, positive1_."""
        stub = self._make_template_stub(multi_positive=True)

        batch = [{
            'anchor_input_ids': [101, 1, 102],
            'anchor_attention_mask': [1, 1, 1],
            'positive_input_ids': [[101, 2, 102], [101, 3, 102]],  # list of lists
            'positive_attention_mask': [[1, 1, 1], [1, 1, 1]],
            'negative_input_ids': [[101, 4, 102]],
            'negative_attention_mask': [[1, 1, 1]],
            'labels': [2.0, 1.0, 0.0],
        }]

        result = stub._embedding_data_collator(batch)

        # Should produce 4 sub-samples: anchor, positive0, positive1, negative0
        self.assertEqual(result['input_ids'].shape[0], 4)
        self.assertEqual(result['num_samples'], 4)
        # Labels should be [2.0, 1.0, 0.0]
        self.assertEqual(result['labels'].tolist(), [2.0, 1.0, 0.0])
        # Verify ordering: anchor=101,1, pos0=101,2, pos1=101,3, neg0=101,4
        self.assertEqual(result['input_ids'][0].tolist(), [101, 1, 102])
        self.assertEqual(result['input_ids'][1].tolist(), [101, 2, 102])
        self.assertEqual(result['input_ids'][2].tolist(), [101, 3, 102])
        self.assertEqual(result['input_ids'][3].tolist(), [101, 4, 102])

    def test_multi_positive_collator_single_positive(self):
        """With multi_positive_embedding=True and 1 positive, still expands to positive0_."""
        stub = self._make_template_stub(multi_positive=True)

        batch = [{
            'anchor_input_ids': [101, 1, 102],
            'anchor_attention_mask': [1, 1, 1],
            'positive_input_ids': [[101, 2, 102]],  # list with 1 element
            'positive_attention_mask': [[1, 1, 1]],
            'negative_input_ids': [[101, 3, 102]],
            'negative_attention_mask': [[1, 1, 1]],
            'labels': [2.0, 0.0],
        }]

        result = stub._embedding_data_collator(batch)

        # Should produce 3 sub-samples: anchor, positive0, negative0
        self.assertEqual(result['input_ids'].shape[0], 3)
        self.assertEqual(result['labels'].tolist(), [2.0, 0.0])

    def test_old_infonce_collator_not_broken_by_flat_list(self):
        """Regression: flat list[int] positive_input_ids must NOT be expanded as multi-positive."""
        stub = self._make_template_stub(multi_positive=False)

        batch = [{
            'anchor_input_ids': [101, 1, 102],
            'anchor_attention_mask': [1, 1, 1],
            'positive_input_ids': [101, 2, 102],  # flat list[int] — this IS a list
            'positive_attention_mask': [1, 1, 1],
            'labels': [1.0],
        }]

        result = stub._embedding_data_collator(batch)

        # Should produce 2 sub-samples: anchor, positive (NOT 3+ from token expansion)
        self.assertEqual(result['input_ids'].shape[0], 2,
                         'Flat positive_input_ids should NOT be expanded into per-token samples')
        self.assertEqual(result['num_samples'], 2)


if __name__ == '__main__':
    unittest.main()
