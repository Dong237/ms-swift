"""Unit tests for SubcenterArcFaceHead and Stage2IpEmbeddingLoss.

Tests cover: head shape/gradient, target-only margin, loss composition,
ip_id=-1 skipping, missing ip_ids error, and proxy parameter registration.
"""
import os
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ['INFONCE_TEMPERATURE'] = '0.1'
os.environ['INFONCE_USE_BATCH'] = 'False'


class TestSubcenterArcFaceHead(unittest.TestCase):

    def test_shape_and_gradient(self):
        """Proxy has correct shape, loss backward gives non-None gradient."""
        from swift.loss.embedding import SubcenterArcFaceHead

        C, K, D = 10, 3, 16
        head = SubcenterArcFaceHead(C, D, K=K, scale=64.0, margin=0.2)

        self.assertEqual(head.proxies.shape, (C, K, D))

        N = 4
        emb = F.normalize(torch.randn(N, D), dim=1)
        targets = torch.tensor([0, 3, 7, 2])

        loss = head(emb, targets)

        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0)

        loss.backward()
        self.assertIsNotNone(head.proxies.grad)
        self.assertGreater(head.proxies.grad.abs().sum().item(), 0)

    def test_margin_only_on_target(self):
        """Angular margin is applied ONLY to the target class logit."""
        from swift.loss.embedding import SubcenterArcFaceHead

        C, K, D = 5, 2, 8
        head_margin = SubcenterArcFaceHead(C, D, K=K, scale=1.0, margin=0.5)
        head_no_margin = SubcenterArcFaceHead(C, D, K=K, scale=1.0, margin=0.0)

        # Share the same proxies
        head_no_margin.proxies.data = head_margin.proxies.data.clone()

        emb = F.normalize(torch.randn(2, D), dim=1)
        targets = torch.tensor([1, 3])

        # Compute cosine similarities manually (no margin)
        W = F.normalize(head_margin.proxies.data, p=2, dim=2)
        cosine = torch.einsum('nd,ckd->nck', F.normalize(emb, p=2, dim=1), W)
        cosine_max, _ = cosine.max(dim=2)  # [N, C]

        # With margin=0, the head should return raw cosine * scale
        loss_no_margin = head_no_margin(emb, targets)

        # With margin>0, only target logits change (get smaller),
        # so loss should be higher
        loss_margin = head_margin(emb, targets)

        self.assertGreater(loss_margin.item(), loss_no_margin.item(),
                           'Margin should increase loss by penalizing target logit')

    def test_numerical_stability_extreme_cosine(self):
        """Head should not produce NaN even when embeddings are nearly identical to a proxy."""
        from swift.loss.embedding import SubcenterArcFaceHead

        C, K, D = 3, 2, 8
        head = SubcenterArcFaceHead(C, D, K=K)

        # Make embedding nearly identical to proxy 0's first sub-center
        emb = F.normalize(head.proxies.data[0, 0].unsqueeze(0).clone(), dim=1)
        targets = torch.tensor([0])

        loss = head(emb, targets)
        self.assertFalse(torch.isnan(loss), 'Loss should not be NaN for extreme cosine values')
        self.assertFalse(torch.isinf(loss), 'Loss should not be Inf for extreme cosine values')


class TestStage2IpEmbeddingLoss(unittest.TestCase):

    def _make_loss(self, num_classes=10, embed_dim=16, K=2, lambda_sub=0.1):
        """Create a Stage2IpEmbeddingLoss using the real __init__ path."""
        os.environ['SUBCENTER_NUM_CLASSES'] = str(num_classes)
        os.environ['SUBCENTER_K'] = str(K)
        os.environ['SUBCENTER_SCALE'] = '32'
        os.environ['SUBCENTER_MARGIN'] = '0.2'
        os.environ['SUBCENTER_LAMBDA'] = str(lambda_sub)

        from swift.loss.embedding import Stage2IpEmbeddingLoss

        # Minimal mock model with config (hidden_size for embed_dim inference)
        model = nn.Linear(1, 1)  # dummy module to attach head to
        model.config = type('Config', (), {'hidden_size': embed_dim})()

        # Minimal mock trainer — Stage2IpEmbeddingLoss.__init__ accesses
        # trainer.model and trainer.__class__.__mro__ (via BaseLoss.__init__)
        class MockTrainer:
            pass
        trainer = MockTrainer()
        trainer.model = model

        loss_fn = Stage2IpEmbeddingLoss(args=None, trainer=trainer)

        return loss_fn, model

    def _cleanup_env(self):
        for key in ['SUBCENTER_NUM_CLASSES', 'SUBCENTER_K', 'SUBCENTER_SCALE',
                     'SUBCENTER_MARGIN', 'SUBCENTER_LAMBDA', 'SUBCENTER_EMBED_DIM']:
            os.environ.pop(key, None)

    def test_composition(self):
        """Combined loss = mp_loss + lambda * sub_loss, finite and positive."""
        loss_fn, _ = self._make_loss(num_classes=5, embed_dim=16, lambda_sub=0.1)

        D = 16
        # 1 group: anchor + 2 positives + 1 negative
        anchor = F.normalize(torch.randn(1, D), dim=1)
        pos = F.normalize(torch.randn(2, D), dim=1)
        neg = F.normalize(torch.randn(1, D), dim=1)
        sentences = torch.cat([anchor, pos, neg], dim=0)
        labels = torch.tensor([2.0, 1.0, 0.0])
        ip_ids = torch.tensor([3])

        loss = loss_fn({'last_hidden_state': sentences}, labels, ip_ids=ip_ids)

        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0)
        self._cleanup_env()

    def test_ip_id_minus_one_skipped(self):
        """Groups with ip_id=-1 should not contribute to sub-center loss."""
        loss_fn, _ = self._make_loss(num_classes=5, embed_dim=16, lambda_sub=0.5)

        D = 16
        # 2 groups: first has ip_id=2, second has ip_id=-1
        s1 = F.normalize(torch.randn(3, D), dim=1)  # anchor, pos, neg
        s2 = F.normalize(torch.randn(3, D), dim=1)  # anchor, pos, neg

        sentences = torch.cat([s1, s2], dim=0)
        labels = torch.tensor([2.0, 0.0, 2.0, 0.0])
        ip_ids_with_sentinel = torch.tensor([2, -1])

        loss_mixed = loss_fn({'last_hidden_state': sentences}, labels, ip_ids=ip_ids_with_sentinel)

        # All -1: sub-center loss should be zero, only mp_loss
        ip_ids_all_sentinel = torch.tensor([-1, -1])
        loss_all_sentinel = loss_fn({'last_hidden_state': sentences}, labels, ip_ids=ip_ids_all_sentinel)

        # Mixed should be higher than all-sentinel (sub-center adds to loss)
        self.assertGreater(loss_mixed.item(), loss_all_sentinel.item(),
                           'Sub-center loss from valid ip_ids should increase total loss')
        self._cleanup_env()

    def test_requires_ip_ids(self):
        """Should raise ValueError when ip_ids is not provided."""
        loss_fn, _ = self._make_loss(num_classes=5, embed_dim=16)

        D = 16
        sentences = F.normalize(torch.randn(3, D), dim=1)
        labels = torch.tensor([2.0, 0.0])

        with self.assertRaises(ValueError, msg='Should require ip_ids'):
            loss_fn({'last_hidden_state': sentences}, labels)
        self._cleanup_env()

    def test_proxy_in_model_params(self):
        """After loss init, proxy should appear in model.named_parameters() and state_dict()."""
        _, model = self._make_loss(num_classes=5, embed_dim=16)

        param_names = [name for name, _ in model.named_parameters()]
        self.assertTrue(
            any('ip_subcenter_head' in name and 'proxies' in name for name in param_names),
            f'Proxy not found in model params. Found: {param_names}')

        state_dict_keys = list(model.state_dict().keys())
        self.assertTrue(
            any('ip_subcenter_head' in key and 'proxies' in key for key in state_dict_keys),
            f'Proxy not found in state_dict. Keys: {state_dict_keys}')
        self._cleanup_env()

    def test_lambda_zero_equals_mp_only(self):
        """With lambda=0, stage2 loss should equal pure multi_positive_infonce loss."""
        loss_fn, _ = self._make_loss(num_classes=5, embed_dim=16, lambda_sub=0.0)

        D = 16
        torch.manual_seed(42)
        anchor = F.normalize(torch.randn(1, D), dim=1)
        pos = F.normalize(torch.randn(1, D), dim=1)
        neg = F.normalize(torch.randn(2, D), dim=1)
        sentences = torch.cat([anchor, pos, neg], dim=0)
        labels = torch.tensor([2.0, 0.0, 0.0])
        ip_ids = torch.tensor([1])

        combined_loss = loss_fn({'last_hidden_state': sentences}, labels, ip_ids=ip_ids)

        # Compute mp_loss independently
        mp_loss = loss_fn.mp_loss({'last_hidden_state': sentences}, labels)

        self.assertAlmostEqual(combined_loss.item(), mp_loss.item(), places=5,
                               msg='lambda=0 should make stage2 loss equal to mp_loss')
        self._cleanup_env()


if __name__ == '__main__':
    unittest.main()
