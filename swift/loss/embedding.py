# Copyright (c) ModelScope Contributors. All rights reserved.
import numpy as np
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate.utils import gather_object
from enum import Enum
from torch import nn
from torch.nn import MSELoss
from transformers.utils import strtobool

from swift.sequence_parallel import sequence_parallel
from swift.utils import get_dist_setting
from .base import BaseLoss


# Code borrowed from sentence_transformers
class SiameseDistanceMetric(Enum):
    """The metric for the contrastive loss"""

    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)  # noqa
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)  # noqa
    COSINE_DISTANCE = lambda x, y: 1 - F.cosine_similarity(x, y)  # noqa


def _parse_pair_sentence(outputs):
    if isinstance(outputs, dict):
        last_hidden_state = outputs['last_hidden_state']
    else:
        last_hidden_state = outputs
    batch_size = last_hidden_state.shape[0]
    shape_len = len(last_hidden_state.shape)
    first_sentence = list(range(0, batch_size, 2))
    second_sentence = list(range(1, batch_size, 2))
    if shape_len == 3:
        sentence1 = last_hidden_state[first_sentence][:, 0].squeeze(dim=1)
        sentence2 = last_hidden_state[second_sentence][:, 0].squeeze(dim=1)
    else:
        sentence1 = last_hidden_state[first_sentence]
        sentence2 = last_hidden_state[second_sentence]
    return sentence1, sentence2


class CosineSimilarityLoss(BaseLoss):

    def __call__(self, outputs, labels, **kwargs) -> torch.Tensor:
        # You need to return a scalar representing the loss.
        cos_score_transformation = nn.Identity()
        loss_fct = MSELoss()
        sentence1, sentence2 = _parse_pair_sentence(outputs)
        output = cos_score_transformation(torch.cosine_similarity(sentence1, sentence2))
        return loss_fct(output, labels.to(output.dtype).view(-1))


class ContrastiveLoss(BaseLoss):

    def __call__(self, outputs, labels, **kwargs) -> torch.Tensor:
        sentence1, sentence2 = _parse_pair_sentence(outputs)
        distance_metric = SiameseDistanceMetric.COSINE_DISTANCE
        distances = distance_metric(sentence1, sentence2)
        margin = 0.5
        labels = labels.to(sentence1.dtype)
        losses = 0.5 * (labels * distances.pow(2) + (1 - labels) * F.relu(margin - distances).pow(2))
        return losses.mean()


class OnlineContrastiveLoss(BaseLoss):

    def __call__(self, outputs, labels, **kwargs) -> torch.Tensor:
        sentence1, sentence2 = _parse_pair_sentence(outputs)
        distance_metric = SiameseDistanceMetric.COSINE_DISTANCE
        distance_matrix = distance_metric(sentence1, sentence2)
        negs = distance_matrix[labels == 0]
        poss = distance_matrix[labels == 1]

        # select hard positive and hard negative pairs
        negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
        positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]

        positive_loss = positive_pairs.pow(2).sum()
        margin = 0.5
        negative_loss = F.relu(margin - negative_pairs).pow(2).sum()
        loss = positive_loss + negative_loss
        return loss


def _parse_multi_negative_sentences(sentences, labels, hard_negatives=None):
    split_indices = torch.nonzero(labels, as_tuple=False).squeeze().tolist()
    if isinstance(split_indices, int):
        split_indices = [split_indices]
    split_indices.append(len(labels))
    split_indices = np.array(split_indices) + np.array(list(range(len(split_indices))))
    split_tensors = []

    for i in range(len(split_indices) - 1):
        start = split_indices[i]
        end = split_indices[i + 1]
        split_part = sentences[start:end]
        if hard_negatives is not None:
            negatives = len(split_part) - 2
            assert negatives > 0
            if negatives > hard_negatives:
                split_part = split_part[:hard_negatives + 2]
            elif negatives < hard_negatives:
                selected = np.random.choice(list(range(negatives)), size=hard_negatives - negatives, replace=True)
                selected += 1  # skip positive
                split_part = torch.cat((split_part, split_part[selected]), dim=0)
        split_tensors.append(split_part)
    return split_tensors


class InfonceLoss(BaseLoss):

    def __call__(self, outputs, labels, **kwargs) -> torch.Tensor:
        temperature = float(os.environ.get('INFONCE_TEMPERATURE', '0.1'))  # temperature
        # calculate CE across the batch, meaning all samples will be negative except the matching positive
        use_batch = strtobool(os.environ.get('INFONCE_USE_BATCH', 'True'))
        hard_negatives = os.environ.get('INFONCE_HARD_NEGATIVES', None)  # how many negative prompts kept in one sample
        # mask out fake negatives
        infonce_mask_fake_negative = strtobool(os.environ.get('INFONCE_MASK_FAKE_NEGATIVE', 'False'))
        fake_neg_margin = float(os.environ.get('INFONCE_FAKE_NEG_MARGIN', '0.1'))
        # enhanced components to align with Qwen3-Embedding denominator; controlled individually
        # defaults set to False for backward compatibility
        infonce_include_qq = strtobool(os.environ.get('INFONCE_INCLUDE_QQ', 'False'))
        infonce_include_dd = strtobool(os.environ.get('INFONCE_INCLUDE_DD', 'False'))
        if hard_negatives is not None:
            hard_negatives = int(hard_negatives)
        if self.is_megatron:
            from megatron.core import mpu
            rank, world_size = mpu.get_data_parallel_rank(), mpu.get_data_parallel_world_size()
        else:
            rank, _, world_size, _ = get_dist_setting()
        # repeat of anchor(1)+positive(1)+negatives(n)
        sentences = outputs['last_hidden_state']

        if world_size > 1 and use_batch:
            if getattr(sequence_parallel, 'dp_group', None) is not None:
                all_sentences = sequence_parallel._gather_object_dp(sentences.unsqueeze(0))
                labels = sequence_parallel._gather_object_dp(labels)
                rank = sequence_parallel.dp_rank
            elif self.is_megatron:
                from megatron.core import mpu
                dp_group = mpu.get_data_parallel_group()
                shapes = [sentences.new_empty((2, ), dtype=torch.long) for _ in range(world_size)]
                dist.all_gather(
                    shapes,
                    sentences.new_tensor(sentences.shape, dtype=torch.long),
                    group=dp_group,
                )
                all_sentences = [sentences.new_empty(shape.tolist()) for shape in shapes]
                dist.all_gather(
                    all_sentences,
                    sentences,
                    group=dp_group,
                )
            else:
                # gather all the sentences and labels across the gpus when calculate loss across all batches of all gpus
                all_sentences = gather_object(sentences.unsqueeze(0))
                labels = gather_object(labels)
            # override the gathered one
            all_sentences[rank] = sentences
            for idx in range(len(all_sentences)):
                if idx == rank:
                    continue
                # we don't calculate grad from other gpus
                all_sentences[idx] = all_sentences[idx].detach().to(sentences.device)
            sentences = torch.cat(all_sentences, dim=0)
            labels = [tensor.to(sentences.device) for tensor in labels]
            labels = torch.stack(labels, dim=0)

        # split tensors into single sample
        # for example: batch_size=2 with tensor anchor(1)+positive(1)+negatives(3) + anchor(1)+positive(1)+negatives(2)
        # labels will be [1,0,0,0,1,0,0], meaning 1 positive, 3 negatives, 1 positive, 2 negatives
        split_tensors = _parse_multi_negative_sentences(sentences, labels, hard_negatives)
        loss = 0
        can_batched = hard_negatives is not None
        if hard_negatives is None and len(set([s.shape[0] for s in split_tensors])) == 1:
            # all tensors have the same batch size
            can_batched = True
        if not use_batch:
            # only calculate loss inside one sample
            if can_batched:
                # negative numbers are equal
                # [B, neg+2, D]
                sentences = torch.stack(split_tensors, dim=0)
                # [B, 1, D] * [B, neg+1, D]
                similarity_matrix = torch.matmul(sentences[:, 0:1], sentences[:, 1:].transpose(1, 2)) / temperature
                # The positive one is the first element
                labels = torch.zeros(len(split_tensors), dtype=torch.int64).to(sentences.device)
                loss = nn.CrossEntropyLoss()(similarity_matrix.squeeze(1), labels)
            else:
                # the negative numbers may be different, use for loop
                for tensor in split_tensors:
                    # [D] * [neg+1, D]
                    similarity_matrix = torch.matmul(tensor[0], tensor[1:].T) / temperature
                    # The positive one is the first element
                    labels = torch.tensor(0).to(tensor.device)
                    loss += nn.CrossEntropyLoss()(similarity_matrix, labels)
                # avg between all batches in one gpu
                loss /= len(split_tensors)
        else:
            if can_batched:
                # [B, neg+2, D]
                sentences = torch.stack(split_tensors, dim=0)
                # base q->d similarities (includes own positive and all in-batch documents)
                queries = sentences[:, 0].squeeze(1)  # [B, D]
                docs_all = sentences[:, 1:].reshape(-1, sentences.size(2))  # [B*(neg+1), D]
                qd_matrix = torch.matmul(queries, docs_all.T)  # [B, B*(neg+1)]
                # target indices: start of each group's document block (its positive)
                labels = torch.tensor(range(0,
                                            sentences.size(0) * (sentences.size(1) - 1),
                                            sentences.size(1) - 1)).view(-1).to(sentences.device)

                logits_list = [qd_matrix]

                if infonce_include_qq:
                    # q->q similarities; exclude self via -inf on diagonal to avoid accidental positives
                    qq_matrix = torch.matmul(queries, queries.T)  # [B, B]
                    qq_matrix = qq_matrix.clone()
                    qq_matrix.fill_diagonal_(float('-inf'))
                    logits_list.append(qq_matrix)

                if infonce_include_dd:
                    # d+ -> d (doc-doc) similarities; exclude self-positive column per row
                    pos_docs = sentences[:, 1].squeeze(1)  # [B, D]
                    dd_matrix = torch.matmul(pos_docs, docs_all.T)  # [B, B*(neg+1)]
                    # mask self positive per row: column index = row_idx * (neg+1)
                    block = sentences.size(1) - 1  # (neg+1)
                    if block > 0:
                        row_idx = torch.arange(dd_matrix.size(0), device=dd_matrix.device)
                        col_idx = row_idx * block
                        dd_matrix[row_idx, col_idx] = float('-inf')
                    logits_list.append(dd_matrix)

                if infonce_mask_fake_negative:
                    # thresholds derived from positive q->d scores per row
                    row_idx = torch.arange(qd_matrix.size(0), device=qd_matrix.device)
                    pos_scores = qd_matrix[row_idx, labels]
                    thresholds = pos_scores.view(-1, 1).detach() + fake_neg_margin

                    # qd block mask
                    qd_block = qd_matrix.clone()
                    qd_mask = qd_block > thresholds
                    qd_block[qd_mask] = float('-inf')

                    components = [qd_block]

                    # qq block mask (if present)
                    if infonce_include_qq:
                        qq_block = qq_matrix.clone()
                        qq_mask = qq_block > thresholds
                        qq_block[qq_mask] = float('-inf')
                        # diagonal already masked unconditionally at construction time
                        components.append(qq_block)

                    # dd block (if present): self-positive column already masked unconditionally
                    if infonce_include_dd:
                        # align with Qwen3-Embedding, no threshold masking for d-d
                        components.append(dd_matrix)

                    similarity_matrix = torch.cat(components, dim=1)
                else:
                    # concatenate all components without masking
                    similarity_matrix = torch.cat(logits_list, dim=1)
                # temperature scaling and CE
                similarity_matrix = similarity_matrix / temperature
                loss = nn.CrossEntropyLoss()(similarity_matrix, labels)
            else:
                all_tensors = []
                for tensor in split_tensors:
                    all_tensors.append(tensor[1:])
                # cat all neg+1 tensors
                sentences = torch.cat(all_tensors, dim=0)
                # prepare query anchors list if q-q is included
                if infonce_include_qq:
                    queries_all = torch.stack([t[0] for t in split_tensors], dim=0)  # [B, D]
                length = 0
                for idx, tensor in enumerate(split_tensors):
                    # [D] * [B*(neg+1), D], neg numbers are different
                    qd_vec = torch.matmul(tensor[0], sentences.T)
                    target = torch.tensor(length).to(tensor.device)
                    logits_parts = []

                    # compute threshold from positive q->d score
                    threshold = (qd_vec[target].detach() + fake_neg_margin)

                    # qd part with masking
                    if infonce_mask_fake_negative:
                        qd_masked = torch.where(qd_vec > threshold, torch.tensor(float('-inf'), device=qd_vec.device),
                                                qd_vec)
                    else:
                        qd_masked = qd_vec
                    logits_parts.append(qd_masked)

                    # qq part
                    if infonce_include_qq:
                        qq_vec = torch.matmul(tensor[0], queries_all.T)  # [B]
                        # exclude self
                        qq_vec = qq_vec.clone()
                        qq_vec[idx] = float('-inf')
                        if infonce_mask_fake_negative:
                            qq_vec = torch.where(qq_vec > threshold, torch.tensor(float('-inf'), device=qq_vec.device),
                                                 qq_vec)
                        logits_parts.append(qq_vec)

                    # dd part
                    if infonce_include_dd:
                        dd_vec = torch.matmul(tensor[1], sentences.T)  # [B*(neg+1)]
                        # mask self positive column for this row only (no threshold masking for d-d)
                        block = split_tensors[idx].size(0) - 1  # (neg+1) for this group
                        dd_vec[length] = float('-inf')
                        logits_parts.append(dd_vec)

                    logits_row = torch.cat(logits_parts, dim=-1)
                    logits_row = logits_row / temperature
                    loss += nn.CrossEntropyLoss()(logits_row.unsqueeze(0), target.unsqueeze(0))
                    # next positive is neg+1
                    length += tensor.size(0) - 1
                loss /= len(split_tensors)
        return loss


def _parse_multi_positive_sentences(sentences, labels):
    """Parse sentence groups where each group has variable positives and negatives.

    Label encoding: 2.0 = group boundary (first positive), 1.0 = additional positive, 0.0 = negative.
    Sentence ordering per group: [anchor, pos0, pos1, ..., neg0, neg1, ...]

    Returns:
        List of (anchor, positives_tensor, negatives_tensor) tuples.
    """
    boundary_indices = torch.nonzero(labels == 2.0, as_tuple=False).squeeze(-1).tolist()
    if isinstance(boundary_indices, int):
        boundary_indices = [boundary_indices]

    if len(boundary_indices) == 0:
        raise ValueError(
            'No group boundary (label==2.0) found in labels. '
            'multi_positive_infonce requires labels with 2.0 markers. '
            f'Got labels: {labels.tolist()}')

    valid_values = {0.0, 1.0, 2.0}
    unique_values = set(labels.tolist())
    if not unique_values <= valid_values:
        raise ValueError(
            f'Labels contain invalid values: {unique_values - valid_values}. '
            f'Expected only values in {{0.0, 1.0, 2.0}}.')

    num_groups = len(boundary_indices)
    expected_sentences = len(labels) + num_groups  # each group has 1 unlabeled anchor
    if sentences.shape[0] != expected_sentences:
        raise ValueError(
            f'Sentence count ({sentences.shape[0]}) does not match '
            f'expected count ({expected_sentences}) from labels. '
            f'Expected len(labels)={len(labels)} + num_groups={num_groups}.')

    boundary_indices.append(len(labels))

    groups = []
    for i in range(len(boundary_indices) - 1):
        label_start = boundary_indices[i]
        label_end = boundary_indices[i + 1]
        group_labels = labels[label_start:label_end]

        # Each label corresponds to a positive/negative after the anchor.
        # Anchors are interleaved: the i-th anchor is at sentence index (label_start + i).
        sent_start = label_start + i
        sent_end = label_end + i + 1  # +1 for this group's anchor
        group_sentences = sentences[sent_start:sent_end]

        anchor = group_sentences[0]
        num_positives = int((group_labels >= 1.0).sum().item())
        positives = group_sentences[1:1 + num_positives]
        negatives = group_sentences[1 + num_positives:]

        groups.append((anchor, positives, negatives))

    return groups


class MultiPositiveInfonceLoss(BaseLoss):
    """Multi-positive InfoNCE loss for embedding training.

    Supports variable numbers of positives per anchor. Row-local only (no batch-global).

    Loss = -log( sum_p exp(sim(q,p)/tau) / (sum_p exp(sim(q,p)/tau) + sum_n exp(sim(q,n)/tau)) )

    With 1 positive, this reduces exactly to standard InfoNCE.
    """

    def __call__(self, outputs, labels, **kwargs) -> torch.Tensor:
        temperature = float(os.environ.get('INFONCE_TEMPERATURE', '0.1'))
        sentences = outputs['last_hidden_state']

        groups = _parse_multi_positive_sentences(sentences, labels)

        loss = torch.tensor(0.0, device=sentences.device, dtype=sentences.dtype)
        for anchor, positives, negatives in groups:
            # anchor: [D], positives: [P, D], negatives: [N, D]
            pos_sim = torch.matmul(positives, anchor) / temperature  # [P]
            neg_sim = torch.matmul(negatives, anchor) / temperature  # [N]

            log_sum_pos = torch.logsumexp(pos_sim, dim=0)
            all_sim = torch.cat([pos_sim, neg_sim], dim=0)  # [P+N]
            log_sum_all = torch.logsumexp(all_sim, dim=0)

            # -log(sum_pos / sum_all) = -(log_sum_pos - log_sum_all)
            loss += -(log_sum_pos - log_sum_all)

        loss /= len(groups)
        return loss


class SubcenterArcFaceHead(nn.Module):
    """Sub-center ArcFace classification head with K sub-centers per class.

    Each class has K learnable proxy vectors. Cosine similarity is computed
    between input embeddings and all proxies, then max-pooled over K to get
    the class logit. An additive angular margin is applied to the target class
    logit before scaling and CrossEntropyLoss.

    Reference: Sub-center ArcFace (ECCV 2020, Deng et al.)
    """

    def __init__(self, num_classes: int, embed_dim: int, K: int = 5, scale: float = 64.0, margin: float = 0.2):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.K = K
        self.scale = scale
        self.margin = margin
        self.eps = 1e-7

        self.proxies = nn.Parameter(torch.empty(num_classes, K, embed_dim))
        nn.init.xavier_uniform_(self.proxies.view(num_classes * K, embed_dim))
        self.proxies.data = self.proxies.data.view(num_classes, K, embed_dim)

        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Sub-center ArcFace loss.

        Args:
            embeddings: [N, D] L2-normalized feature vectors.
            targets: [N] integer class labels.

        Returns:
            Scalar loss tensor.
        """
        # L2-normalize features and proxies
        x = F.normalize(embeddings, p=2, dim=1)  # [N, D]
        W = F.normalize(self.proxies, p=2, dim=2)  # [C, K, D]

        # Cosine similarity: [N, C, K]
        cosine = torch.einsum('nd,ckd->nck', x, W)

        # Max-pool over K sub-centers: [N, C]
        cosine, _ = cosine.max(dim=2)

        # Extract target cosine and apply angular margin
        N = cosine.size(0)
        target_cos = cosine[torch.arange(N, device=cosine.device), targets]
        target_cos = torch.clamp(target_cos, -1.0 + self.eps, 1.0 - self.eps)
        theta = torch.acos(target_cos)
        target_logit = torch.cos(theta + self.margin)

        # Replace target class logit with margin-applied version
        cosine[torch.arange(N, device=cosine.device), targets] = target_logit

        # Scale and compute CE loss
        logits = cosine * self.scale
        return self.ce_loss(logits, targets)


class Stage2IpEmbeddingLoss(BaseLoss):
    """Combined loss for Stage 2 IP embedding training.

    L = L_multi_positive_infonce + lambda_sub * L_subcenter_arcface

    The multi-positive InfoNCE component handles retrieval learning.
    The Sub-center ArcFace component provides geometric class structure.

    Requires ip_ids in kwargs (from Patch 2 metadata plumbing).

    Environment variables:
        SUBCENTER_NUM_CLASSES: Required. Number of IP classes.
        SUBCENTER_K: Sub-centers per class (default: 5).
        SUBCENTER_SCALE: ArcFace scale factor (default: 64).
        SUBCENTER_MARGIN: ArcFace angular margin in radians (default: 0.2).
        SUBCENTER_LAMBDA: Weight for sub-center loss (default: 0.1).
        SUBCENTER_EMBED_DIM: Fallback embedding dimension if config inference fails.
    """

    def __init__(self, args: 'TrainingArguments', trainer: 'Trainer'):
        super().__init__(args, trainer)

        num_classes_str = os.environ.get('SUBCENTER_NUM_CLASSES')
        if num_classes_str is None:
            raise ValueError(
                'SUBCENTER_NUM_CLASSES environment variable is required for stage2_ip_embedding. '
                'Set it to the number of IP classes in your training data. '
                'This value is in <output>.ip_id_map.json from the data construction script.')
        num_classes = int(num_classes_str)

        K = int(os.environ.get('SUBCENTER_K', '5'))
        scale = float(os.environ.get('SUBCENTER_SCALE', '64'))
        margin = float(os.environ.get('SUBCENTER_MARGIN', '0.2'))
        self.lambda_sub = float(os.environ.get('SUBCENTER_LAMBDA', '0.1'))

        # Infer embedding dimension
        embed_dim = None
        try:
            from swift.utils import HfConfigFactory
            embed_dim = HfConfigFactory.get_config_attr(trainer.model.config, 'hidden_size')
        except Exception:
            pass
        if embed_dim is None:
            embed_dim_str = os.environ.get('SUBCENTER_EMBED_DIM')
            if embed_dim_str is not None:
                embed_dim = int(embed_dim_str)
        if embed_dim is None:
            raise ValueError(
                'Cannot infer embedding dimension from model config. '
                'Set SUBCENTER_EMBED_DIM environment variable.')

        self.mp_loss = MultiPositiveInfonceLoss(args, trainer)
        self.head = SubcenterArcFaceHead(num_classes, embed_dim, K, scale, margin)

        # Register head on model so its parameters are picked up by
        # optimizer (created later), DDP gradient sync, and checkpoint saving.
        trainer.model.ip_subcenter_head = self.head

    def __call__(self, outputs, labels, **kwargs) -> torch.Tensor:
        # Multi-positive InfoNCE component
        mp_loss = self.mp_loss(outputs, labels, **kwargs)

        # Sub-center ArcFace component
        ip_ids = kwargs.get('ip_ids')
        if ip_ids is None:
            raise ValueError(
                'stage2_ip_embedding requires ip_ids in training data. '
                'Use --include_ip_id when building data, or set ip_id in JSONL rows.')

        sentences = outputs['last_hidden_state']
        groups = _parse_multi_positive_sentences(sentences, labels)

        if len(ip_ids) != len(groups):
            raise ValueError(
                f'ip_ids length ({len(ip_ids)}) != number of groups ({len(groups)}). '
                f'Each anchor row must have exactly one ip_id.')

        # Collect embeddings with known class labels (anchor + positives only)
        class_embeddings = []
        class_targets = []
        for i, (anchor, positives, _negatives) in enumerate(groups):
            ip_id = ip_ids[i].item()
            if ip_id == -1:
                continue  # old-format row without ip_id
            if ip_id < 0 or ip_id >= self.head.num_classes:
                raise ValueError(
                    f'ip_id={ip_id} out of range [0, {self.head.num_classes}). '
                    f'Check SUBCENTER_NUM_CLASSES matches your ip_id_map.json.')
            # anchor and all positives belong to this ip_id
            group_embs = torch.cat([anchor.unsqueeze(0), positives], dim=0)  # [1+P, D]
            class_embeddings.append(group_embs)
            class_targets.extend([ip_id] * group_embs.size(0))

        if len(class_embeddings) == 0:
            # All groups have ip_id == -1; sub-center loss is undefined
            return mp_loss

        class_embeddings = torch.cat(class_embeddings, dim=0)  # [M, D]
        class_targets = torch.tensor(class_targets, dtype=torch.long, device=class_embeddings.device)

        sub_loss = self.head(class_embeddings, class_targets)

        return mp_loss + self.lambda_sub * sub_loss
