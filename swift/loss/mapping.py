# Copyright (c) ModelScope Contributors. All rights reserved.
from .causal_lm import CustomCrossEntropyLoss
from .embedding import (ContrastiveLoss, CosineSimilarityLoss, InfonceLoss, MultiPositiveInfonceLoss,
                        OnlineContrastiveLoss, Stage2IpEmbeddingLoss)
from .reranker import ListwiseRerankerLoss, PointwiseRerankerLoss

loss_map = {
    'cross_entropy': CustomCrossEntropyLoss,  # examples
    # embedding
    'cosine_similarity': CosineSimilarityLoss,
    'contrastive': ContrastiveLoss,
    'online_contrastive': OnlineContrastiveLoss,
    'infonce': InfonceLoss,
    'multi_positive_infonce': MultiPositiveInfonceLoss,
    'stage2_ip_embedding': Stage2IpEmbeddingLoss,
    # # reranker
    'pointwise_reranker': PointwiseRerankerLoss,
    'listwise_reranker': ListwiseRerankerLoss,
}
