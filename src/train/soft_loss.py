"""Soft-label losses for noisy pairwise-preference supervision (docs §4 'Loss').

KL divergence against a target distribution; equivalent to soft cross-entropy
modulo the (constant w.r.t. parameters) target entropy term.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    """KL(soft_targets || softmax(logits)) averaged over the batch.

    Args:
        logits:       (batch, num_classes) raw model output
        soft_targets: (batch, num_classes) probability distribution; rows sum to 1
    """
    log_probs = F.log_softmax(logits, dim=-1)
    return F.kl_div(log_probs, soft_targets, reduction="batchmean")
