"""Soft-label losses for noisy pairwise-preference supervision (docs §4 'Loss').

KL divergence against a target distribution; equivalent to soft cross-entropy
modulo the (constant w.r.t. parameters) target entropy term.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_cross_entropy(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """KL(soft_targets || softmax(logits)) averaged over the batch.

    Args:
        logits:         (batch, num_classes) raw model output
        soft_targets:   (batch, num_classes) probability distribution; rows sum to 1
        sample_weights: optional (batch,) per-sample weights (e.g. inverse class
                        frequency indexed by label_id). When provided, the
                        per-sample KL is reweighted before averaging.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    if sample_weights is None:
        return F.kl_div(log_probs, soft_targets, reduction="batchmean")
    per_sample_kl = F.kl_div(log_probs, soft_targets, reduction="none").sum(dim=-1)
    weight_sum = sample_weights.sum().clamp_min(1e-8)
    return (per_sample_kl * sample_weights).sum() / weight_sum
