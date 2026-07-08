"""Loss functions for ProtIntel training.

Implements label-smoothing cross-entropy and focal loss with
per-class weights and padding masking.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logger import get_logger

logger = get_logger(__name__)


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing and optional class weights.

    Distributes a fraction of the probability mass uniformly across
    all classes to prevent overconfident predictions and improve
    generalization.

    Args:
        smoothing: Label smoothing factor in [0, 1). A value of 0
            gives standard cross-entropy.
        weight: Optional per-class weight tensor of shape (num_classes,).
        ignore_index: Label index to ignore (for padding positions).
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.register_buffer("weight", weight)
        logger.info(
            f"LabelSmoothingCE: smoothing={smoothing}, "
            f"weighted={weight is not None}"
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute label-smoothing cross-entropy loss.

        Args:
            logits: Predicted logits of shape (B, L, C) or (N, C).
            targets: Ground-truth labels of shape (B, L) or (N,).

        Returns:
            Scalar loss value.
        """
        # Flatten if needed
        if logits.dim() == 3:
            batch_size, seq_len, num_classes = logits.shape
            logits = logits.reshape(-1, num_classes)
            targets = targets.reshape(-1)
        else:
            num_classes = logits.size(-1)

        # Create mask for valid (non-padding) positions
        valid_mask = targets != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        logits = logits[valid_mask]
        targets = targets[valid_mask]

        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Smooth targets
        with torch.no_grad():
            smooth_targets = torch.full_like(
                log_probs, self.smoothing / (num_classes - 1)
            )
            smooth_targets.scatter_(
                1, targets.unsqueeze(1), 1.0 - self.smoothing
            )

        # Weighted loss
        loss = -(smooth_targets * log_probs)

        if self.weight is not None:
            weight = self.weight.to(logits.device)
            per_sample_weight = weight[targets]
            loss = loss.sum(dim=-1) * per_sample_weight
        else:
            loss = loss.sum(dim=-1)

        return loss.mean()


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance.

    Down-weights well-classified examples and focuses learning on
    hard, misclassified examples. Especially useful for rare Q8
    classes (G, I, B).

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Args:
        gamma: Focusing parameter. Higher values increase focus on
            hard examples. Default 2.0.
        alpha: Optional per-class weight tensor of shape (num_classes,).
            If None, no class weighting is applied.
        ignore_index: Label index to ignore (for padding).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.register_buffer("alpha", alpha)
        logger.info(
            f"FocalLoss: gamma={gamma}, weighted={alpha is not None}"
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Predicted logits of shape (B, L, C) or (N, C).
            targets: Ground-truth labels of shape (B, L) or (N,).

        Returns:
            Scalar loss value.
        """
        if logits.dim() == 3:
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)

        # Mask padding
        valid_mask = targets != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        logits = logits[valid_mask]
        targets = targets[valid_mask]

        # Compute probabilities
        probs = F.softmax(logits, dim=-1)
        targets_one_hot = F.one_hot(
            targets, num_classes=logits.size(-1)
        ).float()

        # Gather probabilities for target classes
        p_t = (probs * targets_one_hot).sum(dim=-1)  # (N,)
        p_t = p_t.clamp(min=1e-8)  # Numerical stability

        # Focal term
        focal_weight = (1.0 - p_t) ** self.gamma

        # Log probability
        log_p_t = torch.log(p_t)

        # Apply per-class alpha if provided
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha[targets]
            loss = -alpha_t * focal_weight * log_p_t
        else:
            loss = -focal_weight * log_p_t

        return loss.mean()


def create_loss_function(
    loss_type: str = "cross_entropy",
    label_smoothing: float = 0.1,
    class_weights: Optional[torch.Tensor] = None,
    focal_gamma: float = 2.0,
    ignore_index: int = -100,
) -> nn.Module:
    """Factory function to create the appropriate loss function.

    Args:
        loss_type: Type of loss ('cross_entropy' or 'focal').
        label_smoothing: Smoothing factor for cross-entropy.
        class_weights: Optional per-class weight tensor.
        focal_gamma: Gamma parameter for focal loss.
        ignore_index: Index to ignore in loss computation.

    Returns:
        Configured loss function module.

    Raises:
        ValueError: If an unknown loss_type is specified.
    """
    if loss_type == "cross_entropy":
        return LabelSmoothingCrossEntropy(
            smoothing=label_smoothing,
            weight=class_weights,
            ignore_index=ignore_index,
        )
    elif loss_type == "focal":
        return FocalLoss(
            gamma=focal_gamma,
            alpha=class_weights,
            ignore_index=ignore_index,
        )
    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. "
            f"Supported: 'cross_entropy', 'focal'"
        )
