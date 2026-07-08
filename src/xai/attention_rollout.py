"""Attention rollout for aggregating attention across layers.

Implements the attention rollout algorithm to produce a single
attention matrix summarizing how information flows through the
attention mechanism.
"""

from __future__ import annotations

import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_attention_rollout(
    attention_weights: torch.Tensor,
    head_fusion: str = "mean",
    discard_ratio: float = 0.0,
) -> torch.Tensor:
    """Compute attention rollout from multi-head attention weights.

    The rollout algorithm multiplies attention matrices across layers
    to trace how attention flows from input to output positions.
    Since ProtIntel has a single attention layer, this primarily
    aggregates across attention heads.

    Args:
        attention_weights: Attention weight tensor of shape
            (B, num_heads, L, L) from the attention block.
        head_fusion: How to fuse attention heads. One of:
            - 'mean': Average across heads.
            - 'max': Take maximum across heads.
            - 'min': Take minimum across heads.
        discard_ratio: Fraction of lowest attention values to zero
            out before rollout (for sparsification).

    Returns:
        Aggregated attention matrix of shape (B, L, L) where each
        row represents the attention distribution from that position
        to all other positions.
    """
    batch_size, num_heads, seq_len, _ = attention_weights.shape

    # Fuse across heads
    if head_fusion == "mean":
        fused = attention_weights.mean(dim=1)  # (B, L, L)
    elif head_fusion == "max":
        fused = attention_weights.max(dim=1).values
    elif head_fusion == "min":
        fused = attention_weights.min(dim=1).values
    else:
        raise ValueError(
            f"Unknown head_fusion: {head_fusion}. "
            f"Supported: 'mean', 'max', 'min'"
        )

    # Apply discard ratio (sparsification)
    if discard_ratio > 0:
        flat = fused.reshape(batch_size, -1)
        num_discard = int(flat.size(1) * discard_ratio)
        if num_discard > 0:
            threshold = flat.kthvalue(num_discard + 1, dim=1).values
            threshold = threshold.unsqueeze(1).unsqueeze(1)
            fused = fused * (fused >= threshold).float()
            # Re-normalize rows
            row_sums = fused.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            fused = fused / row_sums

    # Add identity matrix for residual connection
    identity = torch.eye(seq_len, device=fused.device).unsqueeze(0)
    rollout = 0.5 * fused + 0.5 * identity  # Account for residual

    # Normalize rows to sum to 1
    row_sums = rollout.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    rollout = rollout / row_sums

    return rollout


def extract_residue_attention(
    rollout: torch.Tensor,
    residue_idx: int,
) -> torch.Tensor:
    """Extract attention scores for a specific residue.

    Args:
        rollout: Attention rollout matrix of shape (B, L, L) or (L, L).
        residue_idx: Index of the residue to extract attention for.

    Returns:
        Attention vector of shape (L,) or (B, L) showing how much
        attention the specified residue pays to each other position.
    """
    if rollout.dim() == 3:
        return rollout[:, residue_idx, :]
    return rollout[residue_idx, :]


def compute_attention_entropy(
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """Compute the entropy of attention distributions per position.

    High entropy indicates diffuse attention (model is uncertain),
    while low entropy indicates focused attention (model is confident
    about which positions are important).

    Args:
        attention_weights: Attention weights of shape
            (B, num_heads, L, L) or (B, L, L).

    Returns:
        Entropy per position of shape (B, L).
    """
    if attention_weights.dim() == 4:
        # Average across heads first
        attn = attention_weights.mean(dim=1)  # (B, L, L)
    else:
        attn = attention_weights

    # Clamp for numerical stability
    attn = attn.clamp(min=1e-12)

    # Entropy: -sum(p * log(p))
    entropy = -(attn * attn.log()).sum(dim=-1)  # (B, L)

    return entropy
