"""Multi-head self-attention block with pre-LayerNorm and residual connection.

Returns attention weights alongside the output for use in explainability
and attention rollout visualization.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AttentionBlock(nn.Module):
    """Multi-head self-attention with pre-LayerNorm residual connection.

    Uses the pre-norm formulation (LayerNorm before attention) which is
    more training-stable than post-norm. Returns attention weights for
    explainability.

    Architecture:
        x → LayerNorm → MultiHeadAttention → + residual → output
        (attention_weights are captured and returned)

    Args:
        embed_dim: Dimension of the input and output features.
        num_heads: Number of attention heads.
        dropout: Dropout rate for attention weights and output.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_dropout = nn.Dropout(dropout)

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"AttentionBlock: dim={embed_dim}, heads={num_heads}, "
            f"{total_params / 1e3:.1f}K params"
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with attention weight capture.

        Args:
            x: Input tensor of shape (B, L, embed_dim).
            key_padding_mask: Optional boolean mask of shape (B, L).
                True values indicate positions that should be ignored
                (padding positions). Note: this is the inverse of the
                attention_mask from the dataset (where 1 = real token).

        Returns:
            Tuple of:
                - output: Tensor of shape (B, L, embed_dim) after
                  attention + residual connection.
                - attention_weights: Tensor of shape (B, num_heads, L, L)
                  containing the attention probability distributions.
        """
        residual = x

        # Pre-LayerNorm
        x_norm = self.layer_norm(x)

        # Multi-head self-attention
        attn_output, attention_weights = self.attention(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,  # Return per-head weights
        )

        # Dropout + residual
        attn_output = self.output_dropout(attn_output)
        output = attn_output + residual

        return output, attention_weights

    def get_output_dim(self) -> int:
        """Return the output feature dimension.

        Returns:
            The embedding dimension (same as input).
        """
        return self.embed_dim


class FeedForwardBlock(nn.Module):
    """Position-wise feed-forward block with pre-LayerNorm and residual.

    Architecture:
        x → LayerNorm → Linear → ReLU → Dropout → Linear → Dropout → + residual

    Args:
        embed_dim: Input and output dimension.
        hidden_dim: Dimension of the intermediate layer.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"FeedForwardBlock: {embed_dim} → {hidden_dim} → {embed_dim}, "
            f"{total_params / 1e3:.1f}K params"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor of shape (B, L, embed_dim).

        Returns:
            Output tensor of shape (B, L, embed_dim).
        """
        residual = x
        x = self.layer_norm(x)
        x = self.feed_forward(x)
        return x + residual

    def get_output_dim(self) -> int:
        """Return the output feature dimension.

        Returns:
            The embedding dimension (same as input).
        """
        return self.feed_forward[-2].out_features
