"""Conformer Encoder (Convolution + Transformer Hybrid) for ProtIntel.

Combines Depthwise Separable Convolutions with Multi-Head Self-Attention in a Macaron-style
feed-forward structure to capture local sequence motifs and long-range contacts simultaneously.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise Separable 1D Convolution with BatchNorm and Swish (SiLU)."""

    def __init__(self, channels: int, kernel_size: int = 31, dropout: float = 0.1) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, padding=padding, groups=channels
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, L)
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        out = self.act(out)
        out = self.dropout(out)
        return out


class ConformerBlock(nn.Module):
    """Conformer block combining Feed-Forward, MHSA, Conv, and Feed-Forward modules."""

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        kernel_size: int = 31,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

        self.norm_mha = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.norm_conv = nn.LayerNorm(embed_dim)
        self.conv = DepthwiseSeparableConv1d(channels=embed_dim, kernel_size=kernel_size, dropout=dropout)

        self.ffn2 = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Half-step FFN 1
        x = x + 0.5 * self.ffn1(x)

        # MHSA
        mha_in = self.norm_mha(x)
        attn_out, _ = self.mha(mha_in, mha_in, mha_in, key_padding_mask=key_padding_mask)
        x = x + attn_out

        # Conv module
        conv_in = self.norm_conv(x).transpose(1, 2)  # (B, D, L)
        conv_out = self.conv(conv_in).transpose(1, 2)  # (B, L, D)
        x = x + conv_out

        # Half-step FFN 2
        x = x + 0.5 * self.ffn2(x)
        return self.final_norm(x)


class ConformerEncoder(nn.Module):
    """Stack of Conformer Blocks.

    Args:
        input_dim: ESM-2 input dimension (1280).
        embed_dim: Internal conformer dimension (512).
        num_blocks: Number of stacked conformer blocks.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int = 1280,
        embed_dim: int = 512,
        num_blocks: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim) if input_dim != embed_dim else nn.Identity()
        self.blocks = nn.ModuleList([
            ConformerBlock(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_blocks)
        ])

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        out = self.input_proj(x)
        for block in self.blocks:
            out = block(out, key_padding_mask=key_padding_mask)
        return out
