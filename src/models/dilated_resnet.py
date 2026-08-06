"""Dilated SE-ResNet Encoder for Protein Secondary Structure Prediction.

Combines exponential dilation rates (1, 2, 4, 8) to expand effective receptive field
across long-range residue contacts with Squeeze-and-Excitation (SE) channel gating.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SqueezeExcitation1D(nn.Module):
    """Squeeze-and-Excitation block for 1D sequence feature channels."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced_dim = max(8, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced_dim),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, L)
        w = self.fc(x).unsqueeze(-1)  # (B, C, 1)
        return x * w


class DilatedSEResBlock(nn.Module):
    """Residual block with dilated 1D convolution and SE channel attention."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.se = SqueezeExcitation1D(channels)
        self.dropout = nn.Dropout(p=dropout)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.act2(out + residual)
        return out


class DilatedSEResNetEncoder(nn.Module):
    """Multi-block Dilated SE-ResNet sequence encoder.

    Args:
        input_dim: ESM-2 input dimension (1280).
        hidden_dim: Projection dimension (512).
        dilations: Sequence of dilation rates per block (e.g. [1, 2, 4, 8]).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 512,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList([
            DilatedSEResBlock(channels=hidden_dim, dilation=d, dropout=dropout)
            for d in dilations
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (B, L, input_dim) -> transpose to (B, input_dim, L)
        out = x.transpose(1, 2)
        out = self.proj(out)
        for block in self.blocks:
            out = block(out)
        # Transpose back to (B, L, hidden_dim)
        return out.transpose(1, 2)
