"""Multi-scale CNN encoder with residual connections for ProtIntel.

Implements parallel convolutions with kernel sizes 3, 5, 7 to capture
local residue patterns at multiple scales, followed by stacked residual
blocks for deeper feature extraction.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MultiScaleConv1d(nn.Module):
    """Parallel multi-scale 1D convolution block.

    Applies multiple Conv1d layers with different kernel sizes in parallel,
    concatenates their outputs, and projects to a target dimension.

    Args:
        in_channels: Number of input channels (embedding dimension).
        out_channels: Number of output channels per kernel size.
        kernel_sizes: List of kernel sizes for parallel convolutions.
        dropout: Dropout rate applied after projection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: list[int],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.convolutions = nn.ModuleList()
        for k in kernel_sizes:
            self.convolutions.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=k,
                        padding=k // 2,  # 'same' padding
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        # Project concatenated multi-scale features to hidden_dim
        total_channels = out_channels * len(kernel_sizes)
        self.projection = nn.Sequential(
            nn.Conv1d(total_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.out_channels = out_channels
        logger.info(
            f"MultiScaleConv1d: {in_channels} → {out_channels} "
            f"(kernels={kernel_sizes})"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through multi-scale convolutions.

        Args:
            x: Input tensor of shape (B, C_in, L).

        Returns:
            Output tensor of shape (B, C_out, L).
        """
        conv_outputs = [conv(x) for conv in self.convolutions]
        concatenated = torch.cat(conv_outputs, dim=1)  # (B, C_out * N_kernels, L)
        return self.projection(concatenated)  # (B, C_out, L)


class ResidualBlock1d(nn.Module):
    """1D residual block with two convolutions and skip connection.

    Architecture: Conv1d → BN → ReLU → Conv1d → BN → + skip → ReLU

    Args:
        channels: Number of input and output channels (must be equal
            for the skip connection).
        kernel_size: Kernel size for both convolutions.
        dropout: Dropout rate applied after the block.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor of shape (B, C, L).

        Returns:
            Output tensor of shape (B, C, L).
        """
        residual = x
        out = self.block(x)
        out = out + residual
        out = self.relu(out)
        out = self.dropout(out)
        return out


class CNNEncoder(nn.Module):
    """Multi-scale CNN encoder with residual blocks.

    Processes per-residue embeddings through multi-scale parallel
    convolutions followed by stacked residual blocks. All operations
    preserve the sequence length.

    Architecture:
        Input (B, L, input_dim)
        → Transpose to (B, input_dim, L)
        → MultiScaleConv1d (kernels 3, 5, 7)
        → ResidualBlock × N
        → Transpose to (B, L, hidden_dim)

    Args:
        input_dim: Input feature dimension (ESM-2 embedding dim, typically 1280).
        hidden_dim: Output feature dimension after CNN processing.
        kernel_sizes: Kernel sizes for multi-scale convolution.
        num_residual_blocks: Number of residual blocks to stack.
        dropout: Dropout rate for all layers.
    """

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 512,
        kernel_sizes: Optional[list[int]] = None,
        num_residual_blocks: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Multi-scale parallel convolutions
        self.multi_scale = MultiScaleConv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
        )

        # Stacked residual blocks
        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock1d(channels=hidden_dim, kernel_size=3, dropout=dropout)
                for _ in range(num_residual_blocks)
            ]
        )

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"CNNEncoder: {input_dim} → {hidden_dim}, "
            f"{num_residual_blocks} residual blocks, "
            f"{total_params / 1e3:.1f}K params"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CNN encoder.

        Args:
            x: Input tensor of shape (B, L, input_dim).

        Returns:
            Output tensor of shape (B, L, hidden_dim).
        """
        # (B, L, C) → (B, C, L) for Conv1d
        x = x.transpose(1, 2)

        # Multi-scale convolution
        x = self.multi_scale(x)

        # Residual blocks
        x = self.residual_blocks(x)

        # (B, C, L) → (B, L, C)
        x = x.transpose(1, 2)

        return x

    def get_output_dim(self) -> int:
        """Return the output feature dimension.

        Returns:
            The hidden dimension of the CNN output.
        """
        return self.hidden_dim
