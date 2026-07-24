"""Bidirectional LSTM encoder for capturing long-range dependencies.

Processes CNN-encoded features through a multi-layer bidirectional
LSTM to capture sequential dependencies across the full protein chain.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BiLSTMEncoder(nn.Module):
    """Bidirectional LSTM encoder for sequence modeling.

    Processes feature sequences through stacked bidirectional LSTM
    layers. Supports variable-length sequences via pack/pad operations
    for efficient computation.

    Args:
        input_dim: Dimension of input features (CNN output dim).
        hidden_dim: Hidden size per direction. The total output
            dimension is ``2 * hidden_dim`` when bidirectional.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout between LSTM layers (only applied if
            ``num_layers > 1``).
        bidirectional: Whether to use bidirectional LSTM.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim * self.num_directions

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"BiLSTMEncoder: {input_dim} → {self.output_dim} "
            f"({num_layers} layers, {'bi' if bidirectional else 'uni'}directional, "
            f"{total_params / 1e3:.1f}K params)"
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the BiLSTM.

        Args:
            x: Input tensor of shape (B, L, input_dim).
            lengths: Optional tensor of actual sequence lengths of shape (B,).
                Used for packed sequence processing. If None, all sequences
                are treated as having the same length.

        Returns:
            Tuple of:
                - output: Hidden states for all timesteps, shape
                  (B, L, 2 * hidden_dim).
                - (h_n, c_n): Final hidden and cell states, each of shape
                  (num_layers * num_directions, B, hidden_dim).
        """
        batch_size, max_len, _ = x.shape

        if lengths is not None:
            # Sort by length for pack_padded_sequence (must be descending)
            # Note: the batch is already sorted by the collate function,
            # but we handle the general case here
            lengths_cpu = lengths.cpu().clamp(min=1)

            packed = pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            packed_output, (h_n, c_n) = self.lstm(packed)
            output, _ = pad_packed_sequence(
                packed_output, batch_first=True, total_length=max_len
            )
        else:
            output, (h_n, c_n) = self.lstm(x)

        output = self.output_dropout(output)
        return output, (h_n, c_n)

    def get_output_dim(self) -> int:
        """Return the output feature dimension.

        Returns:
            The total output dimension (2 * hidden_dim for bidirectional).
        """
        return self.output_dim

    def get_final_hidden(
        self,
        h_n: torch.Tensor,
    ) -> torch.Tensor:
        """Extract the final hidden state for sequence-level tasks.

        Concatenates the final forward and backward hidden states
        from the last LSTM layer.

        Args:
            h_n: Final hidden states of shape
                (num_layers * num_directions, B, hidden_dim).

        Returns:
            Concatenated final hidden state of shape (B, 2 * hidden_dim)
            for bidirectional, or (B, hidden_dim) for unidirectional.
        """
        if self.bidirectional:
            # h_n shape: (num_layers * 2, B, hidden_dim)
            # Get last layer's forward and backward states
            forward_final = h_n[-2]  # (B, hidden_dim)
            backward_final = h_n[-1]  # (B, hidden_dim)
            return torch.cat([forward_final, backward_final], dim=-1)
        else:
            return h_n[-1]  # (B, hidden_dim)
