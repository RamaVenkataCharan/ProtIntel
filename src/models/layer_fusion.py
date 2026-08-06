"""Multi-Layer Embedding Fusion module for ProtIntel.

Applies a learnable softmax-weighted combination over hidden states extracted
from multiple transformer layers of ESM-2 (e.g. layers 6, 12, 18, 24, 30, 33).
Middle layers preserve local secondary structure geometry (helices/strands), while
top layers capture global evolutionary context.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerFusion(nn.Module):
    """Learnable Softmax Weighting for multi-layer ESM-2 representations.

    Args:
        num_layers: Number of transformer layers to fuse.
        embedding_dim: Dimensionality per residue (e.g. 1280 for ESM-2 650M).
    """

    def __init__(self, num_layers: int, embedding_dim: int = 1280) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        # Learnable layer weights initialized equally
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, hidden_states: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """Fuse hidden states across layers.

        Args:
            hidden_states: List of shape (B, L, D) tensors, or a stacked tensor of
                shape (num_layers, B, L, D).

        Returns:
            Fused representation of shape (B, L, D).
        """
        if isinstance(hidden_states, list):
            stacked = torch.stack(hidden_states, dim=0)  # (N, B, L, D)
        else:
            stacked = hidden_states

        weights = F.softmax(self.layer_weights, dim=0)  # (N,)
        # Reshape weights for broadcasting: (N, 1, 1, 1)
        weights = weights.view(-1, 1, 1, 1)
        fused = (stacked * weights).sum(dim=0)  # (B, L, D)
        return fused
