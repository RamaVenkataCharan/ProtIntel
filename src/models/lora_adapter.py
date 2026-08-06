"""Parameter-Efficient Fine-Tuning via Low-Rank Adaptation (LoRA) for ESM-2.

Allows lightweight fine-tuning of ESM-2 transformer projections without storing
full multi-gigabyte checkpoints or overfitting.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps a standard nn.Linear layer with a Low-Rank Adaptation (LoRA) bypass.

    Args:
        linear_layer: Pre-trained Linear layer to adapt.
        r: Low-rank adaptation dimension (default = 8).
        lora_alpha: Scaling hyperparameter (default = 16.0).
        dropout: Dropout probability on LoRA input.
    """

    def __init__(
        self,
        linear_layer: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.linear = linear_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0

        in_features = linear_layer.in_features
        out_features = linear_layer.out_features

        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            self.dropout = nn.Dropout(p=dropout)
            # Initialize weights
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

        # Freeze original linear weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.linear(x)
        if self.r > 0:
            lora_out = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
            result = result + lora_out * self.scaling
        return result
