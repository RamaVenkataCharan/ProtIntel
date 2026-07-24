"""Prediction heads for per-residue Q3 and Q8 classification.

Each head takes the final hidden representations and produces
per-residue class logits, probabilities, and confidence scores.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PredictionHead(nn.Module):
    """Per-residue classification head.

    Two-layer feed-forward network producing class logits, softmax
    probabilities, and confidence (max probability) per residue.

    Architecture: Linear → ReLU → Dropout → Linear → (logits)
                  logits → Softmax → (probabilities)
                  max(probabilities) → (confidence)

    Args:
        input_dim: Dimension of input features.
        hidden_dim: Dimension of the intermediate layer.
        num_classes: Number of output classes (3 for Q3, 8 for Q8).
        dropout: Dropout rate between layers.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_classes: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"PredictionHead: {input_dim} → {hidden_dim} → {num_classes}, "
            f"{total_params / 1e3:.1f}K params"
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass producing logits, probabilities, and confidence.

        Args:
            x: Input tensor of shape (B, L, input_dim).

        Returns:
            Tuple of:
                - logits: Raw class scores, shape (B, L, num_classes).
                - probabilities: Softmax probabilities, shape (B, L, num_classes).
                - confidence: Max probability per residue, shape (B, L).
        """
        logits = self.classifier(x)  # (B, L, num_classes)
        probabilities = F.softmax(logits, dim=-1)  # (B, L, num_classes)
        confidence = probabilities.max(dim=-1).values  # (B, L)

        return logits, probabilities, confidence
