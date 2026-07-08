"""SHAP-based explainability for ProtIntel.

Provides GradientSHAP attributions via the ExplainabilityEngine.
This module serves as a convenience wrapper.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.xai.explainability_engine import ExplainabilityEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SHAPExplainer:
    """SHAP-based explainer for protein secondary structure predictions.

    Wraps the ExplainabilityEngine to provide GradientSHAP attributions
    with convenient high-level methods.

    Args:
        model: The ProtIntelModel instance.
        device: Computation device.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
    ) -> None:
        self.engine = ExplainabilityEngine(model=model, device=device)
        self.device = device

    def explain(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        target_class: int,
        task: str = "q3",
        n_samples: int = 20,
    ) -> dict[str, torch.Tensor]:
        """Compute SHAP-based explanations for a prediction.

        Args:
            embeddings: Pre-computed embeddings of shape (1, L, 1280).
            attention_mask: Attention mask of shape (1, L).
            target_class: Target class index to explain.
            task: 'q3' or 'q8'.
            n_samples: Number of random baseline samples.

        Returns:
            Dictionary containing:
                - importance: Per-residue importance of shape (L,).
                - method: String identifying the attribution method.
        """
        importance = self.engine.compute_gradient_shap(
            embeddings=embeddings,
            attention_mask=attention_mask,
            target_class=target_class,
            task=task,
            n_samples=n_samples,
        )

        return {
            "importance": importance,
            "method": "gradient_shap",
        }

    def explain_all_classes(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        task: str = "q3",
        n_samples: int = 20,
    ) -> dict[int, torch.Tensor]:
        """Compute SHAP attributions for all classes.

        Args:
            embeddings: Pre-computed embeddings of shape (1, L, 1280).
            attention_mask: Attention mask of shape (1, L).
            task: 'q3' or 'q8'.
            n_samples: Number of samples.

        Returns:
            Dictionary mapping class index → importance tensor of shape (L,).
        """
        num_classes = 3 if task == "q3" else 8
        all_attributions: dict[int, torch.Tensor] = {}

        for cls_idx in range(num_classes):
            importance = self.engine.compute_gradient_shap(
                embeddings=embeddings,
                attention_mask=attention_mask,
                target_class=cls_idx,
                task=task,
                n_samples=n_samples,
            )
            all_attributions[cls_idx] = importance

        return all_attributions
