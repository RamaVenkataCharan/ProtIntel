"""Explainability engine using Captum Integrated Gradients.

Computes per-residue importance scores that indicate which amino acids
most influence the model's secondary structure predictions.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExplainabilityEngine:
    """Wrapper for computing Integrated Gradients attributions.

    Uses Captum's IntegratedGradients method to compute per-residue
    importance scores. Falls back to a gradient-based method if
    Captum is not available.

    Args:
        model: The ProtIntelModel instance.
        device: Computation device.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.eval()
        self._captum_available = False

        try:
            from captum.attr import IntegratedGradients
            self._captum_available = True
            logger.info("Captum available — using IntegratedGradients")
        except ImportError:
            logger.warning(
                "Captum not installed. Using basic gradient attribution. "
                "Install with: pip install captum"
            )

    def compute_integrated_gradients(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        target_class: int,
        task: str = "q3",
        steps: int = 50,
    ) -> torch.Tensor:
        """Compute Integrated Gradients attribution per residue.

        Args:
            embeddings: Pre-computed ESM-2 embeddings of shape (1, L, 1280).
            attention_mask: Attention mask of shape (1, L).
            target_class: Target class index to attribute.
            task: 'q3' or 'q8'.
            steps: Number of interpolation steps for IG.

        Returns:
            Per-residue importance scores of shape (L,), normalized
            to [0, 1].
        """
        embeddings = embeddings.to(self.device).requires_grad_(True)
        attention_mask = attention_mask.to(self.device)

        if self._captum_available:
            return self._ig_captum(embeddings, attention_mask, target_class, task, steps)
        else:
            return self._ig_manual(embeddings, attention_mask, target_class, task, steps)

    def _ig_captum(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        target_class: int,
        task: str,
        steps: int,
    ) -> torch.Tensor:
        """Integrated Gradients using Captum library.

        Args:
            embeddings: Input embeddings requiring gradients.
            attention_mask: Attention mask.
            target_class: Target class for attribution.
            task: 'q3' or 'q8'.
            steps: Number of IG steps.

        Returns:
            Normalized importance scores of shape (L,).
        """
        from captum.attr import IntegratedGradients

        def forward_fn(emb: torch.Tensor) -> torch.Tensor:
            """Forward function for Captum."""
            outputs = self.model(
                embeddings=emb,
                attention_mask=attention_mask,
                seq_lengths=attention_mask.sum(dim=1),
            )
            logits = outputs[f"{task}_logits"]  # (1, L, C)
            # Sum logits for target class across sequence
            return logits[:, :, target_class].sum(dim=1)  # (1,)

        ig = IntegratedGradients(forward_fn)

        # Baseline: zero embeddings
        baseline = torch.zeros_like(embeddings)

        attributions = ig.attribute(
            embeddings,
            baselines=baseline,
            n_steps=steps,
            return_convergence_delta=False,
        )  # (1, L, 1280)

        # Aggregate across embedding dimension
        importance = attributions.abs().sum(dim=-1).squeeze(0)  # (L,)

        # Mask padding
        seq_len = int(attention_mask.sum().item())
        importance = importance[:seq_len]

        # Normalize to [0, 1]
        if importance.max() > 0:
            importance = importance / importance.max()

        return importance.detach().cpu()

    def _ig_manual(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        target_class: int,
        task: str,
        steps: int,
    ) -> torch.Tensor:
        """Manual Integrated Gradients implementation.

        Args:
            embeddings: Input embeddings.
            attention_mask: Attention mask.
            target_class: Target class.
            task: 'q3' or 'q8'.
            steps: Number of interpolation steps.

        Returns:
            Normalized importance scores of shape (L,).
        """
        baseline = torch.zeros_like(embeddings)
        scaled_inputs = [
            baseline + (float(i) / steps) * (embeddings - baseline)
            for i in range(steps + 1)
        ]

        grads_list: list[torch.Tensor] = []
        for scaled in scaled_inputs:
            scaled = scaled.detach().requires_grad_(True)
            outputs = self.model(
                embeddings=scaled,
                attention_mask=attention_mask,
                seq_lengths=attention_mask.sum(dim=1),
            )
            logits = outputs[f"{task}_logits"]
            target_sum = logits[:, :, target_class].sum()
            target_sum.backward()
            grads_list.append(scaled.grad.detach().clone())

        # Approximate integral using trapezoidal rule
        avg_grads = torch.stack(grads_list).mean(dim=0)  # (1, L, 1280)
        attributions = (embeddings - baseline).detach() * avg_grads  # (1, L, 1280)

        importance = attributions.abs().sum(dim=-1).squeeze(0)  # (L,)

        seq_len = int(attention_mask.sum().item())
        importance = importance[:seq_len]

        if importance.max() > 0:
            importance = importance / importance.max()

        return importance.cpu()

    def compute_gradient_shap(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        target_class: int,
        task: str = "q3",
        n_samples: int = 20,
    ) -> torch.Tensor:
        """Compute GradientSHAP attributions per residue.

        Args:
            embeddings: Pre-computed ESM-2 embeddings of shape (1, L, 1280).
            attention_mask: Attention mask of shape (1, L).
            target_class: Target class index.
            task: 'q3' or 'q8'.
            n_samples: Number of random baseline samples.

        Returns:
            Per-residue SHAP values of shape (L,), normalized to [0, 1].
        """
        embeddings = embeddings.to(self.device)
        attention_mask = attention_mask.to(self.device)

        if self._captum_available:
            from captum.attr import GradientShap

            def forward_fn(emb: torch.Tensor) -> torch.Tensor:
                outputs = self.model(
                    embeddings=emb,
                    attention_mask=attention_mask.expand(emb.size(0), -1),
                    seq_lengths=attention_mask.sum(dim=1).expand(emb.size(0)),
                )
                logits = outputs[f"{task}_logits"]
                return logits[:, :, target_class].sum(dim=1)

            gs = GradientShap(forward_fn)
            baselines = torch.randn(n_samples, *embeddings.shape[1:], device=self.device) * 0.01

            attributions = gs.attribute(
                embeddings,
                baselines=baselines,
                n_samples=n_samples,
            )

            importance = attributions.abs().sum(dim=-1).squeeze(0)
            seq_len = int(attention_mask.sum().item())
            importance = importance[:seq_len]

            if importance.max() > 0:
                importance = importance / importance.max()

            return importance.detach().cpu()
        else:
            logger.warning("GradientSHAP requires Captum. Falling back to IG.")
            return self.compute_integrated_gradients(
                embeddings, attention_mask, target_class, task
            )
