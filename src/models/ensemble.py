"""Multi-Model Ensemble Framework for ProtIntel.

Aggregates predictions from multiple independent ProtIntel model architectures
using Logit Averaging, Softmax Probability Averaging, or Weighted Majority Voting
to boost secondary structure prediction accuracy and reliability.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class EnsembleProtIntel(nn.Module):
    """Ensemble wrapper for combining multiple ProtIntelModel instances.

    Args:
        models: List of pre-trained ProtIntelModel checkpoints.
        voting_mode: Ensemble aggregation mode: 'logit_average', 'softmax_average', or 'weighted'.
        model_weights: Optional list of floating point weights per model.
    """

    def __init__(
        self,
        models: List[nn.Module],
        voting_mode: str = "softmax_average",
        model_weights: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)
        self.voting_mode = voting_mode

        if model_weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            total_w = sum(model_weights)
            self.weights = [w / total_w for w in model_weights]

    def forward(
        self,
        sequences: Optional[List[str]] = None,
        embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        seq_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        all_outputs = []
        for model in self.models:
            out = model(
                sequences=sequences,
                embeddings=embeddings,
                attention_mask=attention_mask,
                seq_lengths=seq_lengths,
            )
            all_outputs.append(out)

        if self.voting_mode == "logit_average":
            q3_logits = sum(w * out["q3_logits"] for w, out in zip(self.weights, all_outputs))
            q8_logits = sum(w * out["q8_logits"] for w, out in zip(self.weights, all_outputs))
            q3_probs = F.softmax(q3_logits, dim=-1)
            q8_probs = F.softmax(q8_logits, dim=-1)
        else:  # softmax_average
            q3_probs = sum(w * out["q3_probs"] for w, out in zip(self.weights, all_outputs))
            q8_probs = sum(w * out["q8_probs"] for w, out in zip(self.weights, all_outputs))
            q3_logits = torch.log(q3_probs.clamp(min=1e-8))
            q8_logits = torch.log(q8_probs.clamp(min=1e-8))

        q3_preds = q3_probs.argmax(dim=-1)
        q8_preds = q8_probs.argmax(dim=-1)
        confidence = q3_probs.max(dim=-1).values

        return {
            "q3_logits": q3_logits,
            "q8_logits": q8_logits,
            "q3_probs": q3_probs,
            "q8_probs": q8_probs,
            "q3_preds": q3_preds,
            "q8_preds": q8_preds,
            "confidence": confidence,
            "attention_weights": all_outputs[0]["attention_weights"],
        }
