"""Inference service wrapping the ProtIntel model for API use."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import torch

from src.data.preprocessor import (
    SequencePreprocessor,
    decode_q3_predictions,
    decode_q8_predictions,
)
from src.models.protintel_model import ProtIntelModel
from src.xai.attention_rollout import compute_attention_rollout
from src.xai.explainability_engine import ExplainabilityEngine
from src.utils.config_loader import ModelConfig, load_config
from src.utils.io_utils import compute_sequence_hash
from src.utils.logger import get_logger

logger = get_logger(__name__)


class InferenceService:
    """High-level inference service for the ProtIntel API.

    Wraps model loading, preprocessing, inference, and XAI computation
    into a single service class.

    Args:
        checkpoint_path: Path to the model checkpoint.
        device: Inference device.
        model_config: Model configuration.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str | Path] = None,
        device: str = "cpu",
        model_config: Optional[ModelConfig] = None,
    ) -> None:
        self.device = device
        self.config = model_config or ModelConfig()
        self.preprocessor = SequencePreprocessor()
        self.model: Optional[ProtIntelModel] = None
        self.xai_engine: Optional[ExplainabilityEngine] = None
        self._checkpoint_path = checkpoint_path
        self._model_loaded = False

    def load_model(self) -> None:
        """Load the model from checkpoint."""
        self.model = ProtIntelModel(config=self.config, device=self.device)

        if self._checkpoint_path and Path(self._checkpoint_path).exists():
            checkpoint = torch.load(
                str(self._checkpoint_path),
                map_location=self.device,
                weights_only=False,
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded checkpoint from {self._checkpoint_path}")
        else:
            logger.warning(
                "No checkpoint loaded — model has random weights. "
                "Run train.py first for meaningful predictions."
            )

        self.model.eval()
        self.xai_engine = ExplainabilityEngine(
            model=self.model, device=self.device
        )
        self._model_loaded = True
        logger.info("Inference service ready")

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model_loaded

    def predict(
        self,
        sequence: str,
        return_attention: bool = False,
        return_xai: bool = False,
        xai_method: str = "ig",
    ) -> dict[str, Any]:
        """Run prediction on a single sequence.

        Args:
            sequence: Cleaned amino acid sequence string.
            return_attention: Include attention map.
            return_xai: Compute XAI attributions.
            xai_method: XAI method ('ig', 'shap', 'rollout').

        Returns:
            Prediction result dictionary.
        """
        if not self._model_loaded:
            self.load_model()

        start_time = time.time()

        # Clean sequence
        cleaned = self.preprocessor.clean_sequence(sequence)
        seq_len = len(cleaned)

        # Generate embeddings and run model
        with torch.no_grad():
            outputs = self.model(
                sequences=[cleaned],
                attention_mask=None,
            )

        # Extract results for actual sequence length
        q3_preds = outputs["q3_preds"][0, :seq_len].cpu()
        q8_preds = outputs["q8_preds"][0, :seq_len].cpu()
        q3_probs = outputs["q3_probs"][0, :seq_len].cpu()
        q8_probs = outputs["q8_probs"][0, :seq_len].cpu()
        confidence = outputs["confidence"][0, :seq_len].cpu()

        result: dict[str, Any] = {
            "protein_id": compute_sequence_hash(cleaned),
            "sequence": cleaned,
            "length": seq_len,
            "q3_prediction": list(decode_q3_predictions(q3_preds)),
            "q8_prediction": list(decode_q8_predictions(q8_preds)),
            "q3_probabilities": q3_probs.tolist(),
            "q8_probabilities": q8_probs.tolist(),
            "confidence": confidence.tolist(),
        }

        # Attention map
        if return_attention:
            attn_weights = outputs["attention_weights"]  # (1, H, L, L)
            rollout = compute_attention_rollout(attn_weights)
            result["attention_map"] = rollout[0, :seq_len, :seq_len].cpu().tolist()

        # XAI
        if return_xai and self.xai_engine is not None:
            embeddings = outputs["embeddings"]
            mask = torch.ones(1, embeddings.size(1), dtype=torch.long)
            target_class = int(q3_preds[0].item())

            if xai_method == "ig":
                importance = self.xai_engine.compute_integrated_gradients(
                    embeddings=embeddings,
                    attention_mask=mask,
                    target_class=target_class,
                )
            elif xai_method == "shap":
                importance = self.xai_engine.compute_gradient_shap(
                    embeddings=embeddings,
                    attention_mask=mask,
                    target_class=target_class,
                )
            elif xai_method == "rollout":
                rollout = compute_attention_rollout(outputs["attention_weights"])
                importance = rollout[0, 0, :seq_len].cpu()
                if importance.max() > 0:
                    importance = importance / importance.max()
            else:
                importance = torch.zeros(seq_len)

            result["residue_importance"] = importance.tolist()
            result["xai_method"] = xai_method

        elapsed_ms = (time.time() - start_time) * 1000
        result["processing_time_ms"] = round(elapsed_ms, 2)

        return result

    def get_model_info(self) -> dict[str, Any]:
        """Get model architecture information."""
        if not self._model_loaded:
            self.load_model()

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        return {
            "model_name": "ProtIntel",
            "version": "1.0.0",
            "architecture": "ESM-2 → CNN → BiLSTM → Attention → Q3/Q8",
            "esm2_model": self.config.esm2.model_name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }
