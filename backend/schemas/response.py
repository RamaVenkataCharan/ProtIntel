"""Pydantic response models for the ProtIntel API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class PredictResponse(BaseModel):
    """Response body for a single prediction.

    Attributes:
        protein_id: Identifier for the protein (hash-based if unnamed).
        sequence: The cleaned input sequence.
        length: Number of residues.
        q3_prediction: Per-residue Q3 labels.
        q8_prediction: Per-residue Q8 labels.
        q3_probabilities: Per-residue Q3 class probabilities.
        q8_probabilities: Per-residue Q8 class probabilities.
        confidence: Per-residue prediction confidence.
        attention_map: Optional attention map (L × L).
        residue_importance: Optional per-residue importance scores.
        xai_method: XAI method used, if applicable.
        processing_time_ms: Inference time in milliseconds.
    """

    protein_id: str
    sequence: str
    length: int
    q3_prediction: list[str]
    q8_prediction: list[str]
    q3_probabilities: list[list[float]]
    q8_probabilities: list[list[float]]
    confidence: list[float]
    attention_map: Optional[list[list[float]]] = None
    residue_importance: Optional[list[float]] = None
    xai_method: Optional[str] = None
    processing_time_ms: float


class BatchPredictResponse(BaseModel):
    """Response body for batch prediction."""

    results: list[PredictResponse]
    total_sequences: int
    total_processing_time_ms: float


class ModelInfoResponse(BaseModel):
    """Response for /model_info endpoint."""

    model_name: str = "ProtIntel"
    version: str = "1.0.0"
    architecture: str = "ESM-2 → CNN → BiLSTM → Attention → Q3/Q8"
    esm2_model: str = "facebook/esm2_t33_650M_UR50D"
    total_parameters: int
    trainable_parameters: int
    q3_classes: list[str] = Field(default_factory=lambda: ["H (Helix)", "E (Sheet)", "C (Coil)"])
    q8_classes: list[str] = Field(
        default_factory=lambda: [
            "H (α-helix)", "E (β-strand)", "G (3₁₀-helix)", "I (π-helix)",
            "B (β-bridge)", "T (turn)", "S (bend)", "C (coil)",
        ]
    )


class MetricsResponse(BaseModel):
    """Response for /metrics endpoint."""

    dataset: str = "CB513"
    q3_accuracy: Optional[float] = None
    q8_accuracy: Optional[float] = None
    q3_mcc: Optional[float] = None
    per_class_q3: Optional[dict[str, float]] = None
    per_class_q8: Optional[dict[str, float]] = None


class HealthResponse(BaseModel):
    """Response for /health endpoint."""

    status: str = "healthy"
    model_loaded: bool
    device: str
