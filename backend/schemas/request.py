"""Pydantic request models for the ProtIntel API."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """Request body for single-sequence prediction.

    Attributes:
        sequence: Raw amino acid sequence string or FASTA-formatted text.
        return_attention: Whether to include the attention map in the response.
        return_xai: Whether to compute and return XAI attributions.
        xai_method: XAI method to use ('ig', 'shap', or 'rollout').
    """

    sequence: str = Field(
        ...,
        min_length=5,
        max_length=2048,
        description="Amino acid sequence (FASTA or raw)",
        examples=["MKFLILLFNILCLFPVLAADNHGVSMNAS"],
    )
    return_attention: bool = Field(
        default=False,
        description="Include attention map in response",
    )
    return_xai: bool = Field(
        default=False,
        description="Compute XAI attributions",
    )
    xai_method: Literal["ig", "shap", "rollout"] = Field(
        default="ig",
        description="XAI method: Integrated Gradients, SHAP, or attention rollout",
    )

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, v: str) -> str:
        """Validate and clean the input sequence."""
        # Strip FASTA header if present
        lines = v.strip().split("\n")
        if lines[0].startswith(">"):
            v = "".join(lines[1:])
        else:
            v = "".join(lines)

        v = v.upper().replace(" ", "").replace("-", "").replace("*", "")

        valid_chars = set("ACDEFGHIKLMNPQRSTVWYBZXJOU")
        invalid = set(v) - valid_chars
        if invalid:
            raise ValueError(
                f"Invalid amino acid characters: {sorted(invalid)}. "
                f"Valid: {sorted(valid_chars)}"
            )
        return v


class BatchPredictRequest(BaseModel):
    """Request body for batch prediction.

    Attributes:
        sequences: List of amino acid sequences.
        return_attention: Whether to include attention maps.
        return_xai: Whether to compute XAI attributions.
    """

    sequences: list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of amino acid sequences (max 50)",
    )
    return_attention: bool = False
    return_xai: bool = False
    xai_method: Literal["ig", "shap", "rollout"] = "ig"
