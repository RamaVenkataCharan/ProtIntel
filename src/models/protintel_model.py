"""Unified ProtIntelModel assembling all architectural components.

Chains ESM-2 embeddings → CNN encoder → BiLSTM → self-attention →
feed-forward → Q3/Q8 prediction heads into a single nn.Module.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from src.models.attention_block import AttentionBlock, FeedForwardBlock
from src.models.bilstm_encoder import BiLSTMEncoder
from src.models.cnn_encoder import CNNEncoder
from src.models.embedding_generator import EmbeddingGenerator
from src.models.prediction_head import PredictionHead
from src.utils.config_loader import ModelConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ProtIntelModel(nn.Module):
    """Unified model for explainable protein secondary structure prediction.

    Assembles the full pipeline:
        ESM-2 embeddings (L, 480)
        → CNN encoder (L, hidden_dim)
        → BiLSTM (L, 2*lstm_hidden)
        → Multi-head self-attention (L, embed_dim)
        → Feed-forward (L, embed_dim)
        → Q3 head (L, 3) + Q8 head (L, 8)

    The forward pass returns a comprehensive dictionary of outputs
    including logits, probabilities, predictions, confidence scores,
    attention weights, and embeddings for downstream XAI.

    Args:
        config: Model configuration specifying all component hyperparameters.
        device: Device to place the model on.
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.device_str = device

        # ESM-2 embedding generator
        self.embedding_generator = EmbeddingGenerator(
            model_name=self.config.esm2.model_name,
            embedding_dim=self.config.esm2.embedding_dim,
            freeze=self.config.esm2.freeze,
            finetune_last_n_layers=self.config.esm2.finetune_last_n_layers,
            device=device,
        )

        # CNN encoder
        self.cnn_encoder = CNNEncoder(
            input_dim=self.config.esm2.embedding_dim,
            hidden_dim=self.config.cnn.hidden_dim,
            kernel_sizes=self.config.cnn.kernel_sizes,
            num_residual_blocks=self.config.cnn.num_residual_blocks,
            dropout=self.config.cnn.dropout,
        )

        # BiLSTM encoder
        self.bilstm_encoder = BiLSTMEncoder(
            input_dim=self.config.cnn.hidden_dim,
            hidden_dim=self.config.bilstm.hidden_dim,
            num_layers=self.config.bilstm.num_layers,
            dropout=self.config.bilstm.dropout,
            bidirectional=self.config.bilstm.bidirectional,
        )

        # Dimension adapter (BiLSTM output → attention input)
        bilstm_output_dim = self.bilstm_encoder.get_output_dim()
        attn_dim = self.config.attention.embed_dim
        self.dim_adapter = nn.Identity()
        if bilstm_output_dim != attn_dim:
            self.dim_adapter = nn.Linear(bilstm_output_dim, attn_dim)
            logger.info(
                f"Dimension adapter: {bilstm_output_dim} → {attn_dim}"
            )

        # Self-attention block
        self.attention = AttentionBlock(
            embed_dim=attn_dim,
            num_heads=self.config.attention.num_heads,
            dropout=self.config.attention.dropout,
        )

        # Feed-forward block
        self.feedforward = FeedForwardBlock(
            embed_dim=attn_dim,
            hidden_dim=self.config.feedforward.hidden_dim,
            dropout=self.config.feedforward.dropout,
        )

        # Prediction heads
        self.q3_head = PredictionHead(
            input_dim=attn_dim,
            hidden_dim=self.config.q3_head.hidden_dim,
            num_classes=self.config.q3_head.num_classes,
            dropout=self.config.q3_head.dropout,
        )

        self.q8_head = PredictionHead(
            input_dim=attn_dim,
            hidden_dim=self.config.q8_head.hidden_dim,
            num_classes=self.config.q8_head.num_classes,
            dropout=self.config.q8_head.dropout,
        )

        # Log total parameter count
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        logger.info(
            f"ProtIntelModel initialized: "
            f"{total_params / 1e6:.1f}M total params, "
            f"{trainable_params / 1e6:.1f}M trainable"
        )

    def forward(
        self,
        sequences: Optional[list[str]] = None,
        embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        seq_lengths: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass through the ProtIntel pipeline.

        Either ``sequences`` (raw AA strings) or ``embeddings``
        (pre-computed ESM-2 outputs) must be provided.

        Args:
            sequences: List of amino acid sequence strings. If provided,
                ESM-2 embeddings are generated on-the-fly.
            embeddings: Pre-computed ESM-2 embeddings of shape
                (B, L, 480). If provided, skips embedding generation.
            attention_mask: Binary mask of shape (B, L) where 1 indicates
                real tokens and 0 indicates padding.
            seq_lengths: Actual sequence lengths of shape (B,), used
                for BiLSTM packed sequence processing.

        Returns:
            Dictionary containing:
                - q3_logits: (B, L, 3)
                - q8_logits: (B, L, 8)
                - q3_probs: (B, L, 3)
                - q8_probs: (B, L, 8)
                - q3_preds: (B, L) — argmax class indices
                - q8_preds: (B, L) — argmax class indices
                - confidence: (B, L) — max Q3 probability per residue
                - attention_weights: (B, num_heads, L, L)
                - embeddings: (B, L, 480) — ESM-2 embeddings

        Raises:
            ValueError: If neither sequences nor embeddings are provided.
        """
        # Step 1: Get ESM-2 embeddings
        if embeddings is None:
            if sequences is None:
                raise ValueError(
                    "Either 'sequences' or 'embeddings' must be provided"
                )
            embeddings = self.embedding_generator(sequences, attention_mask)
            embeddings = embeddings.to(next(self.cnn_encoder.parameters()).device)

        raw_embeddings = embeddings.clone()

        # Step 2: CNN encoder
        cnn_output = self.cnn_encoder(embeddings)  # (B, L, hidden_dim)

        # Step 3: BiLSTM encoder
        bilstm_output, (h_n, c_n) = self.bilstm_encoder(
            cnn_output, lengths=seq_lengths
        )  # (B, L, 2*lstm_hidden)

        # Step 4: Dimension adaptation
        adapted = self.dim_adapter(bilstm_output)  # (B, L, attn_dim)

        # Step 5: Self-attention
        # Convert seq_lengths or attention_mask to key_padding_mask (True = ignore)
        key_padding_mask = None
        if seq_lengths is not None:
            B, max_len = adapted.size(0), adapted.size(1)
            device = adapted.device
            seq_range = torch.arange(max_len, device=device).unsqueeze(0).expand(B, max_len)
            seq_limits = seq_lengths.unsqueeze(1).to(device)
            key_padding_mask = seq_range >= seq_limits
        elif attention_mask is not None:
            mask_len = attention_mask.size(1)
            adapted_len = adapted.size(1)
            if mask_len == adapted_len:
                key_padding_mask = attention_mask == 0

        attn_output, attention_weights = self.attention(
            adapted, key_padding_mask=key_padding_mask
        )  # (B, L, attn_dim), (B, heads, L, L)

        # Step 6: Feed-forward
        ff_output = self.feedforward(attn_output)  # (B, L, attn_dim)

        # Step 7: Prediction heads
        q3_logits, q3_probs, q3_confidence = self.q3_head(ff_output)
        q8_logits, q8_probs, q8_confidence = self.q8_head(ff_output)

        # Predictions (argmax)
        q3_preds = q3_probs.argmax(dim=-1)  # (B, L)
        q8_preds = q8_probs.argmax(dim=-1)  # (B, L)

        return {
            "q3_logits": q3_logits,
            "q8_logits": q8_logits,
            "q3_probs": q3_probs,
            "q8_probs": q8_probs,
            "q3_preds": q3_preds,
            "q8_preds": q8_preds,
            "confidence": q3_confidence,
            "attention_weights": attention_weights,
            "embeddings": raw_embeddings,
        }

    def get_downstream_parameters(self) -> list[nn.Parameter]:
        """Get parameters for the downstream model (excluding ESM-2).

        Useful for creating an optimizer that only trains the CNN,
        BiLSTM, attention, and prediction heads while keeping
        ESM-2 frozen.

        Returns:
            List of trainable parameters from non-ESM-2 components.
        """
        if not self.embedding_generator._loaded:
            self.embedding_generator._load_model()
        params: list[nn.Parameter] = []
        for name, param in self.named_parameters():
            if not name.startswith("embedding_generator"):
                params.append(param)
            elif param.requires_grad:
                # Include ESM-2 params only if they're unfrozen
                params.append(param)
        return params

    def count_parameters(self) -> dict[str, int]:
        """Count parameters per component.

        Returns:
            Dictionary mapping component names to parameter counts.
        """
        if not self.embedding_generator._loaded:
            self.embedding_generator._load_model()
        components = {
            "esm2": self.embedding_generator,
            "cnn": self.cnn_encoder,
            "bilstm": self.bilstm_encoder,
            "attention": self.attention,
            "feedforward": self.feedforward,
            "q3_head": self.q3_head,
            "q8_head": self.q8_head,
        }

        counts = {}
        for name, module in components.items():
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(
                p.numel() for p in module.parameters() if p.requires_grad
            )
            counts[name] = {"total": total, "trainable": trainable}

        return counts
