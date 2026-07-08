"""Integration tests for the full ProtIntel model forward pass.

Uses a mock ESM-2 (nn.Embedding) to avoid downloading weights during CI.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.models.cnn_encoder import CNNEncoder
from src.models.bilstm_encoder import BiLSTMEncoder
from src.models.attention_block import AttentionBlock, FeedForwardBlock
from src.models.prediction_head import PredictionHead


class TestCNNEncoder:
    """Shape assertion tests for the CNN encoder."""

    def test_output_shape(self) -> None:
        """Test CNN encoder output shape."""
        encoder = CNNEncoder(input_dim=64, hidden_dim=32, num_residual_blocks=2)
        x = torch.randn(2, 50, 64)  # (batch, seq_len, features)
        out = encoder(x)
        assert out.shape == (2, 50, 32)

    def test_preserves_sequence_length(self) -> None:
        """Test that sequence length is preserved through CNN."""
        encoder = CNNEncoder(input_dim=128, hidden_dim=64)
        for seq_len in [10, 50, 100, 512]:
            x = torch.randn(1, seq_len, 128)
            out = encoder(x)
            assert out.shape[1] == seq_len


class TestBiLSTMEncoder:
    """Shape assertion tests for the BiLSTM encoder."""

    def test_output_shape(self) -> None:
        """Test BiLSTM encoder output shape."""
        encoder = BiLSTMEncoder(input_dim=32, hidden_dim=16, num_layers=2)
        x = torch.randn(2, 50, 32)
        output, (h_n, c_n) = encoder(x)
        assert output.shape == (2, 50, 32)  # 16 * 2 directions

    def test_with_lengths(self) -> None:
        """Test BiLSTM with variable-length sequences."""
        encoder = BiLSTMEncoder(input_dim=32, hidden_dim=16)
        x = torch.randn(3, 50, 32)
        lengths = torch.tensor([50, 30, 20])
        output, _ = encoder(x, lengths=lengths)
        assert output.shape == (3, 50, 32)

    def test_final_hidden(self) -> None:
        """Test extracting final hidden state."""
        encoder = BiLSTMEncoder(input_dim=32, hidden_dim=16)
        x = torch.randn(2, 50, 32)
        _, (h_n, _) = encoder(x)
        final = encoder.get_final_hidden(h_n)
        assert final.shape == (2, 32)


class TestAttentionBlock:
    """Shape assertion tests for the attention block."""

    def test_output_shape(self) -> None:
        """Test attention block output shape."""
        block = AttentionBlock(embed_dim=32, num_heads=4)
        x = torch.randn(2, 50, 32)
        output, weights = block(x)
        assert output.shape == (2, 50, 32)
        assert weights.shape == (2, 4, 50, 50)

    def test_with_padding_mask(self) -> None:
        """Test attention with key padding mask."""
        block = AttentionBlock(embed_dim=32, num_heads=4)
        x = torch.randn(2, 50, 32)
        mask = torch.zeros(2, 50, dtype=torch.bool)
        mask[0, 30:] = True  # Mask last 20 positions for sample 0
        output, weights = block(x, key_padding_mask=mask)
        assert output.shape == (2, 50, 32)


class TestFeedForwardBlock:
    """Shape assertion tests for the feed-forward block."""

    def test_output_shape(self) -> None:
        """Test FFN output shape."""
        block = FeedForwardBlock(embed_dim=32, hidden_dim=64)
        x = torch.randn(2, 50, 32)
        out = block(x)
        assert out.shape == (2, 50, 32)


class TestPredictionHead:
    """Shape assertion tests for prediction heads."""

    def test_q3_head_output(self) -> None:
        """Test Q3 head output shapes."""
        head = PredictionHead(input_dim=32, hidden_dim=16, num_classes=3)
        x = torch.randn(2, 50, 32)
        logits, probs, confidence = head(x)
        assert logits.shape == (2, 50, 3)
        assert probs.shape == (2, 50, 3)
        assert confidence.shape == (2, 50)

    def test_q8_head_output(self) -> None:
        """Test Q8 head output shapes."""
        head = PredictionHead(input_dim=32, hidden_dim=16, num_classes=8)
        x = torch.randn(2, 50, 32)
        logits, probs, confidence = head(x)
        assert logits.shape == (2, 50, 8)

    def test_probabilities_sum_to_one(self) -> None:
        """Test that softmax probabilities sum to 1."""
        head = PredictionHead(input_dim=32, hidden_dim=16, num_classes=3)
        x = torch.randn(2, 50, 32)
        _, probs, _ = head(x)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


class TestFullPipeline:
    """Integration test for the full model pipeline (without ESM-2)."""

    def test_full_forward_pass(self) -> None:
        """Test full pipeline with mock embeddings."""
        batch_size, seq_len, embed_dim = 2, 50, 64
        hidden_dim = 32

        # Mock ESM-2 output
        embeddings = torch.randn(batch_size, seq_len, embed_dim)

        # Pipeline
        cnn = CNNEncoder(input_dim=embed_dim, hidden_dim=hidden_dim, num_residual_blocks=2)
        bilstm = BiLSTMEncoder(input_dim=hidden_dim, hidden_dim=hidden_dim // 2)
        attention = AttentionBlock(embed_dim=hidden_dim, num_heads=4)
        ffn = FeedForwardBlock(embed_dim=hidden_dim, hidden_dim=hidden_dim * 2)
        q3_head = PredictionHead(input_dim=hidden_dim, hidden_dim=hidden_dim // 2, num_classes=3)
        q8_head = PredictionHead(input_dim=hidden_dim, hidden_dim=hidden_dim // 2, num_classes=8)

        # Forward
        cnn_out = cnn(embeddings)
        bilstm_out, _ = bilstm(cnn_out)
        attn_out, attn_weights = attention(bilstm_out)
        ff_out = ffn(attn_out)
        q3_logits, q3_probs, q3_conf = q3_head(ff_out)
        q8_logits, q8_probs, q8_conf = q8_head(ff_out)

        # Assertions
        assert q3_logits.shape == (batch_size, seq_len, 3)
        assert q8_logits.shape == (batch_size, seq_len, 8)
        assert attn_weights.shape[0] == batch_size
        assert q3_conf.shape == (batch_size, seq_len)
