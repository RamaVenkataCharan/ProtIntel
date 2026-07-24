"""Shared pytest fixtures for API tests.

Fixture-model rationale
-----------------------
The ProtIntelModel embeds ESM-2 (facebook/esm2_t12_35M_UR50D), a ~140 MB
download that takes a few seconds to load.  Running that in CI on every push is
impractical.

Instead, every fixture here wires the FastAPI application to a *toy*
InferenceService whose internal ProtIntelModel uses a randomly-initialised
``nn.Embedding(33, 64)`` in place of the real ESM-2 encoder.  All downstream
components (CNN, BiLSTM, Attention, heads) are real but tiny — hidden_dim=32,
num_heads=4, etc.  This means:

  - Tests run in < 2 s per test even on a laptop CPU.
  - The full HTTP surface (serialisation, validation, routing) is exercised
    with real model forward passes.
  - Predictions are numerically arbitrary (random weights), but their *shape*,
    *type*, and *constraint properties* (probs sum to 1, lengths match, etc.)
    are fully verified.

If you later have a trained checkpoint and want to run assertion tests against
its accuracy, add a ``@pytest.mark.slow`` file that loads the real checkpoint
and skips when the checkpoint is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Generator

import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

# ── project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app
from backend.routers.predict import set_inference_service
from src.utils.config_loader import ModelConfig


# ── tiny fixture model ────────────────────────────────────────────────────────

class _TinyEmbeddingGenerator(nn.Module):
    """Drop-in replacement for EmbeddingGenerator that never downloads ESM-2.

    Uses a fixed random nn.Embedding of dim 64 over the 33-token ESM-2
    vocabulary.  The *structure* of the output tensor (B, L, embed_dim) is
    identical to the real generator.
    """

    VOCAB_SIZE = 33    # ESM-2 alphabet size
    EMBED_DIM = 64     # Tiny — real model uses 1280

    def __init__(self) -> None:
        super().__init__()
        self.embedding_dim = self.EMBED_DIM
        self.embed = nn.Embedding(self.VOCAB_SIZE, self.EMBED_DIM, padding_idx=1)
        # Deterministic weights so tests are reproducible.
        torch.manual_seed(0)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

    def forward(
        self,
        sequences: list[str],
        attention_mask: Any = None,
    ) -> torch.Tensor:
        """Tokenise sequences with a simple ASCII→vocab-index map and embed."""
        # Map each character to an index in [2, 32] (leave 0=pad, 1=unknown)
        max_len = max(len(s) for s in sequences)
        batch_ids = torch.ones(len(sequences), max_len, dtype=torch.long)  # 1=unknown
        for i, seq in enumerate(sequences):
            ids = [min(ord(c) % (self.VOCAB_SIZE - 2) + 2, self.VOCAB_SIZE - 1)
                   for c in seq]
            batch_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        return self.embed(batch_ids)  # (B, L, 64)


def _build_tiny_model(cfg: ModelConfig) -> Any:
    """Build a ProtIntelModel whose ESM-2 is swapped for TinyEmbeddingGenerator."""
    from src.models.cnn_encoder import CNNEncoder
    from src.models.bilstm_encoder import BiLSTMEncoder
    from src.models.attention_block import AttentionBlock, FeedForwardBlock
    from src.models.prediction_head import PredictionHead

    E = _TinyEmbeddingGenerator.EMBED_DIM  # 64

    cnn    = CNNEncoder(input_dim=E,  hidden_dim=32, num_residual_blocks=1)
    bilstm = BiLSTMEncoder(input_dim=32, hidden_dim=16, num_layers=1)
    attn   = AttentionBlock(embed_dim=32, num_heads=4)
    ffn    = FeedForwardBlock(embed_dim=32, hidden_dim=64)
    q3     = PredictionHead(input_dim=32, hidden_dim=16, num_classes=3)
    q8     = PredictionHead(input_dim=32, hidden_dim=16, num_classes=8)

    class _TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding_generator = _TinyEmbeddingGenerator()
            self.cnn_encoder    = cnn
            self.bilstm_encoder = bilstm
            self.dim_adapter    = nn.Linear(32, 32)   # identity-ish
            self.attention      = attn
            self.feedforward    = ffn
            self.q3_head        = q3
            self.q8_head        = q8

        def forward(
            self,
            sequences: list[str] | None = None,
            embeddings: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            seq_lengths: torch.Tensor | None = None,
        ) -> dict[str, torch.Tensor]:
            if embeddings is None:
                embeddings = self.embedding_generator(sequences, attention_mask)
            raw = embeddings.clone()
            x = self.cnn_encoder(embeddings)
            x, _ = self.bilstm_encoder(x, lengths=seq_lengths)
            x = self.dim_adapter(x)
            key_mask = (attention_mask == 0) if attention_mask is not None else None
            x, attn_w = self.attention(x, key_padding_mask=key_mask)
            x = self.feedforward(x)
            q3_logits, q3_probs, q3_conf = self.q3_head(x)
            q8_logits, q8_probs, q8_conf = self.q8_head(x)
            return {
                "q3_logits": q3_logits,
                "q8_logits": q8_logits,
                "q3_probs":  q3_probs,
                "q8_probs":  q8_probs,
                "q3_preds":  q3_probs.argmax(dim=-1),
                "q8_preds":  q8_probs.argmax(dim=-1),
                "confidence": q3_conf,
                "attention_weights": attn_w,
                "embeddings": raw,
            }

    return _TinyModel()


# ── InferenceService shim ─────────────────────────────────────────────────────

class _TinyInferenceService:
    """Minimal InferenceService that delegates to _TinyModel.

    Matches the interface used by the router (is_loaded, predict,
    get_model_info, device).
    """

    def __init__(self) -> None:
        cfg = ModelConfig()
        self.model = _build_tiny_model(cfg)
        self.model.eval()
        self.device = "cpu"
        self._model_loaded = True
        # Reuse the real preprocessor / decoders
        from src.data.preprocessor import SequencePreprocessor
        self.preprocessor = SequencePreprocessor()

    @property
    def is_loaded(self) -> bool:
        return self._model_loaded

    def predict(
        self,
        sequence: str,
        return_attention: bool = False,
        return_xai: bool = False,
        xai_method: str = "ig",
    ) -> dict[str, Any]:
        import time
        from src.data.preprocessor import (
            SequencePreprocessor,
            decode_q3_predictions,
            decode_q8_predictions,
        )
        from src.utils.io_utils import compute_sequence_hash

        start = time.time()
        cleaned = self.preprocessor.clean_sequence(sequence)
        seq_len = len(cleaned)

        with torch.no_grad():
            outputs = self.model(sequences=[cleaned])

        q3_preds = outputs["q3_preds"][0, :seq_len].cpu()
        q8_preds = outputs["q8_preds"][0, :seq_len].cpu()
        q3_probs = outputs["q3_probs"][0, :seq_len].cpu()
        q8_probs = outputs["q8_probs"][0, :seq_len].cpu()
        conf     = outputs["confidence"][0, :seq_len].cpu()

        result: dict[str, Any] = {
            "protein_id":       compute_sequence_hash(cleaned),
            "sequence":         cleaned,
            "length":           seq_len,
            "q3_prediction":    list(decode_q3_predictions(q3_preds)),
            "q8_prediction":    list(decode_q8_predictions(q8_preds)),
            "q3_probabilities": q3_probs.tolist(),
            "q8_probabilities": q8_probs.tolist(),
            "confidence":       conf.tolist(),
            "processing_time_ms": round((time.time() - start) * 1000, 2),
        }

        if return_attention:
            from src.xai.attention_rollout import compute_attention_rollout
            rollout = compute_attention_rollout(outputs["attention_weights"])
            result["attention_map"] = rollout[0, :seq_len, :seq_len].cpu().tolist()

        return result

    def get_model_info(self) -> dict[str, Any]:
        total     = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            "model_name":          "ProtIntel",
            "version":             "1.0.0",
            "architecture":        "ESM-2 → CNN → BiLSTM → Attention → Q3/Q8",
            "esm2_model":          "fixture/tiny-embedding",
            "total_parameters":    total,
            "trainable_parameters": trainable,
        }


# ── pytest fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def tiny_service() -> _TinyInferenceService:
    """Session-scoped fixture: build the tiny model once per test session."""
    return _TinyInferenceService()


@pytest.fixture(scope="session")
def client(tiny_service: _TinyInferenceService) -> Generator[TestClient, None, None]:
    """Session-scoped HTTP test client wired to the fixture model.

    Bypasses the startup event (which would try to load a checkpoint and
    download ESM-2) by clearing the startup event handlers and directly
    setting the global inference service.
    """
    app.router.on_startup.clear()
    with TestClient(app, raise_server_exceptions=True) as c:
        set_inference_service(tiny_service)
        yield c


# ── shared test data ──────────────────────────────────────────────────────────

SHORT_SEQ = "MKFLILLFNI"           # 10 AAs — minimum valid length
MEDIUM_SEQ = "MKFLILLFNILCLFPVLAADNHGVSMNAS"  # 30 AAs — representative protein
VALID_SEQS = [SHORT_SEQ, MEDIUM_SEQ]
