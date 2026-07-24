import tempfile
from pathlib import Path
from unittest.mock import MagicMock
import torch
import pytest

from src.models.embedding_generator import EmbeddingGenerator

def test_cache_gating_frozen_uses_cache():
    """Verify that in frozen mode, the generator uses the cache and skips loading."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_cache.db"
        generator = EmbeddingGenerator(
            model_name="facebook/esm2_t12_35M_UR50D",
            embedding_dim=480,
            freeze=True,
            finetune_last_n_layers=0,
            db_path=db_path,
        )
        
        seq = "MKFLILLFNILCLFPVLAADNHGVSMNAS"
        dummy_tensor = torch.randn(len(seq), 480)
        generator.sqlite_cache.set(seq, 480, dummy_tensor)
        
        # Spy on loading
        generator._load_model = MagicMock()
        
        result = generator.generate_single(seq, use_cache=True)
        
        # Should return cached result and NOT call _load_model
        assert torch.allclose(result, dummy_tensor)
        generator._load_model.assert_not_called()

def test_cache_gating_finetuning_ignores_cache():
    """Verify that when ESM-2 layers are unfrozen, cache is bypassed."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_cache.db"
        generator = EmbeddingGenerator(
            model_name="facebook/esm2_t12_35M_UR50D",
            embedding_dim=480,
            freeze=True,
            finetune_last_n_layers=3,  # Fine-tuning active
            db_path=db_path,
        )
        
        seq = "MKFLILLFNILCLFPVLAADNHGVSMNAS"
        dummy_tensor = torch.randn(len(seq), 480)
        generator.sqlite_cache.set(seq, 480, dummy_tensor)
        
        # Mock load and forward
        generator._load_model = MagicMock()
        generator._tokenizer = MagicMock()
        generator._model = MagicMock()
        generator._loaded = True
        
        # Mock outputs
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, len(seq) + 2, 480)
        generator._model.return_value = mock_output
        
        result = generator.generate_single(seq, use_cache=True)
        
        # Should bypass cache, load the model and call forward
        generator._load_model.assert_called()
        generator.model.assert_called()
        assert not torch.allclose(result, dummy_tensor)
