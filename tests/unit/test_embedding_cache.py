import tempfile
from pathlib import Path
import unittest
from unittest.mock import MagicMock

import pytest
import torch

from src.models.embedding_cache import SQLiteEmbeddingCache
from src.models.embedding_generator import EmbeddingGenerator


def test_sqlite_cache_basic_operations():
    """Verify that SQLiteEmbeddingCache can set, get, and delete on dimension mismatch."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_cache.db"
        cache = SQLiteEmbeddingCache(db_path)

        sequence = "MKFLILLFNILCLFPVLAADNHGVSMNAS"
        expected_dim = 480
        dummy_tensor = torch.randn(len(sequence), expected_dim)

        # 1. Initially get should return None
        assert cache.get(sequence, expected_dim) is None

        # 2. Set the cached value
        cache.set(sequence, expected_dim, dummy_tensor)

        # 3. Get should return a tensor identical to dummy_tensor
        retrieved = cache.get(sequence, expected_dim)
        assert retrieved is not None
        assert torch.allclose(retrieved, dummy_tensor)

        # 4. Get with a mismatched dimension should return None and invalidate the entry
        assert cache.get(sequence, expected_dim=1280) is None

        # 5. Subsequent get with original dimension should now also be None (entry deleted)
        assert cache.get(sequence, expected_dim) is None


def test_embedding_generator_skips_inference_on_hit():
    """Verify that EmbeddingGenerator skips model loading/inference on cache hit."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_cache.db"

        # Instantiate generator targeting the temp database
        generator = EmbeddingGenerator(
            model_name="facebook/esm2_t12_35M_UR50D",
            embedding_dim=480,
            db_path=db_path,
        )

        sequence = "MKFLILLFNILCLFPVLAADNHGVSMNAS"
        dummy_tensor = torch.randn(len(sequence), 480)

        # Pre-populate the cache so we get a cache hit immediately
        generator.sqlite_cache.set(sequence, 480, dummy_tensor)

        # Spy on the lazy _load_model method
        generator._load_model = MagicMock()

        # Call generate_single
        result = generator.generate_single(sequence, use_cache=True)

        # Assert:
        # a) result matches the dummy tensor (cache hit worked)
        assert torch.allclose(result, dummy_tensor)
        # b) _load_model was NEVER called (completely skipped transformer load/inference)
        generator._load_model.assert_not_called()
