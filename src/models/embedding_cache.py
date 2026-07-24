"""SQLite-based cache for protein sequence embeddings.

Provides a key-value store to cache ESM-2 embeddings to disk, preventing
costly redundant model forward passes.
"""

from __future__ import annotations

import io
import sqlite3
from pathlib import Path
from typing import Optional

import torch

from src.utils.io_utils import compute_sequence_hash
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SQLiteEmbeddingCache:
    """SQLite-based database cache for protein sequence embeddings.

    Caches embedding tensors of shape (L, embedding_dim) using a SHA-256
    hash of the sequence as the primary key. Performs integrity and dimension
    checks on retrieval to prevent versioning issues.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the cache database and tables if they do not exist."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS embedding_cache (
                        sequence_hash TEXT PRIMARY KEY,
                        sequence TEXT,
                        embedding_dim INTEGER,
                        length INTEGER,
                        embedding BLOB
                    )
                """)
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize SQLite embedding cache database: {e}")

    def get(self, sequence: str, expected_dim: int) -> Optional[torch.Tensor]:
        """Retrieve the cached embedding tensor for a sequence if available.

        Verifies that the retrieved embedding dimension matches the expected
        dimension. Invalidates/deletes entries upon a dimension mismatch.

        Args:
            sequence: The raw amino acid sequence.
            expected_dim: The expected embedding dimension (e.g. 480).

        Returns:
            The cached embedding tensor of shape (L, expected_dim), or None if
            uncached or dimension mismatched.
        """
        seq_hash = compute_sequence_hash(sequence)
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT embedding, embedding_dim FROM embedding_cache WHERE sequence_hash = ?",
                    (seq_hash,),
                )
                row = cursor.fetchone()
                if row is None:
                    return None

                blob_data, dim = row
                if dim != expected_dim:
                    logger.warning(
                        f"Dimension mismatch in cache for sequence {seq_hash[:8]}: "
                        f"expected {expected_dim}, got {dim}. Invalidate cache entry."
                    )
                    cursor.execute(
                        "DELETE FROM embedding_cache WHERE sequence_hash = ?",
                        (seq_hash,),
                    )
                    conn.commit()
                    return None

                # Deserialize PyTorch tensor
                buffer = io.BytesIO(blob_data)
                tensor = torch.load(buffer, map_location="cpu", weights_only=True)
                logger.debug(f"Loaded cached embeddings for sequence {seq_hash[:8]} (dim={dim})")
                return tensor

        except Exception as e:
            logger.error(f"Error reading from SQLite embedding cache: {e}. Falling back to live inference.")
            return None

    def set(self, sequence: str, expected_dim: int, tensor: torch.Tensor) -> None:
        """Cache the embedding tensor for a sequence.

        Args:
            sequence: The raw amino acid sequence.
            expected_dim: The embedding dimension.
            tensor: The embedding tensor of shape (L, expected_dim).
        """
        seq_hash = compute_sequence_hash(sequence)
        try:
            # Serialize PyTorch tensor
            buffer = io.BytesIO()
            torch.save(tensor.cpu(), buffer)
            blob_data = buffer.getvalue()

            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO embedding_cache "
                    "(sequence_hash, sequence, embedding_dim, length, embedding) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (seq_hash, sequence, expected_dim, len(sequence), blob_data),
                )
                conn.commit()
                logger.debug(f"Cached embeddings for sequence {seq_hash[:8]} (dim={expected_dim})")

        except Exception as e:
            logger.error(f"Failed to save embeddings to SQLite cache: {e}")
