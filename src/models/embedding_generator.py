"""ESM-2 embedding generator for protein sequences.

Wraps the facebook/esm2_t12_35M_UR50D model to produce per-residue
embeddings of shape (L, 480). Supports frozen mode, partial fine-tuning,
batch processing, and disk caching.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, EsmModel, EsmTokenizer

from src.utils.io_utils import compute_sequence_hash, ensure_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingGenerator(nn.Module):
    """Generates per-residue protein embeddings using ESM-2.

    Wraps the ESM-2 protein language model to produce context-aware
    embeddings for each amino acid residue. Supports frozen inference,
    partial fine-tuning of the last N transformer layers, and disk
    caching of pre-computed embeddings.

    Args:
        model_name: HuggingFace model identifier for ESM-2.
        embedding_dim: Expected embedding dimension (1280 for 650M model).
        freeze: If True, freeze all ESM-2 parameters (no gradients).
        finetune_last_n_layers: Number of last transformer layers to unfreeze.
            Only effective when ``freeze`` is True. Set to 0 for fully frozen.
        cache_dir: Optional directory for caching embeddings to disk.
        device: Device to load the model on.
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t12_35M_UR50D",
        embedding_dim: int = 480,
        freeze: bool = True,
        finetune_last_n_layers: int = 0,
        cache_dir: Optional[str | Path] = None,
        device: str = "cpu",
        db_path: Optional[str | Path] = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.freeze = freeze
        self.finetune_last_n_layers = finetune_last_n_layers
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            self.cache_dir = project_root / "datasets" / "processed" / "embeddings"
        
        self.db_path = Path(db_path) if db_path else self.cache_dir / "embedding_cache.db"
        self.device_str = device

        from src.models.embedding_cache import SQLiteEmbeddingCache
        self.sqlite_cache = SQLiteEmbeddingCache(self.db_path)

        self._model: Optional[EsmModel] = None
        self._tokenizer: Optional[EsmTokenizer] = None
        self._loaded = False

    def _load_model(self) -> None:
        """Load the ESM-2 model and tokenizer from HuggingFace.

        Called lazily on first use to avoid downloading weights
        during unit tests.
        """
        if self._loaded:
            return

        logger.info(f"Loading ESM-2 model: {self.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self.add_module("esm2_model", self._model)
        self._model = self._model.to(self.device_str)

        # Freeze parameters
        if self.freeze:
            for param in self._model.parameters():
                param.requires_grad = False
            logger.info("ESM-2 parameters frozen (no gradients)")

            # Optionally unfreeze last N layers
            if self.finetune_last_n_layers > 0:
                encoder_layers = self._model.encoder.layer
                total_layers = len(encoder_layers)
                start_unfreeze = total_layers - self.finetune_last_n_layers

                for i in range(start_unfreeze, total_layers):
                    for param in encoder_layers[i].parameters():
                        param.requires_grad = True

                logger.info(
                    f"Unfroze last {self.finetune_last_n_layers} of "
                    f"{total_layers} transformer layers for fine-tuning"
                )

        if self.cache_dir:
            ensure_dir(self.cache_dir)

        total_params = sum(p.numel() for p in self._model.parameters())
        trainable_params = sum(
            p.numel() for p in self._model.parameters() if p.requires_grad
        )
        logger.info(
            f"ESM-2 loaded: {total_params / 1e6:.1f}M total params, "
            f"{trainable_params / 1e6:.1f}M trainable"
        )
        self._loaded = True

    @property
    def model(self) -> EsmModel:
        """Access the underlying ESM-2 model, loading if necessary.

        Returns:
            The loaded EsmModel instance.
        """
        self._load_model()
        assert self._model is not None
        return self._model

    @property
    def tokenizer(self) -> EsmTokenizer:
        """Access the ESM-2 tokenizer, loading if necessary.

        Returns:
            The loaded EsmTokenizer instance.
        """
        self._load_model()
        assert self._tokenizer is not None
        return self._tokenizer

    def _try_load_cached(self, sequence: str) -> Optional[torch.Tensor]:
        """Attempt to load cached embeddings for a sequence from SQLite.

        Args:
            sequence: The amino acid sequence string.

        Returns:
            Cached embedding tensor of shape (L, 480), or None if
            no cache exists.
        """
        return self.sqlite_cache.get(sequence, self.embedding_dim)

    def _save_to_cache(self, sequence: str, embeddings: torch.Tensor) -> None:
        """Save embeddings to the SQLite cache.

        Args:
            sequence: The amino acid sequence string.
            embeddings: Embedding tensor of shape (L, 480).
        """
        self.sqlite_cache.set(sequence, self.embedding_dim, embeddings)

    def generate_single(
        self,
        sequence: str,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Generate embeddings for a single protein sequence.

        Args:
            sequence: Amino acid sequence string (uppercase letters).
            use_cache: Whether to check disk cache first.

        Returns:
            Per-residue embedding tensor of shape (L, embedding_dim) where L
            is the sequence length (BOS/EOS tokens stripped).
        """
        # Disable cache if fine-tuning is active in config
        if self.finetune_last_n_layers > 0 or not self.freeze:
            use_cache = False

        # Check cache
        if use_cache:
            cached = self._try_load_cached(sequence)
            if cached is not None:
                return cached

        self._load_model()

        # Double check if any parameters are trainable after loading
        any_trainable = any(p.requires_grad for p in self.parameters())
        if any_trainable:
            use_cache = False

        # Tokenize
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=1024,
        )
        inputs = {k: v.to(self.device_str) for k, v in inputs.items()}

        # Forward pass
        run_no_grad = (not self.training) or (not any_trainable)
        with torch.no_grad() if run_no_grad else torch.enable_grad():
            outputs = self.model(**inputs)

        # Extract last hidden state and strip BOS/EOS tokens
        hidden_states = outputs.last_hidden_state  # (1, L+2, 1280)
        embeddings = hidden_states[0, 1:-1, :]  # (L, 1280) — strip BOS/EOS

        # Cache to disk
        if use_cache:
            self._save_to_cache(sequence, embeddings)

        return embeddings

    def generate_batch(
        self,
        sequences: list[str],
        use_cache: bool = True,
        max_batch_size: int = 8,
    ) -> list[torch.Tensor]:
        """Generate embeddings for a batch of protein sequences.

        Processes sequences in mini-batches to manage memory. Returns
        embeddings as a list of tensors (varying lengths).

        Args:
            sequences: List of amino acid sequence strings.
            use_cache: Whether to use disk cache.
            max_batch_size: Maximum sequences per mini-batch.

        Returns:
            List of embedding tensors, each of shape (L_i, embedding_dim).
        """
        # Disable cache if fine-tuning is active in config
        if self.finetune_last_n_layers > 0 or not self.freeze:
            use_cache = False

        all_embeddings: list[torch.Tensor] = []
        uncached_indices: list[int] = []
        uncached_sequences: list[str] = []

        # Check cache first
        for i, seq in enumerate(sequences):
            if use_cache:
                cached = self._try_load_cached(seq)
                if cached is not None:
                    all_embeddings.append(cached)
                    continue

            uncached_indices.append(i)
            uncached_sequences.append(seq)
            all_embeddings.append(torch.empty(0))  # Placeholder

        if not uncached_sequences:
            return all_embeddings

        self._load_model()

        # Double check if any parameters are trainable after loading
        any_trainable = any(p.requires_grad for p in self.parameters())
        if any_trainable:
            use_cache = False

        # Process uncached sequences in mini-batches
        for batch_start in range(0, len(uncached_sequences), max_batch_size):
            batch_end = min(batch_start + max_batch_size, len(uncached_sequences))
            batch_seqs = uncached_sequences[batch_start:batch_end]

            inputs = self.tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            )
            inputs = {k: v.to(self.device_str) for k, v in inputs.items()}

            run_no_grad = (not self.training) or (not any_trainable)
            with torch.no_grad() if run_no_grad else torch.enable_grad():
                outputs = self.model(**inputs)

            hidden_states = outputs.last_hidden_state  # (B, L+2, 1280)

            for j, seq in enumerate(batch_seqs):
                seq_len = len(seq)
                # Strip BOS/EOS: positions 1 to seq_len+1
                emb = hidden_states[j, 1 : seq_len + 1, :]  # (seq_len, 1280)

                global_idx = uncached_indices[batch_start + j]
                all_embeddings[global_idx] = emb if any_trainable else emb.cpu()

                if use_cache:
                    self._save_to_cache(seq, emb)

        logger.info(
            f"Generated embeddings for {len(sequences)} sequences "
            f"({len(uncached_sequences)} computed, "
            f"{len(sequences) - len(uncached_sequences)} cached)"
        )
        return all_embeddings

    def forward(
        self,
        sequences: list[str],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for integration into the ProtIntelModel.

        When used as part of the full model pipeline, this method
        generates embeddings and pads them to uniform length for
        batched processing.

        Args:
            sequences: List of amino acid sequence strings.
            attention_mask: Optional attention mask tensor of shape (B, L).
                Used to determine the padding length.

        Returns:
            Padded embedding tensor of shape (B, max_len, embedding_dim).
        """
        embeddings = self.generate_batch(sequences, use_cache=True)

        # Determine max length from embeddings (BOS/EOS stripped)
        max_len = max(emb.size(0) for emb in embeddings)

        # Pad to uniform length on correct device/dtype
        batch_size = len(embeddings)
        device = embeddings[0].device
        dtype = embeddings[0].dtype
        padded = torch.zeros(batch_size, max_len, self.embedding_dim, device=device, dtype=dtype)

        for i, emb in enumerate(embeddings):
            seq_len = min(emb.size(0), max_len)
            padded[i, :seq_len, :] = emb[:seq_len]

        return padded

    def get_output_dim(self) -> int:
        """Return the output embedding dimension.

        Returns:
            The embedding dimension (480 for esm2_t12_35M_UR50D).
        """
        return self.embedding_dim
