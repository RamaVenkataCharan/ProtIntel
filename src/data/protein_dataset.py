"""Protein secondary structure prediction dataset for ProtIntel.

Provides a PyTorch ``Dataset`` implementation that loads protein sequences
and secondary structure labels from CullPDB, CB513, RS126 (NumPy array
format), or paired FASTA + labels files.  Integrates ESM-2 tokenization,
optional embedding caching, and a custom collation function for
variable-length batching.
"""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.augmentation import AugmentationPipeline
from src.data.fasta_parser import parse_fasta
from src.data.preprocessor import (
    Q3_CLASSES,
    Q8_CLASSES,
    Q8_TO_IDX,
    clean_sequence,
    encode_q3,
    encode_q8,
    handle_nonstandard,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────
# CullPDB / CB513 / RS126 NumPy array constants
# ──────────────────────────────────────────────────────────────────────
_NPY_SEQ_LEN: int = 700
_NPY_FEATURE_DIM: int = 57
_AA_ONEHOT_START: int = 0
_AA_ONEHOT_END: int = 21        # exclusive, columns 0–20
_Q8_LABEL_START: int = 35
_Q8_LABEL_END: int = 43         # exclusive, columns 35–42 (8 Q8 classes)
_NOSEQ_COL: int = 43            # no-structure sentinel column
_AA_INDEX_TABLE: str = "ACDEFGHIKLMNPQRSTVWYX"  # 20 AAs + X (unknown)

# Q8 → Q3 reduction
_Q8_TO_Q3_REDUCTION: dict[int, int] = {
    0: 0,  # H → H
    1: 1,  # E → E
    2: 0,  # G → H
    3: 0,  # I → H
    4: 1,  # B → E
    5: 2,  # T → C
    6: 2,  # S → C
    7: 2,  # C → C
}

# ESM-2 tokenizer constants (avoid importing transformers at module level)
_ESM_PAD_TOKEN_ID: int = 1
_LABEL_IGNORE_INDEX: int = -100


def _get_esm_tokenizer() -> Any:
    """Lazily load the ESM-2 tokenizer.

    Returns:
        An ESM-2 tokenizer instance from the ``transformers`` library.

    Raises:
        ImportError: If the ``transformers`` library is not installed.
    """
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "facebook/esm2_t33_650M_UR50D"
        )
        return tokenizer
    except ImportError as exc:
        raise ImportError(
            "The 'transformers' library is required for ESM-2 tokenization. "
            "Install it with: pip install transformers"
        ) from exc


class ProteinDataset(Dataset):
    """PyTorch Dataset for protein secondary structure prediction.

    Supports three data formats:

    1. **CullPDB**: ``.npy.gz`` or ``.npy`` array of shape ``(N, 700, 57)``
       where the first 21 features are one-hot amino acid encodings and
       columns 35–42 are Q8 labels.
    2. **CB513 / RS126**: Same format as CullPDB, loaded from ``.npy.gz``
       or ``.npy`` files.
    3. **FASTA + labels**: Paired ``.fasta`` and ``.labels`` files where
       each line in the labels file corresponds to a sequence in the FASTA
       file.

    Each sample is tokenized using the ESM-2 tokenizer and optionally
    augmented during training.

    Args:
        data_path: Path to the dataset file or directory.
        split: Dataset split name (``"train"``, ``"val"``, or ``"test"``).
        config: Configuration dictionary containing preprocessing
            parameters such as ``max_seq_length``, ``nonstandard_policy``,
            and ``use_embedding_cache``.
        use_cache: Whether to look up pre-computed ESM-2 embeddings
            in the cache directory.

    Attributes:
        sequences: List of cleaned amino acid strings.
        q3_labels: List of Q3 label strings.
        q8_labels: List of Q8 label strings.
        protein_ids: List of protein identifier strings.
    """

    def __init__(
        self,
        data_path: str | Path,
        split: str,
        config: dict[str, Any],
        use_cache: bool = True,
    ) -> None:
        super().__init__()
        self.data_path: Path = Path(data_path)
        self.split: str = split
        self.config: dict[str, Any] = config
        self.use_cache: bool = use_cache

        # Parse config values
        self.max_seq_length: int = config.get("max_seq_length", 512)
        self.nonstandard_policy: str = config.get("nonstandard_policy", "mask")
        self.cache_dir: Path | None = None
        cache_path = config.get("embeddings_dir")
        if cache_path and use_cache:
            self.cache_dir = Path(cache_path)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Augmentation pipeline (only for training)
        self.augmentation: AugmentationPipeline | None = None
        aug_config = config.get("augmentation", {})
        if split == "train" and aug_config.get("enabled", False):
            self._setup_augmentation(aug_config)

        # Storage
        self.sequences: list[str] = []
        self.q3_labels: list[str] = []
        self.q8_labels: list[str] = []
        self.protein_ids: list[str] = []

        # Tokenizer (lazy-loaded on first __getitem__ call)
        self._tokenizer: Any = None

        # Load data based on file format
        self._load_data()

        logger.info(
            f"ProteinDataset [{split}] loaded {len(self.sequences)} sequences "
            f"from {self.data_path.name}"
        )

    def _setup_augmentation(self, aug_config: dict[str, Any]) -> None:
        """Configure the augmentation pipeline from a config dict.

        Args:
            aug_config: Augmentation section of the data configuration.
        """
        from src.data.augmentation import (
            random_mask,
            reverse_complement_protein,
            subsequence_crop,
        )

        aug_prob = aug_config.get("augment_prob", 0.3)
        mask_prob = aug_config.get("mask_prob", 0.05)
        crop_min_frac = aug_config.get("crop_min_frac", 0.8)

        def _mask_fn(seq: str, labels: str) -> tuple[str, str]:
            return random_mask(seq, labels, mask_prob=mask_prob)

        def _crop_fn(seq: str, labels: str) -> tuple[str, str]:
            return subsequence_crop(seq, labels, min_frac=crop_min_frac)

        _mask_fn.__name__ = "random_mask"  # type: ignore[attr-defined]
        _crop_fn.__name__ = "subsequence_crop"  # type: ignore[attr-defined]

        self.augmentation = AugmentationPipeline([
            (_mask_fn, aug_prob),
            (_crop_fn, aug_prob),
            (reverse_complement_protein, aug_prob * 0.5),
        ])

    # ──────────────────────────────────────────────────────────────────
    # Data loading
    # ──────────────────────────────────────────────────────────────────

    def _load_data(self) -> None:
        """Dispatch data loading based on file extension."""
        path = self.data_path

        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        suffix = path.suffix.lower()
        suffixes = [s.lower() for s in path.suffixes]

        if ".npy" in suffixes or ".npz" in suffixes:
            self._load_numpy_dataset(path)
        elif suffix in (".fasta", ".fa", ".faa"):
            self._load_fasta_dataset(path)
        else:
            # Try loading as numpy (CullPDB-style .npy.gz)
            if ".gz" in suffixes:
                self._load_numpy_dataset(path)
            else:
                raise ValueError(
                    f"Unsupported file format: {path.name}. "
                    f"Expected .npy, .npy.gz, .npz, .fasta, .fa, or .faa."
                )

    def _load_numpy_dataset(self, path: Path) -> None:
        """Load a CullPDB/CB513/RS126-format NumPy dataset.

        The array is expected to have shape ``(N, 700, 57)`` or be
        reshapable to that form.

        Args:
            path: Path to the ``.npy``, ``.npy.gz``, or ``.npz`` file.
        """
        # Handle .npy.gz files
        if path.name.endswith(".npy.gz"):
            import io
            with gzip.open(str(path), "rb") as f:
                data = np.load(io.BytesIO(f.read()))
        elif path.suffix == ".npz":
            with np.load(str(path)) as npz:
                keys = list(npz.keys())
                data = npz[keys[0]]
        else:
            data = np.load(str(path))

        # Reshape if necessary
        if data.ndim == 2:
            num_samples = data.shape[0] // _NPY_SEQ_LEN
            if data.shape[0] % _NPY_SEQ_LEN != 0:
                # Try reshape assuming flat rows
                total_features = data.shape[1]
                if total_features == _NPY_SEQ_LEN * _NPY_FEATURE_DIM:
                    data = data.reshape(-1, _NPY_SEQ_LEN, _NPY_FEATURE_DIM)
                else:
                    data = data.reshape(num_samples, _NPY_SEQ_LEN, _NPY_FEATURE_DIM)
            else:
                data = data.reshape(-1, _NPY_SEQ_LEN, _NPY_FEATURE_DIM)

        if data.ndim != 3 or data.shape[1] != _NPY_SEQ_LEN:
            raise ValueError(
                f"Expected array shape (N, {_NPY_SEQ_LEN}, {_NPY_FEATURE_DIM}), "
                f"got {data.shape}."
            )

        logger.info(f"Loaded NumPy array with shape {data.shape} from {path.name}")

        for idx in range(data.shape[0]):
            sample = data[idx]  # shape: (700, 57)

            # Extract amino acid sequence from one-hot encoding (columns 0-20)
            aa_onehot = sample[:, _AA_ONEHOT_START:_AA_ONEHOT_END]
            noseq = sample[:, _NOSEQ_COL]

            # Extract Q8 labels (columns 35-42)
            q8_onehot = sample[:, _Q8_LABEL_START:_Q8_LABEL_END]

            # Find actual sequence length (where noseq == 0)
            seq_mask = noseq == 0
            seq_length = int(seq_mask.sum())

            if seq_length == 0:
                continue

            # Decode amino acid sequence
            aa_indices = aa_onehot[:seq_length].argmax(axis=1)
            seq_chars: list[str] = []
            for aa_idx in aa_indices:
                if 0 <= aa_idx < len(_AA_INDEX_TABLE):
                    seq_chars.append(_AA_INDEX_TABLE[aa_idx])
                else:
                    seq_chars.append("X")
            sequence = "".join(seq_chars)

            # Decode Q8 labels
            q8_indices = q8_onehot[:seq_length].argmax(axis=1)
            q8_chars: list[str] = []
            q3_chars: list[str] = []
            for q8_idx in q8_indices:
                q8_idx_int = int(q8_idx)
                if 0 <= q8_idx_int < len(Q8_CLASSES):
                    q8_chars.append(Q8_CLASSES[q8_idx_int])
                    q3_idx = _Q8_TO_Q3_REDUCTION.get(q8_idx_int, 2)
                    q3_chars.append(Q3_CLASSES[q3_idx])
                else:
                    q8_chars.append("C")
                    q3_chars.append("C")

            q8_label_str = "".join(q8_chars)
            q3_label_str = "".join(q3_chars)

            # Clean and process sequence
            sequence = clean_sequence(sequence)
            sequence = handle_nonstandard(sequence, self.nonstandard_policy)

            # Skip very short sequences
            min_len = self.config.get("min_seq_length", 10)
            if len(sequence) < min_len:
                continue

            # Truncate to max length
            if len(sequence) > self.max_seq_length:
                sequence = sequence[: self.max_seq_length]
                q3_label_str = q3_label_str[: self.max_seq_length]
                q8_label_str = q8_label_str[: self.max_seq_length]

            self.sequences.append(sequence)
            self.q3_labels.append(q3_label_str)
            self.q8_labels.append(q8_label_str)
            self.protein_ids.append(f"{self.split}_{idx}")

    def _load_fasta_dataset(self, path: Path) -> None:
        """Load a paired FASTA + labels dataset.

        Expects a ``.fasta`` file and a companion ``.labels`` file in the
        same directory.  Each line in the labels file contains a Q8 label
        string for the corresponding FASTA sequence.

        Args:
            path: Path to the ``.fasta`` file.
        """
        records = parse_fasta(path)

        # Look for companion labels file
        labels_path = path.with_suffix(".labels")
        if not labels_path.exists():
            # Try alternative naming
            labels_path = path.parent / (path.stem + ".labels")

        label_lines: list[str] = []
        if labels_path.exists():
            with open(labels_path, "r", encoding="utf-8") as f:
                label_lines = [
                    line.strip() for line in f if line.strip()
                ]

        has_labels = len(label_lines) == len(records)
        if label_lines and not has_labels:
            logger.warning(
                f"Labels file has {len(label_lines)} entries but FASTA has "
                f"{len(records)} records. Labels will be ignored."
            )

        for idx, record in enumerate(records):
            sequence = clean_sequence(record["sequence"])
            sequence = handle_nonstandard(sequence, self.nonstandard_policy)

            min_len = self.config.get("min_seq_length", 10)
            if len(sequence) < min_len:
                continue

            if has_labels:
                q8_label = label_lines[idx][: len(sequence)]
                # Derive Q3 from Q8
                q3_chars: list[str] = []
                for ch in q8_label:
                    q8_idx = Q8_TO_IDX.get(ch, 7)
                    q3_idx = _Q8_TO_Q3_REDUCTION.get(q8_idx, 2)
                    q3_chars.append(Q3_CLASSES[q3_idx])
                q3_label = "".join(q3_chars)
            else:
                # Placeholder labels (all coil)
                q8_label = "C" * len(sequence)
                q3_label = "C" * len(sequence)

            # Truncate
            if len(sequence) > self.max_seq_length:
                sequence = sequence[: self.max_seq_length]
                q3_label = q3_label[: self.max_seq_length]
                q8_label = q8_label[: self.max_seq_length]

            self.sequences.append(sequence)
            self.q3_labels.append(q3_label)
            self.q8_labels.append(q8_label)
            self.protein_ids.append(record["id"])

    # ──────────────────────────────────────────────────────────────────
    # Tokenizer property
    # ──────────────────────────────────────────────────────────────────

    @property
    def tokenizer(self) -> Any:
        """Lazily load and cache the ESM-2 tokenizer.

        Returns:
            The ESM-2 tokenizer instance.
        """
        if self._tokenizer is None:
            self._tokenizer = _get_esm_tokenizer()
        return self._tokenizer

    # ──────────────────────────────────────────────────────────────────
    # Dataset interface
    # ──────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample by index.

        Returns:
            A dictionary containing:
                - ``"input_ids"``: ``torch.LongTensor`` of ESM token IDs,
                  shape ``(L,)``.
                - ``"attention_mask"``: ``torch.LongTensor`` of 1s for real
                  tokens and 0s for padding, shape ``(L,)``.
                - ``"q3_labels"``: ``torch.LongTensor`` of Q3 class indices,
                  shape ``(L,)``.
                - ``"q8_labels"``: ``torch.LongTensor`` of Q8 class indices,
                  shape ``(L,)``.
                - ``"sequence"``: Raw amino acid string.
                - ``"protein_id"``: Identifier string.
                - ``"length"``: Actual sequence length before padding.
                - ``"cached_embedding"`` (optional): Pre-computed embedding
                  tensor if found in the cache directory.
        """
        sequence = self.sequences[idx]
        q3_label_str = self.q3_labels[idx]
        q8_label_str = self.q8_labels[idx]
        protein_id = self.protein_ids[idx]

        # Apply augmentation during training
        if self.augmentation is not None:
            sequence, q3_label_str = self.augmentation(sequence, q3_label_str)
            # Re-derive Q8 labels if augmentation changed length
            if len(q8_label_str) != len(sequence):
                q8_label_str = q8_label_str[: len(sequence)]
            # If reversal happened, also reverse Q8
            if len(q8_label_str) == len(sequence):
                _, q8_label_str = self.augmentation(
                    self.sequences[idx], self.q8_labels[idx]
                )
                q8_label_str = q8_label_str[: len(sequence)]

        seq_length = len(sequence)

        # Tokenize with ESM-2
        encoding = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_seq_length + 2,  # +2 for BOS/EOS tokens
        )
        input_ids = encoding["input_ids"].squeeze(0)  # (L,)
        attention_mask = encoding["attention_mask"].squeeze(0)  # (L,)

        # Encode labels
        q3_indices = encode_q3(q3_label_str)
        q8_indices = encode_q8(q8_label_str)

        q3_labels = torch.tensor(q3_indices, dtype=torch.long)
        q8_labels = torch.tensor(q8_indices, dtype=torch.long)

        result: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "q3_labels": q3_labels,
            "q8_labels": q8_labels,
            "sequence": sequence,
            "protein_id": protein_id,
            "length": seq_length,
        }

        # Check embedding cache
        if self.cache_dir is not None and self.use_cache:
            from src.utils.io_utils import compute_sequence_hash
            cache_key = compute_sequence_hash(sequence)
            cache_file = self.cache_dir / f"{cache_key}.pt"
            if cache_file.exists():
                try:
                    cached_emb = torch.load(
                        str(cache_file), map_location="cpu", weights_only=True
                    )
                    result["cached_embedding"] = cached_emb
                except Exception as e:
                    logger.warning(
                        f"Failed to load cached embedding for {protein_id}: {e}"
                    )

        return result


def collate_fn(
    batch: list[dict[str, Any]],
) -> dict[str, Any]:
    """Custom collation function for variable-length protein sequences.

    Sorts the batch by sequence length in descending order and pads
    ``input_ids``, ``attention_mask``, ``q3_labels``, and ``q8_labels``
    to the maximum length in the batch.

    Padding conventions:
        - ``input_ids``: padded with ``1`` (ESM-2 pad token ID).
        - ``attention_mask``: padded with ``0``.
        - ``q3_labels`` and ``q8_labels``: padded with ``-100``
          (ignored by ``torch.nn.CrossEntropyLoss``).

    Args:
        batch: A list of sample dictionaries as returned by
            :meth:`ProteinDataset.__getitem__`.

    Returns:
        A dictionary with batched tensors:
            - ``"input_ids"``: ``torch.LongTensor`` of shape ``(B, L_max)``.
            - ``"attention_mask"``: ``torch.LongTensor`` of shape ``(B, L_max)``.
            - ``"q3_labels"``: ``torch.LongTensor`` of shape ``(B, L_max)``.
            - ``"q8_labels"``: ``torch.LongTensor`` of shape ``(B, L_max)``.
            - ``"sequences"``: List of raw amino acid strings.
            - ``"protein_ids"``: List of identifier strings.
            - ``"lengths"``: ``torch.LongTensor`` of shape ``(B,)``.
            - ``"cached_embeddings"`` (optional): Stacked embedding tensor
              if all samples have cached embeddings.

    Raises:
        ValueError: If the batch is empty.
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    # Sort by length descending (for pack_padded_sequence compatibility)
    batch = sorted(batch, key=lambda x: x["length"], reverse=True)

    # Find max lengths for padding
    max_input_len = max(sample["input_ids"].size(0) for sample in batch)
    max_label_len = max(sample["q3_labels"].size(0) for sample in batch)

    batch_size = len(batch)

    # Allocate padded tensors
    input_ids = torch.full(
        (batch_size, max_input_len), _ESM_PAD_TOKEN_ID, dtype=torch.long
    )
    attention_mask = torch.zeros(batch_size, max_input_len, dtype=torch.long)
    q3_labels = torch.full(
        (batch_size, max_label_len), _LABEL_IGNORE_INDEX, dtype=torch.long
    )
    q8_labels = torch.full(
        (batch_size, max_label_len), _LABEL_IGNORE_INDEX, dtype=torch.long
    )

    sequences: list[str] = []
    protein_ids: list[str] = []
    lengths: list[int] = []
    cached_embeddings: list[torch.Tensor] = []
    all_have_embeddings = True

    for i, sample in enumerate(batch):
        # Copy input_ids
        input_len = sample["input_ids"].size(0)
        input_ids[i, :input_len] = sample["input_ids"]
        attention_mask[i, :input_len] = sample["attention_mask"]

        # Copy labels
        label_len = sample["q3_labels"].size(0)
        q3_labels[i, :label_len] = sample["q3_labels"]
        q8_labels[i, :label_len] = sample["q8_labels"]

        sequences.append(sample["sequence"])
        protein_ids.append(sample["protein_id"])
        lengths.append(sample["length"])

        if "cached_embedding" in sample:
            cached_embeddings.append(sample["cached_embedding"])
        else:
            all_have_embeddings = False

    result: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "q3_labels": q3_labels,
        "q8_labels": q8_labels,
        "sequences": sequences,
        "sequence": sequences,       # Alias for trainer.py
        "protein_ids": protein_ids,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "seq_length": torch.tensor(lengths, dtype=torch.long), # Alias for trainer.py
    }

    if all_have_embeddings and cached_embeddings:
        # Pad and stack embeddings
        max_emb_len = max(emb.size(0) for emb in cached_embeddings)
        emb_dim = cached_embeddings[0].size(-1)
        stacked = torch.zeros(batch_size, max_emb_len, emb_dim)
        for i, emb in enumerate(cached_embeddings):
            stacked[i, : emb.size(0)] = emb
        result["cached_embeddings"] = stacked
        result["embeddings"] = stacked # Alias for trainer.py

    return result
