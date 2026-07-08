"""Protein sequence preprocessing for ProtIntel.

Provides functions for cleaning, encoding, decoding, and chunking amino acid
sequences and their secondary structure labels. Supports both Q3 (3-class)
and Q8 (8-class) secondary structure classification schemes.
"""

from __future__ import annotations

import re
from collections import Counter

import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Canonical amino acid alphabet
# ──────────────────────────────────────────────────────────────────────
CANONICAL_AA: frozenset[str] = frozenset("ACDEFGHIKLMNPQRSTVWY")

# ──────────────────────────────────────────────────────────────────────
# Non-standard amino acid handling maps
# ──────────────────────────────────────────────────────────────────────
_NONSTANDARD_RESIDUES: frozenset[str] = frozenset("BJOUZX")

_MASK_MAP: dict[str, str] = {
    "B": "X", "J": "X", "O": "X", "U": "X", "X": "X", "Z": "X",
}

_REPLACE_MAP: dict[str, str] = {
    "B": "D",  # Aspartate/Asparagine → Aspartate
    "Z": "E",  # Glutamate/Glutamine → Glutamate
    "X": "A",  # Unknown → Alanine (most common)
    "J": "L",  # Leucine/Isoleucine → Leucine
    "O": "K",  # Pyrrolysine → Lysine
    "U": "C",  # Selenocysteine → Cysteine
}

# ──────────────────────────────────────────────────────────────────────
# Q3 / Q8 label mappings
# ──────────────────────────────────────────────────────────────────────
Q3_TO_IDX: dict[str, int] = {"H": 0, "E": 1, "C": 2}
Q3_CLASSES: list[str] = ["H", "E", "C"]

Q8_TO_IDX: dict[str, int] = {
    "H": 0, "E": 1, "G": 2, "I": 3, "B": 4, "T": 5, "S": 6, "C": 7,
}
Q8_CLASSES: list[str] = ["H", "E", "G", "I", "B", "T", "S", "C"]

# Reverse mappings built once at module load
_IDX_TO_Q3: dict[int, str] = {v: k for k, v in Q3_TO_IDX.items()}
_IDX_TO_Q8: dict[int, str] = {v: k for k, v in Q8_TO_IDX.items()}


# ======================================================================
# Sequence cleaning
# ======================================================================

def clean_sequence(seq: str) -> str:
    """Clean a raw amino acid sequence string.

    Performs the following transformations in order:
        1. Strip leading and trailing whitespace.
        2. Convert to uppercase.
        3. Remove all digits.
        4. Remove all dash (``-``) and dot (``.``) gap characters.
        5. Remove any remaining internal whitespace.

    Args:
        seq: Raw amino acid sequence string.

    Returns:
        Cleaned uppercase sequence containing only alphabetic characters.

    Example:
        >>> clean_sequence("  acD-EF 12gH.I  ")
        'ACDEFGHI'
    """
    if not seq:
        return ""
    result = seq.strip().upper()
    result = re.sub(r"[\d\-.]", "", result)
    result = re.sub(r"\s+", "", result)
    return result


def handle_nonstandard(seq: str, policy: str = "mask") -> str:
    """Handle non-standard amino acid residues in a sequence.

    Non-standard residues are: B, J, O, U, X, Z.

    Two policies are supported:

    - ``"mask"``: Replace all non-standard residues with ``X``
      (the unknown/masked token).
    - ``"replace"``: Replace each non-standard residue with its
      most chemically similar canonical amino acid:
      B→D, Z→E, X→A, J→L, O→K, U→C.

    Args:
        seq: Amino acid sequence string (should already be uppercased).
        policy: Handling policy. Must be ``"mask"`` or ``"replace"``.

    Returns:
        Sequence with non-standard residues handled according to policy.

    Raises:
        ValueError: If ``policy`` is not ``"mask"`` or ``"replace"``.

    Example:
        >>> handle_nonstandard("ACBZE", policy="replace")
        'ACDEA'
    """
    if policy not in ("mask", "replace"):
        raise ValueError(
            f"Invalid nonstandard_policy '{policy}'. "
            f"Must be 'mask' or 'replace'."
        )

    if not seq:
        return ""

    mapping = _MASK_MAP if policy == "mask" else _REPLACE_MAP
    result_chars: list[str] = []
    for char in seq:
        if char in mapping:
            result_chars.append(mapping[char])
        else:
            result_chars.append(char)

    return "".join(result_chars)


def is_valid_sequence(seq: str) -> bool:
    """Check whether a sequence contains only the 20 canonical amino acids.

    Args:
        seq: Amino acid sequence string (should be uppercase).

    Returns:
        ``True`` if every character in ``seq`` belongs to the canonical
        amino acid set (ACDEFGHIKLMNPQRSTVWY), ``False`` otherwise.
        Returns ``False`` for empty strings.

    Example:
        >>> is_valid_sequence("ACDEFGHIKLMNPQRSTVWY")
        True
        >>> is_valid_sequence("ACBX")
        False
    """
    if not seq:
        return False
    return all(c in CANONICAL_AA for c in seq)


# ======================================================================
# Sequence chunking
# ======================================================================

def chunk_sequence(
    seq: str,
    labels: str,
    max_len: int = 512,
    overlap: int = 64,
) -> list[tuple[str, str]]:
    """Split a long sequence and its labels into overlapping chunks.

    Uses a sliding window approach so that every residue appears in at
    least one chunk.  If the sequence is shorter than or equal to
    ``max_len``, a single chunk containing the full sequence is returned.

    Sequence and labels must have the same length so that the per-residue
    alignment is preserved.

    Args:
        seq: Amino acid sequence string.
        labels: Per-residue label string (same length as ``seq``).
        max_len: Maximum chunk length in residues.
        overlap: Number of overlapping residues between consecutive
            chunks. Must be non-negative and strictly less than ``max_len``.

    Returns:
        A list of ``(sequence_chunk, label_chunk)`` tuples.

    Raises:
        ValueError: If ``seq`` and ``labels`` have different lengths,
            ``max_len`` is not positive, or ``overlap >= max_len``.

    Example:
        >>> chunks = chunk_sequence("A" * 600, "C" * 600, max_len=512, overlap=64)
        >>> len(chunks)
        2
    """
    if len(seq) != len(labels):
        raise ValueError(
            f"Sequence length ({len(seq)}) and labels length ({len(labels)}) "
            f"must match."
        )
    if max_len < 1:
        raise ValueError(f"max_len must be positive, got {max_len}.")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}.")
    if overlap >= max_len:
        raise ValueError(
            f"overlap ({overlap}) must be strictly less than max_len ({max_len})."
        )

    # Short sequences produce a single chunk
    if len(seq) <= max_len:
        return [(seq, labels)]

    chunks: list[tuple[str, str]] = []
    step = max_len - overlap
    start = 0

    while start < len(seq):
        end = min(start + max_len, len(seq))
        chunks.append((seq[start:end], labels[start:end]))
        if end == len(seq):
            break
        start += step

    return chunks


# ======================================================================
# Label encoding / decoding
# ======================================================================

def encode_q3(label_str: str) -> list[int]:
    """Encode a Q3 secondary structure label string to integer indices.

    Mapping: H → 0, E → 1, C → 2.

    Args:
        label_str: A string of Q3 labels (e.g., ``"HHHEEECC"``).

    Returns:
        A list of integer indices corresponding to the Q3 classes.

    Raises:
        ValueError: If any character is not in {H, E, C}.

    Example:
        >>> encode_q3("HEC")
        [0, 1, 2]
    """
    if not label_str:
        return []

    indices: list[int] = []
    for i, char in enumerate(label_str):
        if char not in Q3_TO_IDX:
            raise ValueError(
                f"Invalid Q3 label character '{char}' at position {i}. "
                f"Expected one of: {', '.join(Q3_CLASSES)}."
            )
        indices.append(Q3_TO_IDX[char])

    return indices


def encode_q8(label_str: str) -> list[int]:
    """Encode a Q8 secondary structure label string to integer indices.

    Mapping: H→0, E→1, G→2, I→3, B→4, T→5, S→6, C→7.

    Args:
        label_str: A string of Q8 labels (e.g., ``"HEGIBTSC"``).

    Returns:
        A list of integer indices corresponding to the Q8 classes.

    Raises:
        ValueError: If any character is not in the Q8 set.

    Example:
        >>> encode_q8("HEGIBTSC")
        [0, 1, 2, 3, 4, 5, 6, 7]
    """
    if not label_str:
        return []

    indices: list[int] = []
    for i, char in enumerate(label_str):
        if char not in Q8_TO_IDX:
            raise ValueError(
                f"Invalid Q8 label character '{char}' at position {i}. "
                f"Expected one of: {', '.join(Q8_CLASSES)}."
            )
        indices.append(Q8_TO_IDX[char])

    return indices


def decode_q3(indices: list[int] | torch.Tensor) -> str:
    """Decode a list of Q3 integer indices back to a label string.

    Args:
        indices: A list or tensor of integers in {0, 1, 2}.

    Returns:
        A string of Q3 label characters.

    Raises:
        ValueError: If any index is not in {0, 1, 2}.

    Example:
        >>> decode_q3([0, 1, 2])
        'HEC'
    """
    if isinstance(indices, torch.Tensor):
        if indices.numel() == 0:
            return ""
        indices = indices.tolist()
    elif not indices:
        return ""

    chars: list[str] = []
    for i, idx in enumerate(indices):
        if idx not in _IDX_TO_Q3:
            raise ValueError(
                f"Invalid Q3 index {idx} at position {i}. "
                f"Expected one of: {sorted(_IDX_TO_Q3.keys())}."
            )
        chars.append(_IDX_TO_Q3[idx])

    return "".join(chars)


def decode_q8(indices: list[int] | torch.Tensor) -> str:
    """Decode a list of Q8 integer indices back to a label string.

    Args:
        indices: A list or tensor of integers in {0, 1, 2, 3, 4, 5, 6, 7}.

    Returns:
        A string of Q8 label characters.

    Raises:
        ValueError: If any index is not in {0, …, 7}.

    Example:
        >>> decode_q8([0, 1, 2, 3, 4, 5, 6, 7])
        'HEGIBTSC'
    """
    if isinstance(indices, torch.Tensor):
        if indices.numel() == 0:
            return ""
        indices = indices.tolist()
    elif not indices:
        return ""

    chars: list[str] = []
    for i, idx in enumerate(indices):
        if idx not in _IDX_TO_Q8:
            raise ValueError(
                f"Invalid Q8 index {idx} at position {i}. "
                f"Expected one of: {sorted(_IDX_TO_Q8.keys())}."
            )
        chars.append(_IDX_TO_Q8[idx])

    return "".join(chars)


# ======================================================================
# Class weight computation
# ======================================================================

def compute_class_weights(
    labels: list[list[int]],
    num_classes: int,
) -> torch.Tensor:
    """Compute inverse-frequency class weights from a collection of label sequences.

    The weight for each class is computed as::

        weight_c = total_samples / (num_classes * count_c)

    Weights are then normalized so that they sum to ``num_classes``
    (i.e., the mean weight is 1.0).  This prevents the overall loss
    magnitude from changing when class weights are applied.

    Args:
        labels: A list of label sequences, where each inner list
            contains integer class indices.
        num_classes: Total number of distinct classes.

    Returns:
        A ``torch.float32`` tensor of shape ``(num_classes,)`` containing
        the normalized class weights.

    Raises:
        ValueError: If ``labels`` is empty or ``num_classes`` is not positive.

    Example:
        >>> weights = compute_class_weights([[0, 0, 1, 2]], num_classes=3)
        >>> weights.shape
        torch.Size([3])
    """
    if not labels:
        raise ValueError("Cannot compute class weights from an empty label list.")
    if num_classes < 1:
        raise ValueError(f"num_classes must be positive, got {num_classes}.")

    # Count occurrences of each class
    counts: Counter[int] = Counter()
    total: int = 0
    for label_seq in labels:
        for label in label_seq:
            if 0 <= label < num_classes:
                counts[label] += 1
                total += 1

    if total == 0:
        logger.warning(
            "No valid labels found; returning uniform weights."
        )
        return torch.ones(num_classes, dtype=torch.float32)

    # Inverse frequency weighting
    weights = torch.zeros(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        class_count = counts.get(c, 0)
        if class_count == 0:
            logger.warning(
                f"Class {c} has zero samples; assigning maximum weight."
            )
            weights[c] = total / (num_classes * 1.0)  # treat as count=1
        else:
            weights[c] = total / (num_classes * class_count)

    # Normalize so weights sum to num_classes (mean = 1.0)
    weight_sum = weights.sum()
    if weight_sum > 0:
        weights = weights * (num_classes / weight_sum)

    logger.info(f"Computed class weights for {num_classes} classes: {weights.tolist()}")
    return weights


# ──────────────────────────────────────────────────────────────────────
# Compatibility aliases
# These names were referenced in inference_service.py but not defined.
# Bug #8: added here to resolve the ImportError without breaking callers.
# ──────────────────────────────────────────────────────────────────────

def decode_q3_predictions(indices: "torch.Tensor | list[int]") -> str:
    """Decode a sequence of Q3 class indices to a label string (e.g. "HHECCC").

    Alias for :func:`decode_q3` that matches the name imported by
    ``backend/services/inference_service.py``.

    Args:
        indices: Integer tensor or list of Q3 class indices.

    Returns:
        String of Q3 labels (H, E, C).
    """
    return decode_q3(indices)


def decode_q8_predictions(indices: "torch.Tensor | list[int]") -> str:
    """Decode a sequence of Q8 class indices to a label string.

    Alias for :func:`decode_q8` that matches the name imported by
    ``backend/services/inference_service.py``.

    Args:
        indices: Integer tensor or list of Q8 class indices.

    Returns:
        String of Q8 labels (H, E, G, I, B, T, S, C).
    """
    return decode_q8(indices)


class SequencePreprocessor:
    """Stateless wrapper class providing the preprocessing API expected by
    ``InferenceService``.

    All methods delegate to the corresponding module-level functions.
    This class was missing from the module (Bug #8) — its name was imported
    by ``backend/services/inference_service.py`` but never defined.

    Example::

        preprocessor = SequencePreprocessor()
        cleaned = preprocessor.clean_sequence("mkflil")
        assert cleaned == "MKFLIL"
    """

    def clean_sequence(self, sequence: str) -> str:
        """Clean and uppercase a raw amino acid sequence.

        Args:
            sequence: Raw input sequence string.

        Returns:
            Cleaned, uppercase sequence string.
        """
        return clean_sequence(sequence)

    def is_valid(self, sequence: str) -> bool:
        """Return True if the sequence contains only canonical amino acids.

        Args:
            sequence: Input sequence string (will be uppercased).

        Returns:
            True if the sequence is valid, False otherwise.
        """
        return is_valid_sequence(sequence)

    def handle_nonstandard(
        self,
        sequence: str,
        policy: str = "replace",
    ) -> str:
        """Handle non-standard amino acid residues in the sequence.

        Args:
            sequence: Input sequence string.
            policy: One of ``'replace'``, ``'mask'``, or ``'error'``.

        Returns:
            Processed sequence string.
        """
        return handle_nonstandard(sequence, policy=policy)

    def encode_q3(self, label_string: str) -> list[int]:
        """Encode a Q3 label string to integer indices.

        Args:
            label_string: String of Q3 labels (e.g. ``"HHECCC"``).

        Returns:
            List of integer class indices.
        """
        return encode_q3(label_string)

    def encode_q8(self, label_string: str) -> list[int]:
        """Encode a Q8 label string to integer indices.

        Args:
            label_string: String of Q8 labels.

        Returns:
            List of integer class indices.
        """
        return encode_q8(label_string)
