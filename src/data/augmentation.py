"""Sequence augmentation strategies for protein structure prediction.

Provides data augmentation functions that operate on (sequence, label) pairs
while preserving per-residue alignment. Augmentations are designed to improve
model generalization without introducing biologically invalid artifacts.
"""

from __future__ import annotations

import random
from typing import Callable

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Type alias for augmentation functions
AugmentFn = Callable[[str, str], tuple[str, str]]


def random_mask(
    seq: str,
    labels: str,
    mask_prob: float = 0.05,
    mask_token: str = "X",
) -> tuple[str, str]:
    """Randomly replace residues with a mask token.

    Each residue is independently replaced with ``mask_token`` with
    probability ``mask_prob``.  The corresponding label is **not**
    modified, preserving the supervision signal.

    This augmentation simulates missing or uncertain residues and
    encourages the model to use contextual information.

    Args:
        seq: Amino acid sequence string.
        labels: Per-residue label string (same length as ``seq``).
        mask_prob: Probability of masking each residue. Must be in [0, 1].
        mask_token: Character to use as the mask. Defaults to ``"X"``.

    Returns:
        A tuple of ``(masked_sequence, labels)`` where ``labels`` is
        returned unchanged.

    Raises:
        ValueError: If ``seq`` and ``labels`` have different lengths,
            or ``mask_prob`` is outside [0, 1].

    Example:
        >>> masked_seq, lbl = random_mask("ACDEF", "HHECC", mask_prob=0.5)
        >>> len(masked_seq) == len("ACDEF")
        True
    """
    if len(seq) != len(labels):
        raise ValueError(
            f"Sequence length ({len(seq)}) and labels length ({len(labels)}) "
            f"must match."
        )
    if not (0.0 <= mask_prob <= 1.0):
        raise ValueError(
            f"mask_prob must be in [0, 1], got {mask_prob}."
        )

    if not seq:
        return (seq, labels)

    masked_chars: list[str] = []
    for char in seq:
        if random.random() < mask_prob:
            masked_chars.append(mask_token)
        else:
            masked_chars.append(char)

    return ("".join(masked_chars), labels)


def subsequence_crop(
    seq: str,
    labels: str,
    min_frac: float = 0.8,
) -> tuple[str, str]:
    """Randomly crop a contiguous subsequence preserving label alignment.

    Selects a random contiguous region of at least ``min_frac * len(seq)``
    residues.  Both the sequence and labels are cropped to the same
    region, maintaining per-residue correspondence.

    Args:
        seq: Amino acid sequence string.
        labels: Per-residue label string (same length as ``seq``).
        min_frac: Minimum fraction of the original length to retain.
            Must be in (0, 1].

    Returns:
        A tuple of ``(cropped_sequence, cropped_labels)``.

    Raises:
        ValueError: If ``seq`` and ``labels`` have different lengths,
            or ``min_frac`` is outside (0, 1].

    Example:
        >>> cropped_seq, cropped_lbl = subsequence_crop("ACDEF", "HHECC", min_frac=0.8)
        >>> len(cropped_seq) >= 4
        True
    """
    if len(seq) != len(labels):
        raise ValueError(
            f"Sequence length ({len(seq)}) and labels length ({len(labels)}) "
            f"must match."
        )
    if not (0.0 < min_frac <= 1.0):
        raise ValueError(
            f"min_frac must be in (0, 1], got {min_frac}."
        )

    n = len(seq)
    if n == 0:
        return (seq, labels)

    min_crop_len = max(1, int(n * min_frac))

    # If min_crop_len equals the sequence length, return unmodified
    if min_crop_len >= n:
        return (seq, labels)

    # Sample a crop length between min_crop_len and n (inclusive)
    crop_len = random.randint(min_crop_len, n)

    # Sample a start position
    max_start = n - crop_len
    start = random.randint(0, max_start)

    return (seq[start : start + crop_len], labels[start : start + crop_len])


def reverse_complement_protein(
    seq: str,
    labels: str,
) -> tuple[str, str]:
    """Reverse a protein sequence and its labels.

    Unlike nucleotide sequences, proteins do not have a true
    ``complement`` operation.  However, reversing both the sequence
    and the labels serves as a simple augmentation that can improve
    regularization.  The model must learn that local structural motifs
    are sequence-direction-agnostic in terms of amino acid composition.

    Args:
        seq: Amino acid sequence string.
        labels: Per-residue label string (same length as ``seq``).

    Returns:
        A tuple of ``(reversed_sequence, reversed_labels)``.

    Raises:
        ValueError: If ``seq`` and ``labels`` have different lengths.

    Example:
        >>> reverse_complement_protein("ACDEF", "HHECC")
        ('FEDCA', 'CCEHH')
    """
    if len(seq) != len(labels):
        raise ValueError(
            f"Sequence length ({len(seq)}) and labels length ({len(labels)}) "
            f"must match."
        )
    return (seq[::-1], labels[::-1])


class AugmentationPipeline:
    """Applies a sequence of augmentation functions with per-step probability.

    Each augmentation is applied independently with its configured
    probability.  Functions are applied sequentially, so the output
    of one augmentation is the input to the next.

    Attributes:
        augmentations: List of ``(function, probability)`` tuples.

    Args:
        augmentations: A list of ``(augment_fn, probability)`` tuples
            where ``augment_fn`` is a callable with signature
            ``(str, str) -> tuple[str, str]`` and ``probability`` is
            a float in [0, 1] specifying how often to apply it.

    Raises:
        ValueError: If any probability is outside [0, 1].

    Example:
        >>> pipeline = AugmentationPipeline([
        ...     (random_mask, 0.5),
        ...     (reverse_complement_protein, 0.3),
        ... ])
        >>> aug_seq, aug_lbl = pipeline("ACDEF", "HHECC")
    """

    def __init__(
        self,
        augmentations: list[tuple[AugmentFn, float]],
    ) -> None:
        for fn, prob in augmentations:
            if not (0.0 <= prob <= 1.0):
                raise ValueError(
                    f"Augmentation probability for {fn.__name__} must be "
                    f"in [0, 1], got {prob}."
                )
        self.augmentations: list[tuple[AugmentFn, float]] = augmentations
        logger.info(
            f"AugmentationPipeline initialized with "
            f"{len(augmentations)} augmentation(s): "
            + ", ".join(
                f"{fn.__name__}(p={p:.2f})" for fn, p in augmentations
            )
        )

    def __call__(self, seq: str, labels: str) -> tuple[str, str]:
        """Apply the augmentation pipeline to a sequence-label pair.

        Args:
            seq: Amino acid sequence string.
            labels: Per-residue label string (same length as ``seq``).

        Returns:
            A tuple of ``(augmented_sequence, augmented_labels)``.
        """
        current_seq = seq
        current_labels = labels

        for augment_fn, prob in self.augmentations:
            if random.random() < prob:
                current_seq, current_labels = augment_fn(
                    current_seq, current_labels
                )

        return (current_seq, current_labels)

    def __repr__(self) -> str:
        """Return a developer-friendly representation of the pipeline."""
        steps = ", ".join(
            f"{fn.__name__}(p={p:.2f})" for fn, p in self.augmentations
        )
        return f"AugmentationPipeline([{steps}])"

    def __len__(self) -> int:
        """Return the number of augmentation steps in the pipeline."""
        return len(self.augmentations)
