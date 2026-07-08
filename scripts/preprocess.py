"""Preprocess downloaded datasets for ProtIntel training.

Loads raw CullPDB and CB513 numpy files, validates them, and
optionally splits training data into train/validation subsets.

Usage:
    python scripts/preprocess.py
    python scripts/preprocess.py --raw-dir datasets/raw --output-dir datasets/processed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.utils.io_utils import ensure_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Constants for CullPDB format
CULLPDB_SEQ_LEN = 700
CULLPDB_NUM_FEATURES = 57


def validate_numpy_dataset(data: np.ndarray, name: str) -> bool:
    """Validate the shape and content of a loaded numpy dataset.

    Args:
        data: The loaded numpy array.
        name: Human-readable dataset name for logging.

    Returns:
        True if the dataset passes validation checks.
    """
    logger.info(f"Validating {name}:")
    logger.info(f"  Shape: {data.shape}")
    logger.info(f"  Dtype: {data.dtype}")
    logger.info(f"  Min/Max: {data.min():.4f} / {data.max():.4f}")

    if data.ndim == 2:
        expected_cols = CULLPDB_SEQ_LEN * CULLPDB_NUM_FEATURES
        if data.shape[1] != expected_cols:
            logger.warning(
                f"  Unexpected column count: {data.shape[1]} "
                f"(expected {expected_cols})"
            )
            return False
        num_proteins = data.shape[0]
    elif data.ndim == 3:
        num_proteins = data.shape[0]
        if data.shape[1] != CULLPDB_SEQ_LEN or data.shape[2] != CULLPDB_NUM_FEATURES:
            logger.warning(
                f"  Unexpected shape: {data.shape} "
                f"(expected (N, {CULLPDB_SEQ_LEN}, {CULLPDB_NUM_FEATURES}))"
            )
            return False
    else:
        logger.warning(f"  Unexpected number of dimensions: {data.ndim}")
        return False

    logger.info(f"  Number of proteins: {num_proteins}")
    logger.info(f"  Validation: PASSED ✓")
    return True


def split_train_val(
    data: np.ndarray,
    val_fraction: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a dataset into training and validation subsets.

    Args:
        data: Full training numpy array.
        val_fraction: Fraction of data to use for validation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_data, val_data) numpy arrays.
    """
    rng = np.random.RandomState(seed)
    num_samples = data.shape[0]
    num_val = max(1, int(num_samples * val_fraction))

    indices = rng.permutation(num_samples)
    val_indices = indices[:num_val]
    train_indices = indices[num_val:]

    logger.info(
        f"Split: {len(train_indices)} train + {len(val_indices)} val "
        f"(val_fraction={val_fraction})"
    )
    return data[train_indices], data[val_indices]


def compute_dataset_statistics(data: np.ndarray, name: str) -> dict[str, float]:
    """Compute and log statistics for a dataset.

    Args:
        data: Numpy array of shape (N, 700, 57).
        name: Dataset name for logging.

    Returns:
        Dictionary of computed statistics.
    """
    if data.ndim == 2:
        data = data.reshape(-1, CULLPDB_SEQ_LEN, CULLPDB_NUM_FEATURES)

    num_proteins = data.shape[0]

    # Compute sequence lengths
    aa_profiles = data[:, :, :22]  # (N, 700, 22)
    seq_masks = aa_profiles.sum(axis=2) > 0  # (N, 700)
    seq_lengths = seq_masks.sum(axis=1)  # (N,)

    # Compute SS class distribution
    ss_profiles = data[:, :, 44:52]  # (N, 700, 8)
    ss_classes = np.argmax(ss_profiles, axis=2)  # (N, 700)

    ss_labels = ["H", "E", "G", "I", "B", "T", "S", "C"]
    ss_counts = {}
    total_residues = 0
    for i, label in enumerate(ss_labels):
        count = int(((ss_classes == i) & seq_masks).sum())
        ss_counts[label] = count
        total_residues += count

    logger.info(f"\n{name} Statistics:")
    logger.info(f"  Proteins: {num_proteins}")
    logger.info(f"  Sequence lengths: min={seq_lengths.min()}, max={seq_lengths.max()}, "
                f"mean={seq_lengths.mean():.1f}, median={np.median(seq_lengths):.1f}")
    logger.info(f"  Total residues: {total_residues}")
    logger.info(f"  SS class distribution:")
    for label, count in ss_counts.items():
        pct = 100 * count / total_residues if total_residues > 0 else 0
        logger.info(f"    {label}: {count:>8d} ({pct:>5.1f}%)")

    return {
        "num_proteins": num_proteins,
        "mean_length": float(seq_lengths.mean()),
        "min_length": int(seq_lengths.min()),
        "max_length": int(seq_lengths.max()),
        "total_residues": total_residues,
    }


def main() -> None:
    """Main entry point for the preprocessing script."""
    parser = argparse.ArgumentParser(
        description="Preprocess downloaded datasets for ProtIntel."
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(PROJECT_ROOT / "datasets" / "raw"),
        help="Directory containing raw downloaded files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "datasets" / "processed"),
        help="Directory to save processed files.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.05,
        help="Fraction of CullPDB to use as validation if RS126 is unavailable.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val splitting.",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    logger.info("=" * 60)
    logger.info("ProtIntel Data Preprocessor")
    logger.info("=" * 60)

    # Process CullPDB
    cullpdb_path = raw_dir / "cullpdb+profile_6133_filtered.npy"
    if cullpdb_path.exists():
        logger.info(f"\nLoading CullPDB from {cullpdb_path}")
        cullpdb_data = np.load(str(cullpdb_path))

        if cullpdb_data.ndim == 1:
            num_samples = cullpdb_data.shape[0] // (CULLPDB_SEQ_LEN * CULLPDB_NUM_FEATURES)
            cullpdb_data = cullpdb_data.reshape(
                num_samples, CULLPDB_SEQ_LEN, CULLPDB_NUM_FEATURES
            )

        if validate_numpy_dataset(cullpdb_data, "CullPDB"):
            compute_dataset_statistics(cullpdb_data, "CullPDB")

            # Split into train/val
            train_data, val_data = split_train_val(
                cullpdb_data, val_fraction=args.val_fraction, seed=args.seed
            )

            train_path = output_dir / "cullpdb_train.npy"
            val_path = output_dir / "rs126_val.npy"
            np.save(str(train_path), train_data)
            np.save(str(val_path), val_data)
            logger.info(f"Saved training data: {train_path}")
            logger.info(f"Saved validation data: {val_path}")
    else:
        logger.warning(f"CullPDB not found at {cullpdb_path}")
        logger.info("Run: python scripts/download_data.py")

    # Process CB513
    cb513_path = raw_dir / "cb513+profile_split1.npy"
    if cb513_path.exists():
        logger.info(f"\nLoading CB513 from {cb513_path}")
        cb513_data = np.load(str(cb513_path))

        if cb513_data.ndim == 1:
            num_samples = cb513_data.shape[0] // (CULLPDB_SEQ_LEN * CULLPDB_NUM_FEATURES)
            cb513_data = cb513_data.reshape(
                num_samples, CULLPDB_SEQ_LEN, CULLPDB_NUM_FEATURES
            )

        if validate_numpy_dataset(cb513_data, "CB513"):
            compute_dataset_statistics(cb513_data, "CB513")

            test_path = output_dir / "cb513_test.npy"
            np.save(str(test_path), cb513_data)
            logger.info(f"Saved test data: {test_path}")
    else:
        logger.warning(f"CB513 not found at {cb513_path}")

    logger.info("\n" + "=" * 60)
    logger.info("Preprocessing complete!")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("  1. Run: python scripts/generate_embeddings.py")
    logger.info("  2. Run: python train.py")


if __name__ == "__main__":
    main()
