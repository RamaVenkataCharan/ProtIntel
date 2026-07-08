"""Pre-compute and cache ESM-2 embeddings for all dataset sequences.

Iterates through the training, validation, and test datasets, generates
per-residue ESM-2 embeddings, and saves them as .pt files keyed by
sequence hash. This eliminates redundant computation during training.

Usage:
    python scripts/generate_embeddings.py
    python scripts/generate_embeddings.py --device cuda --batch-size 8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from tqdm import tqdm

from src.data.protein_dataset import ProteinDataset
from src.models.embedding_generator import EmbeddingGenerator
from src.utils.config_loader import load_config
from src.utils.io_utils import compute_sequence_hash, ensure_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_sequences_from_dataset(dataset: ProteinDataset) -> list[str]:
    """Extract unique amino acid sequences from a dataset.

    Args:
        dataset: A ProteinDataset instance.

    Returns:
        List of unique sequence strings.
    """
    sequences: list[str] = []
    seen_hashes: set[str] = set()

    for i in range(len(dataset)):
        sample = dataset[i]
        seq = sample["sequence"]
        seq_hash = compute_sequence_hash(seq)

        if seq_hash not in seen_hashes:
            seen_hashes.add(seq_hash)
            sequences.append(seq)

    logger.info(f"Extracted {len(sequences)} unique sequences from {len(dataset)} samples")
    return sequences


def generate_embeddings_for_sequences(
    sequences: list[str],
    generator: EmbeddingGenerator,
    cache_dir: Path,
    batch_size: int = 4,
) -> int:
    """Generate and cache ESM-2 embeddings for a list of sequences.

    Skips sequences that already have cached embeddings.

    Args:
        sequences: List of amino acid sequence strings.
        generator: EmbeddingGenerator instance.
        cache_dir: Directory for cached embedding files.
        batch_size: Number of sequences per batch.

    Returns:
        Number of newly generated embeddings.
    """
    ensure_dir(cache_dir)
    generated_count = 0
    skipped_count = 0

    # Filter out already-cached sequences
    uncached: list[str] = []
    for seq in sequences:
        cache_key = compute_sequence_hash(seq)
        cache_path = cache_dir / f"{cache_key}.pt"
        if cache_path.exists():
            skipped_count += 1
        else:
            uncached.append(seq)

    logger.info(
        f"Sequences: {len(sequences)} total, "
        f"{skipped_count} cached, {len(uncached)} to generate"
    )

    if not uncached:
        logger.info("All embeddings already cached. Nothing to do.")
        return 0

    # Process in batches
    progress = tqdm(range(0, len(uncached), batch_size), desc="Generating embeddings")
    for batch_start in progress:
        batch_end = min(batch_start + batch_size, len(uncached))
        batch_seqs = uncached[batch_start:batch_end]

        try:
            embeddings = generator.generate_batch(
                batch_seqs, use_cache=True, max_batch_size=batch_size
            )
            generated_count += len(embeddings)
            progress.set_postfix(generated=generated_count)
        except Exception as e:
            logger.error(f"Error generating embeddings for batch {batch_start}: {e}")
            # Try one-by-one for failed batch
            for seq in batch_seqs:
                try:
                    generator.generate_single(seq, use_cache=True)
                    generated_count += 1
                except Exception as inner_e:
                    logger.error(
                        f"Failed to generate embedding for sequence "
                        f"(len={len(seq)}): {inner_e}"
                    )

    return generated_count


def main() -> None:
    """Main entry point for the embedding generation script."""
    parser = argparse.ArgumentParser(
        description="Pre-compute ESM-2 embeddings for ProtIntel datasets."
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device to use (cpu, cuda, auto).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Number of sequences per batch.",
    )
    parser.add_argument(
        "--model-name", type=str, default=None,
        help="ESM-2 model name (default: from config).",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("ProtIntel ESM-2 Embedding Generator")
    logger.info("=" * 60)

    # Load config
    config = load_config()

    # Determine device
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Initialize embedding generator
    model_name = args.model_name or config.model.esm2.model_name
    cache_dir = Path(config.data.embeddings_cache_dir)

    generator = EmbeddingGenerator(
        model_name=model_name,
        embedding_dim=config.model.esm2.embedding_dim,
        freeze=True,
        cache_dir=cache_dir,
        device=device,
    )

    start_time = time.time()
    total_generated = 0

    # Process each dataset split
    processed_dir = Path(config.data.processed_dir)
    raw_dir = Path(config.data.raw_dir)

    dataset_files = {
        "CullPDB (train)": [
            processed_dir / "cullpdb_train.npy",
            raw_dir / "cullpdb+profile_6133_filtered.npy",
            raw_dir / "cullpdb+profile_6133_filtered.npy.gz",
        ],
        "RS126 (val)": [
            processed_dir / "rs126_val.npy",
            raw_dir / "rs126+profile_split1.npy",
            raw_dir / "rs126+profile_split1.npy.gz",
        ],
        "CB513 (test)": [
            processed_dir / "cb513_test.npy",
            raw_dir / "cb513+profile_split1.npy",
            raw_dir / "cb513+profile_split1.npy.gz",
        ],
    }

    for name, paths in dataset_files.items():
        # Find first existing path
        data_path = None
        for p in paths:
            if p.exists():
                data_path = p
                break

        if data_path is None:
            logger.warning(f"Skipping {name}: no data file found")
            continue

        logger.info(f"\nProcessing {name} from {data_path}")
        config = {
            "max_seq_length": 512,
            "min_seq_length": 10,
            "nonstandard_policy": "mask",
        }
        dataset = ProteinDataset(data_path=data_path, split=name, config=config, use_cache=False)
        sequences = extract_sequences_from_dataset(dataset)

        count = generate_embeddings_for_sequences(
            sequences=sequences,
            generator=generator,
            cache_dir=cache_dir,
            batch_size=args.batch_size,
        )
        total_generated += count
        logger.info(f"  Generated {count} new embeddings for {name}")

    elapsed = time.time() - start_time
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Embedding generation complete!")
    logger.info(f"  Total new embeddings: {total_generated}")
    logger.info(f"  Cache directory: {cache_dir}")
    logger.info(f"  Time elapsed: {elapsed:.1f}s")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
