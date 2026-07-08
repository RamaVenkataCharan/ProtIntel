"""Top-level training entry point for ProtIntel.

Loads configuration, initializes model, data, and trainer, then
runs the full training loop.

Usage:
    python train.py
    python train.py --device cuda --epochs 50 --batch-size 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data.data_module import DataModule
from src.models.protintel_model import ProtIntelModel
from src.training.trainer import ModelTrainer
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.reproducibility import get_device, set_seed

logger = get_logger(__name__)


def main() -> None:
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description="Train the ProtIntel protein secondary structure prediction model."
    )
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, auto)")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config()

    # Apply CLI overrides
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.seed is not None:
        config.training.seed = args.seed

    # Reproducibility
    set_seed(config.training.seed, config.training.deterministic)

    # Device
    device = get_device(args.device)

    logger.info("=" * 60)
    logger.info("ProtIntel Training Pipeline")
    logger.info("=" * 60)
    logger.info(f"  Device: {device}")
    logger.info(f"  Epochs: {config.training.epochs}")
    logger.info(f"  Batch size: {config.training.batch_size}")
    logger.info(f"  Learning rate: {config.training.learning_rate}")

    # Initialize data module
    logger.info("\nLoading datasets...")
    data_module = DataModule.from_config(
        config=config.data,
        batch_size=config.training.batch_size,
    )
    data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    logger.info(f"  Training batches: {len(train_loader)}")
    logger.info(f"  Validation batches: {len(val_loader)}")

    # Compute class weights if configured
    q3_weights = None
    q8_weights = None
    if config.training.loss.use_class_weights:
        logger.info("\nComputing class weights...")
        try:
            import numpy as np
            from src.data.preprocessor import compute_class_weights

            # Collect all labels from training set
            q3_all, q8_all = [], []
            for i in range(len(data_module.train_dataset)):
                sample = data_module.train_dataset[i]
                q3_mask = sample["q3_labels"] != -100
                q8_mask = sample["q8_labels"] != -100
                q3_all.append(sample["q3_labels"][q3_mask].tolist())
                q8_all.append(sample["q8_labels"][q8_mask].tolist())

            if q3_all:
                q3_weights = compute_class_weights(q3_all, 3)
                logger.info(f"  Q3 weights: {q3_weights.tolist()}")
            if q8_all:
                q8_weights = compute_class_weights(q8_all, 8)
                logger.info(f"  Q8 weights: {q8_weights.tolist()}")
        except Exception as e:
            logger.warning(f"Could not compute class weights: {e}")

    # Initialize model
    logger.info("\nInitializing model...")
    model = ProtIntelModel(config=config.model, device=str(device))

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    # Initialize trainer
    trainer = ModelTrainer(
        model=model,
        config=config.training,
        device=str(device),
        q3_class_weights=q3_weights,
        q8_class_weights=q8_weights,
    )

    # Train
    logger.info("\nStarting training...")
    results = trainer.fit(train_loader, val_loader)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"  Best checkpoint: {results['best_checkpoint']}")
    logger.info(f"  Total time: {results['total_time_seconds'] / 60:.1f} minutes")

    if results["history"]["val_q3_accuracy"]:
        best_q3 = max(results["history"]["val_q3_accuracy"])
        best_q8 = max(results["history"]["val_q8_accuracy"])
        logger.info(f"  Best val Q3 accuracy: {best_q3:.4f}")
        logger.info(f"  Best val Q8 accuracy: {best_q8:.4f}")

    logger.info("\nNext steps:")
    logger.info("  python evaluate.py  — Run full benchmark evaluation")
    logger.info("  python infer.py SEQUENCE  — Test single sequence prediction")


if __name__ == "__main__":
    main()
