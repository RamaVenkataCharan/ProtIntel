"""Top-level evaluation entry point for ProtIntel.

Usage:
    python evaluate.py
    python evaluate.py --checkpoint models/best_checkpoint.pt --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_module import DataModule
from src.evaluation.evaluator import Evaluator
from src.evaluation.visualizer import Visualizer
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.reproducibility import get_device, set_seed

logger = get_logger(__name__)


def main() -> None:
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate ProtIntel on benchmark datasets.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--output-dir", type=str, default="logs/evaluation", help="Output directory")
    args = parser.parse_args()

    config = load_config()
    set_seed(config.training.seed)
    device = get_device(args.device)

    checkpoint_path = args.checkpoint or config.inference.checkpoint_path
    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.info("Run: python train.py first")
        sys.exit(1)

    # Load model
    evaluator = Evaluator.from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_config=config.model,
        device=str(device),
    )

    # Load test data
    data_module = DataModule.from_config(config=config.data, batch_size=args.batch_size)
    data_module.setup(stage="test")

    output_dir = Path(args.output_dir)
    visualizer = Visualizer(output_dir=output_dir)

    # Evaluate on CB513
    if data_module.test_dataset is not None:
        test_loader = data_module.test_dataloader()
        results = evaluator.evaluate_and_save(
            dataloader=test_loader,
            output_path=output_dir / "cb513_results.json",
            dataset_name="CB513",
        )

        # Generate visualizations
        if "q3_confusion_matrix" in results:
            visualizer.plot_confusion_matrix(
                results["q3_confusion_matrix"],
                ["H", "E", "C"],
                title="Q3 Confusion Matrix (CB513)",
                filename="q3_confusion_cb513",
            )
        if "q8_confusion_matrix" in results:
            visualizer.plot_confusion_matrix(
                results["q8_confusion_matrix"],
                ["H", "E", "G", "I", "B", "T", "S", "C"],
                title="Q8 Confusion Matrix (CB513)",
                filename="q8_confusion_cb513",
            )

        visualizer.plot_per_class_accuracy(results, task="q3", filename="cb513_per_class")
        visualizer.plot_per_class_accuracy(results, task="q8", filename="cb513_per_class")

    logger.info("\nEvaluation complete! Results saved to: " + str(output_dir))


if __name__ == "__main__":
    main()
