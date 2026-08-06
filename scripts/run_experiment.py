"""Single-Variable Experiment Runner for ProtIntel.

Executes isolated single-variable experiments, logs telemetry to
experiments/experiment_XXX/, evaluates on RS126 (validation) and CB513 (test),
and strictly verifies improvement before accepting changes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.data.data_module import DataModule
from src.evaluation.evaluator import Evaluator
from src.evaluation.visualizer import Visualizer
from src.models.protintel_model import ProtIntelModel
from src.training.trainer import ModelTrainer
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.reproducibility import get_device, set_seed

logger = get_logger(__name__)


def run_experiment(
    exp_name: str,
    override_config: Optional[Dict[str, Any]] = None,
    baseline_metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    exp_dir = PROJECT_ROOT / "experiments" / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    config = load_config()

    # Apply single-variable overrides
    if override_config:
        for section, params in override_config.items():
            if hasattr(config, section):
                sec_obj = getattr(config, section)
                for k, v in params.items():
                    if hasattr(sec_obj, k):
                        setattr(sec_obj, k, v)

    # Save experiment config
    config_save_path = exp_dir / "config.json"
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(config.dict() if hasattr(config, "dict") else {}, f, indent=2)

    set_seed(config.training.seed, config.training.deterministic)
    device = get_device("auto")

    logger.info("=" * 60)
    logger.info(f"Running Experiment: {exp_name}")
    logger.info(f"  Directory: {exp_dir}")
    logger.info(f"  Device: {device}")
    logger.info("=" * 60)

    # Setup datasets
    data_module = DataModule.from_config(config=config.data, batch_size=config.training.batch_size)
    data_module.setup(stage="fit")

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Initialize model & trainer
    model = ProtIntelModel(config=config.model, device=str(device))
    trainer = ModelTrainer(model=model, config=config.training, device=str(device))

    # Train
    start_time = time.time()
    train_results = trainer.fit(train_loader, val_loader)
    train_duration = time.time() - start_time

    best_ckpt = train_results.get("best_checkpoint")
    if best_ckpt and Path(best_ckpt).exists():
        shutil.copy(best_ckpt, exp_dir / "best_checkpoint.pt")

    # Evaluate on RS126 (Validation)
    evaluator = Evaluator(model=model, device=str(device))
    val_metrics = evaluator.evaluate(val_loader, dataset_name="RS126")

    # Evaluate on CB513 (Test)
    data_module.setup(stage="test")
    test_loader = data_module.test_dataloader()
    cb513_metrics = evaluator.evaluate_and_save(
        dataloader=test_loader,
        output_path=exp_dir / "cb513_results.json",
        dataset_name="CB513",
    )

    # Generate Visualizations
    visualizer = Visualizer(output_dir=exp_dir)
    if "q3_confusion_matrix" in cb513_metrics:
        visualizer.plot_confusion_matrix(
            cb513_metrics["q3_confusion_matrix"],
            ["H", "E", "C"],
            title=f"Q3 Confusion Matrix ({exp_name})",
            filename="q3_confusion",
        )
    if "q8_confusion_matrix" in cb513_metrics:
        visualizer.plot_confusion_matrix(
            cb513_metrics["q8_confusion_matrix"],
            ["H", "E", "G", "I", "B", "T", "S", "C"],
            title=f"Q8 Confusion Matrix ({exp_name})",
            filename="q8_confusion",
        )

    # Compile Final Report
    exp_results = {
        "experiment_name": exp_name,
        "duration_seconds": round(train_duration, 2),
        "validation_rs126": {
            "q3_accuracy": val_metrics.get("q3_accuracy", 0.0),
            "q8_accuracy": val_metrics.get("q8_accuracy", 0.0),
            "q3_mcc": val_metrics.get("q3_mcc", 0.0),
        },
        "test_cb513": {
            "q3_accuracy": cb513_metrics.get("q3_accuracy", 0.0),
            "q8_accuracy": cb513_metrics.get("q8_accuracy", 0.0),
            "q3_mcc": cb513_metrics.get("q3_mcc", 0.0),
            "q8_macro_f1": cb513_metrics.get("q8_macro_f1", 0.0),
        },
    }

    # Compare against baseline if provided
    if baseline_metrics:
        val_diff = exp_results["validation_rs126"]["q3_accuracy"] - baseline_metrics.get("val_q3", 0.0)
        cb513_diff = exp_results["test_cb513"]["q3_accuracy"] - baseline_metrics.get("cb513_q3", 0.0)
        exp_results["diff_val_q3"] = round(val_diff, 4)
        exp_results["diff_cb513_q3"] = round(cb513_diff, 4)

        if val_diff > 0 and cb513_diff >= 0:
            exp_results["status"] = "ACCEPTED"
        else:
            exp_results["status"] = "REJECTED"

    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(exp_results, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"Experiment {exp_name} Complete!")
    logger.info(f"  RS126 Val Q3: {exp_results['validation_rs126']['q3_accuracy']*100:.2f}%")
    logger.info(f"  CB513 Test Q3: {exp_results['test_cb513']['q3_accuracy']*100:.2f}%")
    logger.info(f"  CB513 Test Q8: {exp_results['test_cb513']['q8_accuracy']*100:.2f}%")
    logger.info(f"  CB513 Test MCC: {exp_results['test_cb513']['q3_mcc']:.4f}")
    if "status" in exp_results:
        logger.info(f"  Status: {exp_results['status']}")
    logger.info("=" * 60)

    return exp_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run single-variable experiment.")
    parser.add_argument("--name", type=str, required=True, help="Experiment name")
    args = parser.parse_args()
    run_experiment(exp_name=args.name)
