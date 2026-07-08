"""Evaluation metrics for protein secondary structure prediction.

Implements Q3/Q8 accuracy, per-class metrics, MCC, confusion matrix,
and other evaluation metrics using torchmetrics.
"""

from __future__ import annotations

from typing import Optional

import torch
import torchmetrics

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ProteinMetrics:
    """Comprehensive metric tracker for PSSP evaluation.

    Tracks Q3 and Q8 accuracy, per-class precision/recall/F1,
    Matthews Correlation Coefficient, and confusion matrices.
    All metrics ignore padding positions (label == -100).

    Args:
        num_q3_classes: Number of Q3 classes (default 3).
        num_q8_classes: Number of Q8 classes (default 8).
        device: Device for metric tensors.
    """

    def __init__(
        self,
        num_q3_classes: int = 3,
        num_q8_classes: int = 8,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.num_q3_classes = num_q3_classes
        self.num_q8_classes = num_q8_classes

        # Q3 metrics
        self.q3_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_q3_classes, average="micro"
        ).to(device)
        self.q3_per_class_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_q3_classes, average="none"
        ).to(device)
        self.q3_f1_macro = torchmetrics.F1Score(
            task="multiclass", num_classes=num_q3_classes, average="macro"
        ).to(device)
        self.q3_f1_weighted = torchmetrics.F1Score(
            task="multiclass", num_classes=num_q3_classes, average="weighted"
        ).to(device)
        self.q3_precision = torchmetrics.Precision(
            task="multiclass", num_classes=num_q3_classes, average="macro"
        ).to(device)
        self.q3_recall = torchmetrics.Recall(
            task="multiclass", num_classes=num_q3_classes, average="macro"
        ).to(device)
        self.q3_mcc = torchmetrics.MatthewsCorrCoef(
            task="multiclass", num_classes=num_q3_classes
        ).to(device)
        self.q3_confusion = torchmetrics.ConfusionMatrix(
            task="multiclass", num_classes=num_q3_classes, normalize="true"
        ).to(device)

        # Q8 metrics
        self.q8_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_q8_classes, average="micro"
        ).to(device)
        self.q8_per_class_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_q8_classes, average="none"
        ).to(device)
        self.q8_f1_macro = torchmetrics.F1Score(
            task="multiclass", num_classes=num_q8_classes, average="macro"
        ).to(device)
        self.q8_f1_weighted = torchmetrics.F1Score(
            task="multiclass", num_classes=num_q8_classes, average="weighted"
        ).to(device)
        self.q8_mcc = torchmetrics.MatthewsCorrCoef(
            task="multiclass", num_classes=num_q8_classes
        ).to(device)
        self.q8_confusion = torchmetrics.ConfusionMatrix(
            task="multiclass", num_classes=num_q8_classes, normalize="true"
        ).to(device)

    def update(
        self,
        q3_preds: torch.Tensor,
        q3_targets: torch.Tensor,
        q8_preds: torch.Tensor,
        q8_targets: torch.Tensor,
        ignore_index: int = -100,
    ) -> None:
        """Update all metrics with a batch of predictions.

        Args:
            q3_preds: Q3 predictions of shape (B, L).
            q3_targets: Q3 ground-truth labels of shape (B, L).
            q8_preds: Q8 predictions of shape (B, L).
            q8_targets: Q8 ground-truth labels of shape (B, L).
            ignore_index: Index value to ignore (padding).
        """
        # Flatten and mask padding
        q3_preds_flat = q3_preds.reshape(-1)
        q3_targets_flat = q3_targets.reshape(-1)
        q8_preds_flat = q8_preds.reshape(-1)
        q8_targets_flat = q8_targets.reshape(-1)

        q3_mask = q3_targets_flat != ignore_index
        q8_mask = q8_targets_flat != ignore_index

        if q3_mask.any():
            q3_p = q3_preds_flat[q3_mask].to(self.device)
            q3_t = q3_targets_flat[q3_mask].to(self.device)
            self.q3_accuracy.update(q3_p, q3_t)
            self.q3_per_class_accuracy.update(q3_p, q3_t)
            self.q3_f1_macro.update(q3_p, q3_t)
            self.q3_f1_weighted.update(q3_p, q3_t)
            self.q3_precision.update(q3_p, q3_t)
            self.q3_recall.update(q3_p, q3_t)
            self.q3_mcc.update(q3_p, q3_t)
            self.q3_confusion.update(q3_p, q3_t)

        if q8_mask.any():
            q8_p = q8_preds_flat[q8_mask].to(self.device)
            q8_t = q8_targets_flat[q8_mask].to(self.device)
            self.q8_accuracy.update(q8_p, q8_t)
            self.q8_per_class_accuracy.update(q8_p, q8_t)
            self.q8_f1_macro.update(q8_p, q8_t)
            self.q8_f1_weighted.update(q8_p, q8_t)
            self.q8_mcc.update(q8_p, q8_t)
            self.q8_confusion.update(q8_p, q8_t)

    def compute(self) -> dict[str, float | list[float]]:
        """Compute all accumulated metrics.

        Returns:
            Dictionary of metric names to values. Per-class metrics
            are returned as lists.
        """
        q3_class_names = ["H", "E", "C"]
        q8_class_names = ["H", "E", "G", "I", "B", "T", "S", "C"]

        results: dict[str, float | list[float]] = {}

        # Q3 metrics
        results["q3_accuracy"] = self.q3_accuracy.compute().item()
        q3_per_class = self.q3_per_class_accuracy.compute()
        for i, name in enumerate(q3_class_names):
            results[f"q3_accuracy_{name}"] = q3_per_class[i].item()
        results["q3_f1_macro"] = self.q3_f1_macro.compute().item()
        results["q3_f1_weighted"] = self.q3_f1_weighted.compute().item()
        results["q3_precision"] = self.q3_precision.compute().item()
        results["q3_recall"] = self.q3_recall.compute().item()
        results["q3_mcc"] = self.q3_mcc.compute().item()
        results["q3_confusion_matrix"] = self.q3_confusion.compute().cpu().tolist()

        # Q8 metrics
        results["q8_accuracy"] = self.q8_accuracy.compute().item()
        q8_per_class = self.q8_per_class_accuracy.compute()
        for i, name in enumerate(q8_class_names):
            results[f"q8_accuracy_{name}"] = q8_per_class[i].item()
        results["q8_f1_macro"] = self.q8_f1_macro.compute().item()
        results["q8_f1_weighted"] = self.q8_f1_weighted.compute().item()
        results["q8_mcc"] = self.q8_mcc.compute().item()
        results["q8_confusion_matrix"] = self.q8_confusion.compute().cpu().tolist()

        return results

    def reset(self) -> None:
        """Reset all metrics to their initial state."""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, torchmetrics.Metric):
                attr.reset()

    def log_summary(self, prefix: str = "") -> dict[str, float]:
        """Compute and log a human-readable summary of key metrics.

        Args:
            prefix: Optional prefix for metric names (e.g., 'val_').

        Returns:
            Dictionary of key metric values.
        """
        results = self.compute()

        summary = {
            f"{prefix}q3_accuracy": results["q3_accuracy"],
            f"{prefix}q8_accuracy": results["q8_accuracy"],
            f"{prefix}q3_mcc": results["q3_mcc"],
            f"{prefix}q3_f1_macro": results["q3_f1_macro"],
            f"{prefix}q8_f1_macro": results["q8_f1_macro"],
        }

        logger.info(f"{'='*50}")
        logger.info(f"Metrics Summary ({prefix.rstrip('_') or 'eval'}):")
        logger.info(f"  Q3 Accuracy: {results['q3_accuracy']:.4f}")
        logger.info(f"  Q8 Accuracy: {results['q8_accuracy']:.4f}")
        logger.info(f"  Q3 MCC:      {results['q3_mcc']:.4f}")
        logger.info(f"  Q3 F1 macro: {results['q3_f1_macro']:.4f}")
        logger.info(f"  Q8 F1 macro: {results['q8_f1_macro']:.4f}")

        # Per-class Q3
        for cls in ["H", "E", "C"]:
            key = f"q3_accuracy_{cls}"
            if key in results:
                logger.info(f"  Q3 {cls}: {results[key]:.4f}")

        logger.info(f"{'='*50}")
        return summary
