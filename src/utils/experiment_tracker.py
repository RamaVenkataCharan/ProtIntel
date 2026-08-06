"""Experiment tracking and telemetry module for ProtIntel.

Logs hyperparameters, training loss, validation loss, Q3/Q8 accuracies,
Matthews Correlation Coefficient (MCC), confusion matrices, training duration,
and GPU memory utilization to structured JSON experiment logs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExperimentTracker:
    """Tracks and persists experiment runs and benchmark metrics.

    Args:
        experiment_name: Unique identifier for the experiment run.
        output_dir: Directory where experiment logs will be saved.
        config: Dictionary of model and training hyperparameters.
    """

    def __init__(
        self,
        experiment_name: str,
        output_dir: str | Path = "logs/experiments",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()

        self.telemetry: Dict[str, Any] = {
            "experiment_name": experiment_name,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hyperparameters": config or {},
            "epochs": [],
            "best_metrics": {},
            "system_info": self._get_system_info(),
        }

    def _get_system_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            info["device_name"] = torch.cuda.get_device_name(0)
            info["device_count"] = torch.cuda.device_count()
        return info

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_q3_acc: float,
        val_q8_acc: float,
        val_q3_mcc: float,
        val_q8_mcc: float,
        val_q8_macro_f1: Optional[float] = None,
        lr: Optional[float] = None,
    ) -> None:
        """Log metrics for a completed training epoch."""
        vram_mb = 0.0
        if torch.cuda.is_available():
            vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        epoch_record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_q3_acc": float(val_q3_acc),
            "val_q8_acc": float(val_q8_acc),
            "val_q3_mcc": float(val_q3_mcc),
            "val_q8_mcc": float(val_q8_mcc),
            "val_q8_macro_f1": float(val_q8_macro_f1) if val_q8_macro_f1 is not None else None,
            "learning_rate": float(lr) if lr is not None else None,
            "vram_allocated_mb": round(vram_mb, 2),
            "elapsed_seconds": round(time.time() - self.start_time, 2),
        }

        self.telemetry["epochs"].append(epoch_record)
        logger.info(
            f"[ExpTracker] Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Q3: {val_q3_acc*100:.2f}% | "
            f"Val Q8: {val_q8_acc*100:.2f}% | Val MCC: {val_q3_mcc:.4f}"
        )

    def log_final_evaluation(
        self,
        dataset_name: str,
        results: Dict[str, Any],
        q3_confusion_matrix: Optional[List[List[int]]] = None,
        q8_confusion_matrix: Optional[List[List[int]]] = None,
    ) -> None:
        """Log final benchmark evaluation on held-out test sets (e.g. CB513)."""
        self.telemetry["best_metrics"][dataset_name] = {
            "q3_accuracy": float(results.get("q3_accuracy", 0.0)),
            "q8_accuracy": float(results.get("q8_accuracy", 0.0)),
            "q3_mcc": float(results.get("q3_mcc", 0.0)),
            "q8_mcc": float(results.get("q8_mcc", 0.0)),
            "q8_macro_f1": float(results.get("q8_macro_f1", 0.0)),
            "q3_confusion_matrix": q3_confusion_matrix,
            "q8_confusion_matrix": q8_confusion_matrix,
        }
        self.telemetry["total_duration_seconds"] = round(time.time() - self.start_time, 2)
        self.save()

    def save(self) -> Path:
        """Save telemetry log to JSON disk artifact."""
        save_path = self.output_dir / f"{self.experiment_name}_telemetry.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.telemetry, f, indent=2)
        logger.info(f"Experiment log saved to: {save_path}")
        return save_path
