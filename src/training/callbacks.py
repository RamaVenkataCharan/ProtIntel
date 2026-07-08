"""Training callbacks for early stopping and model checkpointing."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from src.utils.io_utils import ensure_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EarlyStopping:
    """Early stopping to halt training when validation performance plateaus.

    Monitors a specified metric and stops training if no improvement
    is observed for a given number of epochs (patience).

    Args:
        patience: Number of epochs with no improvement before stopping.
        monitor: Name of the metric to monitor.
        mode: 'max' if higher is better, 'min' if lower is better.
        min_delta: Minimum change to qualify as an improvement.
    """

    def __init__(
        self,
        patience: int = 15,
        monitor: str = "val_q3_accuracy",
        mode: str = "max",
        min_delta: float = 0.001,
    ) -> None:
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta

        self.best_value: Optional[float] = None
        self.counter = 0
        self.should_stop = False
        self.best_epoch = 0

        if mode == "max":
            self.compare = lambda current, best: current > best + min_delta
        else:
            self.compare = lambda current, best: current < best - min_delta

        logger.info(
            f"EarlyStopping: monitor={monitor}, patience={patience}, "
            f"mode={mode}, min_delta={min_delta}"
        )

    def step(self, metrics: dict[str, float], epoch: int) -> bool:
        """Check if training should be stopped.

        Args:
            metrics: Dictionary of metric name → value.
            epoch: Current epoch number.

        Returns:
            True if training should stop, False otherwise.
        """
        if self.monitor not in metrics:
            logger.warning(
                f"EarlyStopping: metric '{self.monitor}' not found. "
                f"Available: {list(metrics.keys())}"
            )
            return False

        current = metrics[self.monitor]

        if self.best_value is None:
            self.best_value = current
            self.best_epoch = epoch
            return False

        if self.compare(current, self.best_value):
            self.best_value = current
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            logger.info(
                f"EarlyStopping: no improvement for {self.counter}/{self.patience} "
                f"epochs (best {self.monitor}={self.best_value:.4f} at epoch {self.best_epoch})"
            )

        if self.counter >= self.patience:
            self.should_stop = True
            logger.info(
                f"EarlyStopping: stopping training. "
                f"Best {self.monitor}={self.best_value:.4f} at epoch {self.best_epoch}"
            )
            return True

        return False


class ModelCheckpoint:
    """Save top-K model checkpoints based on a monitored metric.

    Args:
        save_dir: Directory to save checkpoint files.
        monitor: Metric name to monitor.
        mode: 'max' or 'min'.
        save_top_k: Number of best checkpoints to keep.
    """

    def __init__(
        self,
        save_dir: str = "models/",
        monitor: str = "val_q3_accuracy",
        mode: str = "max",
        save_top_k: int = 3,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k

        self.best_checkpoints: list[tuple[float, Path]] = []

        if mode == "max":
            self.is_better = lambda a, b: a > b
        else:
            self.is_better = lambda a, b: a < b

        ensure_dir(self.save_dir)
        logger.info(
            f"ModelCheckpoint: dir={save_dir}, monitor={monitor}, "
            f"save_top_k={save_top_k}"
        )

    def step(
        self,
        model: nn.Module,
        metrics: dict[str, float],
        epoch: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Optional[Path]:
        """Potentially save a checkpoint if the metric improves.

        Args:
            model: The model to save.
            metrics: Dictionary of metric name → value.
            epoch: Current epoch number.
            optimizer: Optional optimizer state to include.

        Returns:
            Path to the saved checkpoint, or None if not saved.
        """
        if self.monitor not in metrics:
            return None

        current = metrics[self.monitor]
        metric_str = f"{current:.4f}".replace(".", "_")
        filename = f"checkpoint_epoch{epoch:03d}_{self.monitor}_{metric_str}.pt"
        filepath = self.save_dir / filename

        # Check if this is better than the worst saved checkpoint
        should_save = len(self.best_checkpoints) < self.save_top_k
        if not should_save and self.best_checkpoints:
            worst_value = self.best_checkpoints[-1][0]
            if self.is_better(current, worst_value):
                should_save = True

        if should_save:
            # Save checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metrics": metrics,
                self.monitor: current,
            }
            if optimizer is not None:
                checkpoint["optimizer_state_dict"] = optimizer.state_dict()

            torch.save(checkpoint, str(filepath))
            logger.info(
                f"Saved checkpoint: {filepath.name} "
                f"({self.monitor}={current:.4f})"
            )

            # Update best checkpoints list
            self.best_checkpoints.append((current, filepath))
            # Sort: best first
            self.best_checkpoints.sort(
                key=lambda x: x[0], reverse=(self.mode == "max")
            )

            # Remove worst if exceeding save_top_k
            while len(self.best_checkpoints) > self.save_top_k:
                _, removed_path = self.best_checkpoints.pop()
                if removed_path.exists():
                    removed_path.unlink()
                    logger.info(f"Removed old checkpoint: {removed_path.name}")

            # Always maintain a 'best' symlink/copy
            best_path = self.save_dir / "best_checkpoint.pt"
            if self.best_checkpoints:
                best_src = self.best_checkpoints[0][1]
                shutil.copy2(str(best_src), str(best_path))

            return filepath

        return None

    def get_best_checkpoint_path(self) -> Optional[Path]:
        """Return the path to the best checkpoint.

        Returns:
            Path to the best checkpoint, or None if no checkpoints exist.
        """
        best_path = self.save_dir / "best_checkpoint.pt"
        if best_path.exists():
            return best_path
        if self.best_checkpoints:
            return self.best_checkpoints[0][1]
        return None
