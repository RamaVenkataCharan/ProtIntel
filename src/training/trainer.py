"""Model trainer for ProtIntel with AMP, gradient accumulation, and logging."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_  # type: ignore[attr-defined]
sys.modules["tensorflow"] = None

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.training.losses import create_loss_function
from src.training.metrics import ProteinMetrics
from src.utils.config_loader import TrainingConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """Full training pipeline for ProtIntelModel.

    Handles the training loop with mixed precision, gradient accumulation,
    learning rate scheduling, early stopping, checkpointing, and
    TensorBoard logging.

    Args:
        model: The ProtIntelModel to train.
        config: Training configuration.
        device: Device to train on.
        q3_class_weights: Optional per-class weights for Q3 loss.
        q8_class_weights: Optional per-class weights for Q8 loss.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainingConfig] = None,
        device: str = "cpu",
        q3_class_weights: Optional[torch.Tensor] = None,
        q8_class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        self.model = model.to(device)
        self.config = config or TrainingConfig()
        self.device = torch.device(device)

        # Loss functions
        self.q3_loss_fn = create_loss_function(
            loss_type=self.config.loss.type,
            label_smoothing=self.config.loss.label_smoothing,
            class_weights=q3_class_weights,
            focal_gamma=self.config.loss.focal_gamma,
        )
        self.q8_loss_fn = create_loss_function(
            loss_type=self.config.loss.type,
            label_smoothing=self.config.loss.label_smoothing,
            class_weights=q8_class_weights,
            focal_gamma=self.config.loss.focal_gamma,
        )

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision
        self.use_amp = self.config.mixed_precision and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        # Callbacks
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping.patience,
            monitor=self.config.early_stopping.monitor,
            mode=self.config.early_stopping.mode,
            min_delta=self.config.early_stopping.min_delta,
        ) if self.config.early_stopping.enabled else None

        self.checkpoint = ModelCheckpoint(
            save_dir=self.config.checkpoint.save_dir,
            monitor=self.config.checkpoint.monitor,
            mode=self.config.checkpoint.mode,
            save_top_k=self.config.checkpoint.save_top_k,
        )

        # TensorBoard
        self.writer: Optional[SummaryWriter] = None
        if self.config.logging.tensorboard:
            self.writer = SummaryWriter(self.config.logging.tensorboard_dir)

        # Task weights
        self.q3_weight = self.config.task_weights.q3
        self.q8_weight = self.config.task_weights.q8

        self.global_step = 0

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create the optimizer based on configuration.

        Returns:
            Configured optimizer instance.
        """
        # Get downstream parameters (exclude frozen ESM-2)
        if hasattr(self.model, "get_downstream_parameters"):
            params = self.model.get_downstream_parameters()
        else:
            params = [p for p in self.model.parameters() if p.requires_grad]

        opt_name = self.config.optimizer.lower()
        if opt_name == "adamw":
            return torch.optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=tuple(self.config.optimizer_params.betas),
                eps=self.config.optimizer_params.eps,
            )
        elif opt_name == "adam":
            return torch.optim.Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif opt_name == "sgd":
            return torch.optim.SGD(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        """Create the learning rate scheduler.

        Returns:
            Configured scheduler, or None if not specified.
        """
        sched_name = self.config.scheduler.lower()
        if sched_name == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=self.config.early_stopping.mode,
                factor=self.config.scheduler_params.factor,
                patience=self.config.scheduler_params.patience,
                min_lr=self.config.scheduler_params.min_lr,
            )
        elif sched_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.scheduler_params.T_0,
                T_mult=self.config.scheduler_params.T_mult,
            )
        elif sched_name == "onecycle":
            return None  # Created after knowing steps_per_epoch
        return None

    def _compute_loss(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined Q3 + Q8 loss.

        Args:
            outputs: Model output dictionary.
            batch: Batch dictionary with labels.

        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains
            individual loss values for logging.
        """
        loss_dict: dict[str, float] = {}

        q3_loss = self.q3_loss_fn(
            outputs["q3_logits"], batch["q3_labels"].to(self.device)
        )
        q8_loss = self.q8_loss_fn(
            outputs["q8_logits"], batch["q8_labels"].to(self.device)
        )

        total_loss = self.q3_weight * q3_loss + self.q8_weight * q8_loss

        loss_dict["q3_loss"] = q3_loss.item()
        loss_dict["q8_loss"] = q8_loss.item()
        loss_dict["total_loss"] = total_loss.item()

        return total_loss, loss_dict

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        """Run one training epoch.

        Args:
            train_loader: Training data loader.
            epoch: Current epoch number.

        Returns:
            Dictionary of average training metrics.
        """
        self.model.train()
        metrics = ProteinMetrics(device=str(self.device))
        epoch_losses: dict[str, list[float]] = {
            "total_loss": [], "q3_loss": [], "q8_loss": []
        }

        accum_steps = self.config.gradient_accumulation_steps
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            # Move tensors to device
            embeddings = batch.get("embeddings")
            if embeddings is not None:
                embeddings = embeddings.to(self.device)

            attention_mask = batch["attention_mask"].to(self.device)
            seq_lengths = batch["seq_length"].to(self.device)

            # Forward pass with AMP
            with autocast(enabled=self.use_amp):
                outputs = self.model(
                    sequences=batch.get("sequence"),
                    embeddings=embeddings,
                    attention_mask=attention_mask,
                    seq_lengths=seq_lengths,
                )
                loss, loss_dict = self._compute_loss(outputs, batch)
                loss = loss / accum_steps

            # Backward pass
            self.scaler.scale(loss).backward()

            # Gradient accumulation
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                if self.config.gradient_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            # Track metrics
            for key, val in loss_dict.items():
                epoch_losses[key].append(val)

            with torch.no_grad():
                metrics.update(
                    q3_preds=outputs["q3_preds"].cpu(),
                    q3_targets=batch["q3_labels"],
                    q8_preds=outputs["q8_preds"].cpu(),
                    q8_targets=batch["q8_labels"],
                )

            # Logging
            self.global_step += 1
            if self.writer and self.global_step % self.config.logging.log_every_n_steps == 0:
                self.writer.add_scalar("train/loss", loss_dict["total_loss"], self.global_step)
                self.writer.add_scalar("train/q3_loss", loss_dict["q3_loss"], self.global_step)
                self.writer.add_scalar("train/q8_loss", loss_dict["q8_loss"], self.global_step)
                self.writer.add_scalar(
                    "train/lr", self.optimizer.param_groups[0]["lr"], self.global_step
                )

        # Compute epoch summary
        metric_results = metrics.compute()
        avg_losses = {k: sum(v) / len(v) for k, v in epoch_losses.items()}

        results = {**avg_losses, **{f"train_{k}": v for k, v in metric_results.items()
                                     if isinstance(v, float)}}

        logger.info(
            f"Epoch {epoch} [train]: loss={avg_losses['total_loss']:.4f}, "
            f"Q3={metric_results['q3_accuracy']:.4f}, "
            f"Q8={metric_results['q8_accuracy']:.4f}"
        )

        return results

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        """Run validation.

        Args:
            val_loader: Validation data loader.
            epoch: Current epoch number.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        metrics = ProteinMetrics(device=str(self.device))
        val_losses: list[float] = []

        for batch in val_loader:
            embeddings = batch.get("embeddings")
            if embeddings is not None:
                embeddings = embeddings.to(self.device)

            attention_mask = batch["attention_mask"].to(self.device)
            seq_lengths = batch["seq_length"].to(self.device)

            with autocast(enabled=self.use_amp):
                outputs = self.model(
                    sequences=batch.get("sequence"),
                    embeddings=embeddings,
                    attention_mask=attention_mask,
                    seq_lengths=seq_lengths,
                )
                loss, _ = self._compute_loss(outputs, batch)
                val_losses.append(loss.item())

            metrics.update(
                q3_preds=outputs["q3_preds"].cpu(),
                q3_targets=batch["q3_labels"],
                q8_preds=outputs["q8_preds"].cpu(),
                q8_targets=batch["q8_labels"],
            )

        metric_results = metrics.compute()
        avg_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0

        results: dict[str, float] = {"val_loss": avg_loss}
        for k, v in metric_results.items():
            if isinstance(v, float):
                results[f"val_{k}"] = v

        # TensorBoard logging
        if self.writer:
            self.writer.add_scalar("val/loss", avg_loss, epoch)
            self.writer.add_scalar("val/q3_accuracy", metric_results["q3_accuracy"], epoch)
            self.writer.add_scalar("val/q8_accuracy", metric_results["q8_accuracy"], epoch)
            self.writer.add_scalar("val/q3_mcc", metric_results["q3_mcc"], epoch)

        logger.info(
            f"Epoch {epoch} [val]:   loss={avg_loss:.4f}, "
            f"Q3={metric_results['q3_accuracy']:.4f}, "
            f"Q8={metric_results['q8_accuracy']:.4f}, "
            f"MCC={metric_results['q3_mcc']:.4f}"
        )

        return results

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict[str, Any]:
        """Run the full training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.

        Returns:
            Dictionary with training history and best checkpoint path.
        """
        logger.info("=" * 60)
        logger.info("Starting ProtIntel Training")
        logger.info(f"  Epochs: {self.config.epochs}")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Learning rate: {self.config.learning_rate}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Mixed precision: {self.use_amp}")
        logger.info("=" * 60)

        history: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [],
            "train_q3_accuracy": [], "val_q3_accuracy": [],
            "train_q8_accuracy": [], "val_q8_accuracy": [],
        }

        start_time = time.time()

        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validate
            val_metrics = self.validate(val_loader, epoch)

            # Update history
            history["train_loss"].append(train_metrics.get("total_loss", 0.0))
            history["val_loss"].append(val_metrics.get("val_loss", 0.0))
            history["train_q3_accuracy"].append(train_metrics.get("train_q3_accuracy", 0.0))
            history["val_q3_accuracy"].append(val_metrics.get("val_q3_accuracy", 0.0))
            history["train_q8_accuracy"].append(train_metrics.get("train_q8_accuracy", 0.0))
            history["val_q8_accuracy"].append(val_metrics.get("val_q8_accuracy", 0.0))

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    monitor_val = val_metrics.get(self.config.early_stopping.monitor, 0.0)
                    self.scheduler.step(monitor_val)
                else:
                    self.scheduler.step()

            # Checkpointing
            self.checkpoint.step(
                model=self.model,
                metrics=val_metrics,
                epoch=epoch,
                optimizer=self.optimizer,
            )

            # Early stopping
            if self.early_stopping is not None:
                if self.early_stopping.step(val_metrics, epoch):
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

            epoch_time = time.time() - epoch_start
            logger.info(f"  Epoch {epoch} completed in {epoch_time:.1f}s")

        total_time = time.time() - start_time
        logger.info(f"\nTraining completed in {total_time / 60:.1f} minutes")

        if self.writer:
            self.writer.close()

        best_path = self.checkpoint.get_best_checkpoint_path()
        return {
            "history": history,
            "best_checkpoint": str(best_path) if best_path else None,
            "total_time_seconds": total_time,
        }
