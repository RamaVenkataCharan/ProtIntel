"""Full benchmark evaluation for ProtIntel.

Loads a trained checkpoint and evaluates on CB513 (and optionally CASP12),
reporting all metrics with per-class breakdowns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

from src.data.data_module import DataModule
from src.models.protintel_model import ProtIntelModel
from src.training.metrics import ProteinMetrics
from src.utils.config_loader import ModelConfig, DataConfig
from src.utils.io_utils import save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Evaluator:
    """Benchmark evaluator for protein secondary structure prediction.

    Loads a trained model checkpoint and runs comprehensive evaluation
    on test datasets, producing full metric reports.

    Args:
        model: ProtIntelModel instance.
        device: Evaluation device.
    """

    def __init__(
        self,
        model: ProtIntelModel,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.device = torch.device(device)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model_config: Optional[ModelConfig] = None,
        device: str = "cpu",
    ) -> "Evaluator":
        """Create an Evaluator from a saved checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint .pt file.
            model_config: Model configuration (uses defaults if None).
            device: Evaluation device.

        Returns:
            Configured Evaluator instance.
        """
        config = model_config or ModelConfig()
        model = ProtIntelModel(config=config, device=device)

        checkpoint = torch.load(
            str(checkpoint_path), map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")

        if "metrics" in checkpoint:
            logger.info(f"Checkpoint metrics: {checkpoint['metrics']}")

        return cls(model=model, device=device)

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        dataset_name: str = "test",
    ) -> dict[str, Any]:
        """Run evaluation on a dataset.

        Args:
            dataloader: DataLoader for the evaluation dataset.
            dataset_name: Name of the dataset for logging.

        Returns:
            Dictionary of all computed metrics.
        """
        logger.info(f"Evaluating on {dataset_name}...")
        metrics = ProteinMetrics(device=str(self.device))

        for batch in dataloader:
            embeddings = batch.get("embeddings")
            if embeddings is not None:
                embeddings = embeddings.to(self.device)

            attention_mask = batch["attention_mask"].to(self.device)
            seq_lengths = batch["seq_length"].to(self.device)

            outputs = self.model(
                sequences=batch.get("sequence"),
                embeddings=embeddings,
                attention_mask=attention_mask,
                seq_lengths=seq_lengths,
            )

            metrics.update(
                q3_preds=outputs["q3_preds"].cpu(),
                q3_targets=batch["q3_labels"],
                q8_preds=outputs["q8_preds"].cpu(),
                q8_targets=batch["q8_labels"],
            )

        results = metrics.log_summary(prefix=f"{dataset_name}_")
        full_results = metrics.compute()
        full_results["dataset"] = dataset_name

        return full_results

    def evaluate_and_save(
        self,
        dataloader: DataLoader,
        output_path: str | Path,
        dataset_name: str = "cb513",
    ) -> dict[str, Any]:
        """Evaluate and save results to a JSON file.

        Args:
            dataloader: DataLoader for evaluation.
            output_path: Path to save the results JSON.
            dataset_name: Name of the dataset.

        Returns:
            Dictionary of all computed metrics.
        """
        results = self.evaluate(dataloader, dataset_name)

        # Filter out non-serializable items for JSON
        serializable_results = {}
        for k, v in results.items():
            if isinstance(v, (float, int, str, list)):
                serializable_results[k] = v

        save_json(serializable_results, output_path)
        logger.info(f"Results saved to {output_path}")

        return results
