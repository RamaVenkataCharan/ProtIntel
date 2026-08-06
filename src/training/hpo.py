"""Hyperparameter Optimization (HPO) pipeline using Optuna for ProtIntel.

Automates search over learning rate, weight decay, dropout, kernel sizes,
hidden dimensions, loss types, and focal loss parameters targeting maximum validation Q3/Q8 MCC.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import optuna
from src.data.data_module import DataModule
from src.models.protintel_model import ProtIntelModel
from src.training.trainer import ModelTrainer
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.reproducibility import get_device, set_seed

logger = get_logger(__name__)


def objective(trial: optuna.Trial, base_config_path: str = "configs/training.yaml") -> float:
    config = load_config()

    # Sample hyperparameters
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    loss_type = trial.suggest_categorical("loss_type", ["cross_entropy", "focal"])
    focal_gamma = trial.suggest_float("focal_gamma", 1.5, 4.0)

    config.training.epochs = 5  # Fast HPO trials
    config.training.learning_rate = lr
    config.training.weight_decay = weight_decay
    config.training.loss.type = loss_type
    config.training.loss.focal_gamma = focal_gamma
    config.model.cnn.dropout = dropout
    config.model.bilstm.dropout = dropout

    set_seed(42)
    device = get_device("auto")

    data_module = DataModule.from_config(config=config.data, batch_size=16)
    data_module.setup(stage="fit")

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    model = ProtIntelModel(config=config.model, device=str(device))
    trainer = ModelTrainer(model=model, config=config.training, device=str(device))

    results = trainer.fit(train_loader, val_loader)
    best_val_q3 = max(results["history"]["val_q3_accuracy"]) if results["history"]["val_q3_accuracy"] else 0.0
    return float(best_val_q3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna HPO for ProtIntel.")
    parser.add_argument("--trials", type=int, default=10, help="Number of Optuna trials")
    args = parser.parse_args()

    logger.info(f"Starting Optuna hyperparameter search with {args.trials} trials...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials)

    logger.info("=" * 60)
    logger.info("Optuna Study Complete!")
    logger.info(f"Best Trial Value (Val Q3 Acc): {study.best_value:.4f}")
    logger.info("Best Parameters:")
    for param_name, param_val in study.best_params.items():
        logger.info(f"  {param_name}: {param_val}")


if __name__ == "__main__":
    main()
