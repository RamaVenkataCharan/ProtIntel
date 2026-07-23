"""Configuration loader for ProtIntel.

Loads and validates YAML configuration files using Pydantic dataclasses.
All configuration values are centralized here — no hardcoded constants
should exist elsewhere in the codebase.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


def get_project_root() -> Path:
    """Get the absolute path to the project root directory.

    Returns:
        Path to the project root (directory containing configs/).
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "configs").is_dir():
            return parent
    return Path.cwd()


PROJECT_ROOT = get_project_root()


def load_yaml(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML file. Can be absolute or relative
            to the project root.

    Returns:
        Dictionary containing the parsed YAML configuration.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config if config is not None else {}


class ESM2Config(BaseModel):
    """ESM-2 backbone configuration."""

    model_name: str = "facebook/esm2_t12_35M_UR50D"
    embedding_dim: int = 480
    freeze: bool = True
    finetune_last_n_layers: int = 0


class CNNConfig(BaseModel):
    """Multi-scale CNN encoder configuration."""

    input_dim: int = 480
    hidden_dim: int = 512
    kernel_sizes: list[int] = Field(default_factory=lambda: [3, 5, 7])
    num_residual_blocks: int = 3
    dropout: float = 0.1


class BiLSTMConfig(BaseModel):
    """Bidirectional LSTM encoder configuration."""

    input_dim: int = 512
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True


class AttentionConfig(BaseModel):
    """Multi-head self-attention configuration."""

    embed_dim: int = 512
    num_heads: int = 8
    dropout: float = 0.1


class FeedForwardConfig(BaseModel):
    """Feed-forward block configuration."""

    hidden_dim: int = 1024
    dropout: float = 0.1


class HeadConfig(BaseModel):
    """Prediction head configuration."""

    input_dim: int = 512
    hidden_dim: int = 256
    num_classes: int = 3
    dropout: float = 0.2


class ModelConfig(BaseModel):
    """Complete model architecture configuration."""

    esm2: ESM2Config = Field(default_factory=ESM2Config)
    cnn: CNNConfig = Field(default_factory=CNNConfig)
    bilstm: BiLSTMConfig = Field(default_factory=BiLSTMConfig)
    attention: AttentionConfig = Field(default_factory=AttentionConfig)
    feedforward: FeedForwardConfig = Field(default_factory=FeedForwardConfig)
    q3_head: HeadConfig = Field(default_factory=lambda: HeadConfig(num_classes=3))
    q8_head: HeadConfig = Field(default_factory=lambda: HeadConfig(num_classes=8))


class EarlyStoppingConfig(BaseModel):
    """Early stopping configuration."""

    enabled: bool = True
    patience: int = 15
    monitor: str = "val_q3_accuracy"
    mode: str = "max"
    min_delta: float = 0.001


class LossConfig(BaseModel):
    """Loss function configuration."""

    type: str = "cross_entropy"
    label_smoothing: float = 0.1
    use_class_weights: bool = True
    focal_gamma: float = 2.0
    focal_alpha: list[float] | None = None


class TaskWeightsConfig(BaseModel):
    """Multi-task loss weighting configuration."""

    q3: float = 1.0
    q8: float = 0.5


class CheckpointConfig(BaseModel):
    """Model checkpointing configuration."""

    save_top_k: int = 3
    monitor: str = "val_q3_accuracy"
    mode: str = "max"
    save_dir: str = "models/"


class LoggingConfig(BaseModel):
    """Logging configuration."""

    tensorboard: bool = True
    tensorboard_dir: str = "logs/"
    wandb: bool = False
    wandb_project: str = "protintel"
    log_every_n_steps: int = 50
    log_model_graph: bool = True


class SchedulerParamsConfig(BaseModel):
    """Learning rate scheduler parameters."""

    factor: float = 0.5
    patience: int = 5
    min_lr: float = 1e-7
    T_0: int = 10
    T_mult: int = 2
    max_lr: float = 5e-4
    pct_start: float = 0.3


class OptimizerParamsConfig(BaseModel):
    """Optimizer parameters."""

    betas: list[float] = Field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8


class TrainingConfig(BaseModel):
    """Complete training configuration."""

    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 4
    mixed_precision: bool = True
    optimizer: str = "adamw"
    optimizer_params: OptimizerParamsConfig = Field(default_factory=OptimizerParamsConfig)
    scheduler: str = "reduce_on_plateau"
    scheduler_params: SchedulerParamsConfig = Field(default_factory=SchedulerParamsConfig)
    warmup_epochs: int = 3
    warmup_start_lr: float = 1e-6
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    task_weights: TaskWeightsConfig = Field(default_factory=TaskWeightsConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    seed: int = 42
    deterministic: bool = True


class PreprocessingConfig(BaseModel):
    """Sequence preprocessing configuration."""

    max_sequence_length: int = 512
    sliding_window_overlap: int = 64
    min_sequence_length: int = 10
    nonstandard_policy: str = "replace"
    nonstandard_mapping: dict[str, str] = Field(
        default_factory=lambda: {
            "B": "D",
            "Z": "E",
            "J": "L",
            "O": "K",
            "U": "C",
            "X": "X",
        }
    )


class DataLoaderConfig(BaseModel):
    """DataLoader configuration."""

    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    drop_last: bool = False


class LabelConfig(BaseModel):
    """Label encoding configuration."""

    q3_classes: list[str] = Field(default_factory=lambda: ["H", "E", "C"])
    q8_classes: list[str] = Field(default_factory=lambda: ["H", "E", "G", "I", "B", "T", "S", "C"])
    q3_mapping: dict[str, int] = Field(
        default_factory=lambda: {"H": 0, "E": 1, "C": 2}
    )
    q8_mapping: dict[str, int] = Field(
        default_factory=lambda: {"H": 0, "E": 1, "G": 2, "I": 3, "B": 4, "T": 5, "S": 6, "C": 7}
    )
    q8_to_q3: dict[str, str] = Field(
        default_factory=lambda: {
            "H": "H", "G": "H", "I": "H",
            "E": "E", "B": "E",
            "T": "C", "S": "C", "C": "C",
        }
    )


class DataConfig(BaseModel):
    """Complete data pipeline configuration."""

    raw_dir: str = "datasets/raw"
    processed_dir: str = "datasets/processed"
    embeddings_cache_dir: str = "datasets/processed/embeddings"
    max_samples: int | None = None
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    dataloader: DataLoaderConfig = Field(default_factory=DataLoaderConfig)
    labels: LabelConfig = Field(default_factory=LabelConfig)


class InferenceXAIConfig(BaseModel):
    """XAI settings for inference."""

    integrated_gradients_steps: int = 50
    gradient_shap_samples: int = 20
    attention_rollout_head_fusion: str = "mean"


class InferenceConfig(BaseModel):
    """Complete inference configuration."""

    checkpoint_path: str = "models/best_checkpoint.pt"
    device: str = "cpu"
    max_sequences: int = 50
    max_sequence_length: int = 1024
    timeout_seconds: int = 300
    return_probabilities: bool = True
    return_confidence: bool = True
    return_attention: bool = False
    return_xai: bool = False
    default_xai_method: str = "ig"
    xai: InferenceXAIConfig = Field(default_factory=InferenceXAIConfig)


class ProtIntelConfig(BaseModel):
    """Root configuration object aggregating all sub-configs.

    This is the single top-level config that the entire application uses.
    """

    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)


def load_config(
    model_config: str | Path = "configs/model.yaml",
    training_config: str | Path = "configs/training.yaml",
    data_config: str | Path = "configs/data.yaml",
    inference_config: str | Path = "configs/inference.yaml",
) -> ProtIntelConfig:
    """Load and merge all configuration files into a single ProtIntelConfig.

    Args:
        model_config: Path to model architecture YAML.
        training_config: Path to training hyperparameters YAML.
        data_config: Path to data pipeline YAML.
        inference_config: Path to inference settings YAML.

    Returns:
        A fully validated ProtIntelConfig instance.
    """
    model_raw = load_yaml(model_config).get("model", {})
    training_raw = load_yaml(training_config).get("training", {})
    data_raw = load_yaml(data_config).get("data", {})
    inference_raw = load_yaml(inference_config).get("inference", {})

    # Apply environment variable overrides
    device_override = os.environ.get("DEVICE")
    if device_override:
        inference_raw["device"] = device_override

    model_path_override = os.environ.get("MODEL_PATH")
    if model_path_override:
        inference_raw["checkpoint_path"] = model_path_override

    return ProtIntelConfig(
        model=ModelConfig(**model_raw) if model_raw else ModelConfig(),
        training=TrainingConfig(**training_raw) if training_raw else TrainingConfig(),
        data=DataConfig(**data_raw) if data_raw else DataConfig(),
        inference=InferenceConfig(**inference_raw) if inference_raw else InferenceConfig(),
    )
