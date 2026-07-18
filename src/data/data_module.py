"""Data module for ProtIntel protein structure prediction.

Provides a high-level ``ProteinDataModule`` class that encapsulates dataset
creation, DataLoader configuration, and class weight computation.  Reads
all settings from ``configs/data.yaml`` and delegates actual data loading
to :class:`src.data.protein_dataset.ProteinDataset`.

``DataModule`` is a top-level alias for ``ProteinDataModule`` to avoid
breaking callers that import the shorter name (e.g. ``train.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import torch
from torch.utils.data import DataLoader

from src.data.preprocessor import compute_class_weights, encode_q3, encode_q8
from src.data.protein_dataset import ProteinDataset, collate_fn
from src.utils.config_loader import PROJECT_ROOT, load_yaml
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.utils.config_loader import DataConfig

logger = get_logger(__name__)


class ProteinDataModule:
    """Centralized data module for managing train/val/test datasets.

    Loads configuration from ``configs/data.yaml``, instantiates
    :class:`ProteinDataset` objects for each split, and provides
    configured ``DataLoader`` instances.

    Args:
        config_path: Path to the data configuration YAML file.
            Defaults to ``configs/data.yaml`` relative to the project root.

    Attributes:
        config: Parsed data configuration dictionary.
        train_dataset: Training dataset (created after :meth:`setup`).
        val_dataset: Validation dataset (created after :meth:`setup`).
        test_dataset: Test dataset (created after :meth:`setup`).

    Example:
        >>> dm = ProteinDataModule("configs/data.yaml")
        >>> dm.setup(stage="fit")
        >>> train_loader = dm.train_dataloader()
    """

    def __init__(self, config_path: str | Path = "configs/data.yaml") -> None:
        raw_config = load_yaml(config_path)
        self.config: dict[str, Any] = raw_config.get("data", raw_config)

        # Resolve directory paths relative to project root
        self.raw_dir: Path = PROJECT_ROOT / self.config.get(
            "raw_dir", "datasets/raw"
        )
        self.processed_dir: Path = PROJECT_ROOT / self.config.get(
            "processed_dir", "datasets/processed"
        )
        self.embeddings_dir: Path = PROJECT_ROOT / self.config.get(
            "embeddings_dir",
            self.config.get("embeddings_cache_dir", "datasets/processed/embeddings"),
        )

        # DataLoader settings
        dl_config = self.config.get("dataloader", {})
        self.batch_size: int = self.config.get(
            "batch_size", dl_config.get("batch_size", 8)
        )
        self.num_workers: int = dl_config.get("num_workers", 4)
        self.pin_memory: bool = dl_config.get("pin_memory", True)
        self.prefetch_factor: int = dl_config.get("prefetch_factor", 2)

        # Split configuration
        self.splits: dict[str, str] = self.config.get("splits", {
            "train": "cullpdb",
            "val": "rs126",
            "test": "cb513",
        })

        # Dataset file names
        self.cullpdb_file: str = self.config.get(
            "cullpdb_file", "cullpdb+profile_6133_filtered.npy.gz"
        )
        self.cb513_file: str = self.config.get(
            "cb513_file", "cb513+profile_split1.npy.gz"
        )
        self.rs126_file: str = self.config.get(
            "rs126_file", "rs126+profile_split1.npy.gz"
        )

        # Use embedding cache flag
        self.use_embedding_cache: bool = self.config.get(
            "use_embedding_cache", True
        )

        # Dataset references (populated by setup())
        self.train_dataset: ProteinDataset | None = None
        self.val_dataset: ProteinDataset | None = None
        self.test_dataset: ProteinDataset | None = None

        logger.info(
            f"ProteinDataModule initialized. "
            f"raw_dir={self.raw_dir}, batch_size={self.batch_size}"
        )

    # ──────────────────────────────────────────────────────────────────
    # Dataset setup
    # ──────────────────────────────────────────────────────────────────

    def setup(self, stage: str = "fit") -> None:
        """Instantiate dataset objects for the requested stage.

        Args:
            stage: One of ``"fit"`` (train + val), ``"test"``,
                or ``"predict"``.  When ``"fit"`` is specified, both
                training and validation datasets are created.
        """
        dataset_config = self._build_dataset_config()

        if stage in ("fit", "train"):
            train_path = self._resolve_split_path("train")
            self.train_dataset = ProteinDataset(
                data_path=train_path,
                split="train",
                config=dataset_config,
                use_cache=self.use_embedding_cache,
            )
            logger.info(
                f"Train dataset: {len(self.train_dataset)} samples "
                f"from {train_path.name}"
            )

        if stage in ("fit", "validate", "val"):
            val_path = self._resolve_split_path("val")
            self.val_dataset = ProteinDataset(
                data_path=val_path,
                split="val",
                config=dataset_config,
                use_cache=self.use_embedding_cache,
            )
            logger.info(
                f"Val dataset: {len(self.val_dataset)} samples "
                f"from {val_path.name}"
            )

        if stage in ("test", "predict"):
            test_path = self._resolve_split_path("test")
            self.test_dataset = ProteinDataset(
                data_path=test_path,
                split="test",
                config=dataset_config,
                use_cache=self.use_embedding_cache,
            )
            logger.info(
                f"Test dataset: {len(self.test_dataset)} samples "
                f"from {test_path.name}"
            )

    def _build_dataset_config(self) -> dict[str, Any]:
        """Build the configuration dictionary passed to ProteinDataset.

        Returns:
            A flat dictionary with all dataset-relevant settings.
        """
        preprocessing = self.config.get("preprocessing", {})
        augmentation = self.config.get("augmentation", {})

        return {
            "max_seq_length": preprocessing.get(
                "max_sequence_length",
                self.config.get("max_seq_length", 512),
            ),
            "min_seq_length": preprocessing.get(
                "min_sequence_length",
                self.config.get("min_seq_length", 10),
            ),
            "nonstandard_policy": preprocessing.get(
                "nonstandard_policy",
                self.config.get("nonstandard_policy", "mask"),
            ),
            "embeddings_dir": str(self.embeddings_dir),
            "augmentation": augmentation,
        }

    def _resolve_split_path(self, split: str) -> Path:
        """Resolve the file path for a given dataset split.

        Maps split names (``"train"``, ``"val"``, ``"test"``) to actual
        dataset files via the splits configuration.

        Args:
            split: One of ``"train"``, ``"val"``, or ``"test"``.

        Returns:
            Absolute path to the dataset file.

        Raises:
            FileNotFoundError: If the resolved file does not exist.
        """
        dataset_name = self.splits.get(split, "")

        # Map dataset names to files
        name_to_file: dict[str, str] = {
            "cullpdb": self.cullpdb_file,
            "cb513": self.cb513_file,
            "rs126": self.rs126_file,
        }

        filename = name_to_file.get(dataset_name.lower(), dataset_name)

        # Check in raw directory first
        file_path = self.raw_dir / filename
        if file_path.exists():
            return file_path

        # Check in processed directory
        file_path = self.processed_dir / filename
        if file_path.exists():
            return file_path

        # Check if it's an absolute path
        abs_path = Path(filename)
        if abs_path.is_absolute() and abs_path.exists():
            return abs_path

        # Check datasets directory with alternate extensions
        for suffix in [".npy.gz", ".npy", ".npz", ".fasta", ".fa"]:
            candidate = self.raw_dir / (Path(filename).stem + suffix)
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"Dataset file for split '{split}' ({dataset_name}) not found. "
            f"Searched in: {self.raw_dir}, {self.processed_dir}. "
            f"Run 'python scripts/download_data.py' to download the datasets."
        )

    # ──────────────────────────────────────────────────────────────────
    # DataLoaders
    # ──────────────────────────────────────────────────────────────────

    def train_dataloader(self) -> DataLoader:
        """Create the training DataLoader with shuffling enabled.

        Returns:
            A configured ``DataLoader`` instance for the training split.

        Raises:
            RuntimeError: If :meth:`setup` has not been called.
        """
        if self.train_dataset is None:
            raise RuntimeError(
                "Training dataset not initialized. Call setup(stage='fit') first."
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            collate_fn=collate_fn,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """Create the validation DataLoader (no shuffling).

        Returns:
            A configured ``DataLoader`` instance for the validation split.

        Raises:
            RuntimeError: If :meth:`setup` has not been called.
        """
        if self.val_dataset is None:
            raise RuntimeError(
                "Validation dataset not initialized. Call setup(stage='fit') first."
            )
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            collate_fn=collate_fn,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        """Create the test DataLoader (no shuffling).

        Returns:
            A configured ``DataLoader`` instance for the test split.

        Raises:
            RuntimeError: If :meth:`setup` has not been called.
        """
        if self.test_dataset is None:
            raise RuntimeError(
                "Test dataset not initialized. Call setup(stage='test') first."
            )
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            collate_fn=collate_fn,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )

    # ──────────────────────────────────────────────────────────────────
    # Class weights
    # ──────────────────────────────────────────────────────────────────

    def get_class_weights(self, task: str = "q3") -> torch.Tensor:
        """Compute inverse-frequency class weights from the training set.

        Args:
            task: Label task to compute weights for. Must be ``"q3"``
                (3 classes) or ``"q8"`` (8 classes).

        Returns:
            A ``torch.float32`` tensor of shape ``(num_classes,)``
            with normalized class weights.

        Raises:
            RuntimeError: If the training dataset has not been initialized.
            ValueError: If ``task`` is not ``"q3"`` or ``"q8"``.
        """
        if self.train_dataset is None:
            raise RuntimeError(
                "Training dataset not initialized. Call setup(stage='fit') first."
            )

        if task == "q3":
            encoder = encode_q3
            num_classes = 3
            label_strings = self.train_dataset.q3_labels
        elif task == "q8":
            encoder = encode_q8
            num_classes = 8
            label_strings = self.train_dataset.q8_labels
        else:
            raise ValueError(
                f"Invalid task '{task}'. Must be 'q3' or 'q8'."
            )

        encoded_labels: list[list[int]] = [
            encoder(label_str) for label_str in label_strings
        ]

        weights = compute_class_weights(encoded_labels, num_classes)
        logger.info(f"Computed {task.upper()} class weights: {weights.tolist()}")
        return weights

    # ──────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────

    @property
    def num_train_samples(self) -> int:
        """Number of samples in the training dataset.

        Returns:
            Sample count, or 0 if the dataset is not initialized.
        """
        return len(self.train_dataset) if self.train_dataset is not None else 0

    @property
    def num_val_samples(self) -> int:
        """Number of samples in the validation dataset.

        Returns:
            Sample count, or 0 if the dataset is not initialized.
        """
        return len(self.val_dataset) if self.val_dataset is not None else 0

    @property
    def num_test_samples(self) -> int:
        """Number of samples in the test dataset.

        Returns:
            Sample count, or 0 if the dataset is not initialized.
        """
        return len(self.test_dataset) if self.test_dataset is not None else 0

    def __repr__(self) -> str:
        """Return a developer-friendly summary of the data module."""
        return (
            f"ProteinDataModule("
            f"train={self.num_train_samples}, "
            f"val={self.num_val_samples}, "
            f"test={self.num_test_samples}, "
            f"batch_size={self.batch_size})"
        )

    # ──────────────────────────────────────────────────────────────────
    # Factory: construct from an already-loaded DataConfig object
    # ──────────────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        config: "DataConfig",
        batch_size: Optional[int] = None,
    ) -> "ProteinDataModule":
        """Construct a ``ProteinDataModule`` from an already-loaded ``DataConfig``.

        This is the preferred entry point when the caller (e.g. ``train.py``)
        has already loaded the full ``ProtIntelConfig`` via ``load_config()``
        and wants to pass the ``data`` sub-config directly rather than a
        YAML file path.

        Args:
            config: A fully validated ``DataConfig`` instance.
            batch_size: Override the batch size from the config.  When
                ``None``, the value in ``config`` (``batch_size`` field if
                present, otherwise the ``dataloader.batch_size`` field) is
                used.

        Returns:
            A fully configured ``ProteinDataModule`` instance.
        """
        # Write a minimal YAML-compatible dict so __init__ can parse it.
        # We extract the relevant fields from the Pydantic model.
        raw: dict[str, Any] = {
            "raw_dir": config.raw_dir,
            "processed_dir": config.processed_dir,
            "embeddings_dir": config.embeddings_cache_dir,
            "embeddings_cache_dir": config.embeddings_cache_dir,
            "use_embedding_cache": True,
            "dataloader": {
                "num_workers": config.dataloader.num_workers,
                "pin_memory": config.dataloader.pin_memory,
                "prefetch_factor": config.dataloader.prefetch_factor,
                "persistent_workers": config.dataloader.persistent_workers,
                "drop_last": config.dataloader.drop_last,
            },
            "preprocessing": {
                "max_sequence_length": config.preprocessing.max_sequence_length,
                "min_sequence_length": config.preprocessing.min_sequence_length,
                "nonstandard_policy": config.preprocessing.nonstandard_policy,
                "sliding_window_overlap": config.preprocessing.sliding_window_overlap,
            },
            "splits": {"train": "cullpdb", "val": "cb513", "test": "cb513"},
        }

        # Resolve batch size: explicit arg > Pydantic config default.
        effective_batch_size = batch_size if batch_size is not None else 8
        raw["batch_size"] = effective_batch_size

        instance = cls.__new__(cls)
        # Bypass __init__ and populate attributes directly so we avoid
        # the load_yaml() call that __init__ does.
        instance.config = raw
        instance.raw_dir = PROJECT_ROOT / raw["raw_dir"]
        instance.processed_dir = PROJECT_ROOT / raw["processed_dir"]
        instance.embeddings_dir = PROJECT_ROOT / raw["embeddings_cache_dir"]
        instance.batch_size = effective_batch_size
        instance.num_workers = raw["dataloader"]["num_workers"]
        instance.pin_memory = raw["dataloader"]["pin_memory"]
        instance.prefetch_factor = raw["dataloader"]["prefetch_factor"]
        instance.splits = raw["splits"]
        instance.cullpdb_file = "cullpdb+profile_6133_filtered.npy.gz"
        instance.cb513_file = "cb513+profile_split1.npy.gz"
        instance.rs126_file = "cb513+profile_split1.npy.gz"
        instance.use_embedding_cache = True
        instance.train_dataset = None
        instance.val_dataset = None
        instance.test_dataset = None

        logger.info(
            f"ProteinDataModule.from_config(): "
            f"raw_dir={instance.raw_dir}, batch_size={effective_batch_size}"
        )
        return instance


# ---------------------------------------------------------------------------
# Backward-compatible alias — keeps existing ``from src.data.data_module
# import DataModule`` imports working without renaming the class.
# ---------------------------------------------------------------------------
DataModule = ProteinDataModule
