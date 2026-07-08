"""Reproducibility utilities for deterministic training.

Sets random seeds across Python, NumPy, and PyTorch to ensure
reproducible experiments. Optionally enables CUDA deterministic mode.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility across all random sources.

    Args:
        seed: Integer seed value for all random number generators.
        deterministic: If True, enables PyTorch deterministic algorithms
            and sets CUBLAS workspace config. This may reduce performance
            but ensures bitwise reproducibility on CUDA.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError:
            logger.warning(
                "Could not enable fully deterministic algorithms. "
                "Some operations may not have deterministic implementations."
            )

    logger.info(f"Random seed set to {seed} (deterministic={deterministic})")


def get_device(device_str: str = "auto") -> torch.device:
    """Resolve the compute device string to a torch.device.

    Args:
        device_str: Device specification. Valid values are:
            - ``"auto"``: Use CUDA if available, else CPU
            - ``"cpu"``: Force CPU
            - ``"cuda"``: Use default CUDA device
            - ``"cuda:N"``: Use specific CUDA device N

    Returns:
        Resolved ``torch.device`` instance.
    """
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(device)}")
        logger.info(
            f"  VRAM: {torch.cuda.get_device_properties(device).total_mem / 1e9:.1f} GB"
        )

    return device
