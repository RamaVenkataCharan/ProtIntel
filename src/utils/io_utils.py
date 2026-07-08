"""File I/O helper utilities for ProtIntel.

Provides convenience functions for reading, writing, and managing
data files in various formats (NumPy, PyTorch, JSON, text).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it does not exist.

    Args:
        path: Directory path to create.

    Returns:
        The resolved Path object.
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def save_tensor(tensor: torch.Tensor, path: str | Path) -> None:
    """Save a PyTorch tensor to disk.

    Args:
        tensor: The tensor to save.
        path: Destination file path (should end in .pt).
    """
    save_path = Path(path)
    ensure_dir(save_path.parent)
    torch.save(tensor, str(save_path))
    logger.debug(f"Saved tensor of shape {tensor.shape} to {save_path}")


def load_tensor(path: str | Path, device: str = "cpu") -> torch.Tensor:
    """Load a PyTorch tensor from disk.

    Args:
        path: Path to the .pt file.
        device: Device to map the tensor to.

    Returns:
        The loaded tensor.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    load_path = Path(path)
    if not load_path.exists():
        raise FileNotFoundError(f"Tensor file not found: {load_path}")

    tensor = torch.load(str(load_path), map_location=device, weights_only=True)
    logger.debug(f"Loaded tensor of shape {tensor.shape} from {load_path}")
    return tensor


def save_numpy(array: np.ndarray, path: str | Path, compressed: bool = False) -> None:
    """Save a NumPy array to disk.

    Args:
        array: The array to save.
        path: Destination file path (.npy or .npz).
        compressed: If True, save as compressed .npz format.
    """
    save_path = Path(path)
    ensure_dir(save_path.parent)
    if compressed:
        np.savez_compressed(str(save_path), data=array)
    else:
        np.save(str(save_path), array)
    logger.debug(f"Saved numpy array of shape {array.shape} to {save_path}")


def load_numpy(path: str | Path) -> np.ndarray:
    """Load a NumPy array from disk.

    Supports both .npy and .npz formats. For .npz files, returns
    the first array found.

    Args:
        path: Path to the .npy or .npz file.

    Returns:
        The loaded NumPy array.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    load_path = Path(path)
    if not load_path.exists():
        raise FileNotFoundError(f"Numpy file not found: {load_path}")

    if load_path.suffix == ".npz":
        with np.load(str(load_path)) as data:
            keys = list(data.keys())
            array = data[keys[0]]
    else:
        array = np.load(str(load_path))

    logger.debug(f"Loaded numpy array of shape {array.shape} from {load_path}")
    return array


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Save data as a JSON file.

    Args:
        data: JSON-serializable data.
        path: Destination file path.
        indent: JSON indentation level.
    """
    save_path = Path(path)
    ensure_dir(save_path.parent)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    logger.debug(f"Saved JSON to {save_path}")


def load_json(path: str | Path) -> Any:
    """Load data from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        The parsed JSON data.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    load_path = Path(path)
    if not load_path.exists():
        raise FileNotFoundError(f"JSON file not found: {load_path}")

    with open(load_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def compute_sequence_hash(sequence: str) -> str:
    """Compute a SHA-256 hash of a protein sequence string.

    Used to create unique cache keys for embedding files.

    Args:
        sequence: The amino acid sequence string.

    Returns:
        A hexadecimal hash string (first 16 characters).
    """
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()[:16]


def decompress_gzip(src: str | Path, dst: str | Path) -> Path:
    """Decompress a gzip-compressed file.

    Args:
        src: Path to the .gz file.
        dst: Destination path for the decompressed file.

    Returns:
        Path to the decompressed file.

    Raises:
        FileNotFoundError: If the source file does not exist.
    """
    src_path = Path(src)
    dst_path = Path(dst)

    if not src_path.exists():
        raise FileNotFoundError(f"Gzip file not found: {src_path}")

    ensure_dir(dst_path.parent)
    with gzip.open(str(src_path), "rb") as f_in:
        with open(str(dst_path), "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    logger.info(f"Decompressed {src_path} → {dst_path}")
    return dst_path


def get_file_size_mb(path: str | Path) -> float:
    """Get file size in megabytes.

    Args:
        path: Path to the file.

    Returns:
        File size in MB, or 0.0 if file does not exist.
    """
    file_path = Path(path)
    if not file_path.exists():
        return 0.0
    return file_path.stat().st_size / (1024 * 1024)
