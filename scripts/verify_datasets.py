#!/usr/bin/env python3
"""Dataset verification script for ProtIntel.

Verifies the integrity, shape, and cross-dataset consistency of CullPDB, CB513,
and RS126.
"""

from __future__ import annotations

import gzip
import io
import sys
from pathlib import Path
import numpy as np

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _get_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "configs").is_dir():
            return parent
    return Path.cwd()


PROJECT_ROOT = _get_project_root()
RAW_DIR = PROJECT_ROOT / "datasets" / "raw"


def check_gzip_magic_bytes(filepath: Path) -> bool:
    if not filepath.exists():
        return False
    try:
        with open(filepath, "rb") as f:
            header = f.read(2)
            return header == b"\x1f\x8b"
    except Exception:
        return False


def check_npy_magic_bytes(filepath: Path) -> bool:
    if not filepath.exists():
        return False
    try:
        with open(filepath, "rb") as f:
            header = f.read(6)
            return header == b"\x93NUMPY"
    except Exception:
        return False


def verify_dataset(
    filepath: Path,
    expected_proteins: int | tuple[int, int],
) -> tuple[bool, str, dict]:
    """Verify a single dataset file.

    Returns:
        (success, message, metadata)
    """
    if not filepath.exists():
        return False, "File does not exist", {}

    size_mb = filepath.stat().st_size / (1024 * 1024)
    if size_mb < 1.0:
        return False, f"File is too small ({size_mb:.3f} MB)", {}

    is_gzip = check_gzip_magic_bytes(filepath)
    is_npy = check_npy_magic_bytes(filepath)
    if not (is_gzip or is_npy):
        return False, "Not a valid gzip or numpy file (magic bytes mismatch)", {}

    try:
        if is_gzip:
            with gzip.open(str(filepath), "rb") as f:
                data = np.load(io.BytesIO(f.read()))
        else:
            data = np.load(str(filepath))
    except Exception as e:
        return False, f"Failed to load numpy array: {e}", {}

    # Reshape logic matching ProteinDataset
    orig_shape = data.shape
    if data.ndim == 2:
        num_cols = data.shape[1]
        if num_cols % 700 == 0:
            D = num_cols // 700
            data = data.reshape(-1, 700, D)
        elif data.shape[0] % 700 == 0:
            N = data.shape[0] // 700
            D = data.shape[1]
            data = data.reshape(N, 700, D)
        else:
            D = data.shape[1]
            N = data.shape[0] // 700
            data = data.reshape(-1, 700, D)

    if data.ndim != 3 or data.shape[1] != 700:
        return False, f"Invalid reshaped shape: {data.shape} (expected (N, 700, D))", {}

    N, seq_len, D = data.shape

    # Check protein count range
    if isinstance(expected_proteins, tuple):
        min_p, max_p = expected_proteins
        if not (min_p <= N <= max_p):
            return False, f"Protein count {N} out of range [{min_p}, {max_p}]", {
                "size_mb": size_mb, "shape": data.shape, "dtype": str(data.dtype), "N": N, "D": D
            }
    else:
        if N != expected_proteins:
            return False, f"Expected {expected_proteins} proteins, got {N}", {
                "size_mb": size_mb, "shape": data.shape, "dtype": str(data.dtype), "N": N, "D": D
            }

    # Check feature dimension
    if D not in (56, 57):
        return False, f"Invalid feature dimension: {D} (expected 56 or 57)", {
            "size_mb": size_mb, "shape": data.shape, "dtype": str(data.dtype), "N": N, "D": D
        }

    return True, "OK", {
        "size_mb": size_mb, "shape": data.shape, "dtype": str(data.dtype), "N": N, "D": D
    }


def main() -> None:
    print("=" * 70)
    print("  ProtIntel — Dataset Verification Suite")
    print("=" * 70)
    print(f"  Raw directory: {RAW_DIR}")
    print("=" * 70)
    print()

    required_datasets = {
        "CullPDB": ("cullpdb+profile_6133_filtered.npy.gz", (5500, 6000)),
        "CB513": ("cb513+profile_split1.npy.gz", 514),
        "RS126": ("rs126+profile.npy.gz", 126),
    }

    optional_datasets = {
        "CASP12": ("casp12.fasta", None),
    }

    results = {}
    dimensions = {}
    all_passed = True

    # Validate required datasets
    for name, (filename, expected_p) in required_datasets.items():
        filepath = RAW_DIR / filename
        print(f"[*] Checking {name} ({filename})...")
        success, msg, meta = verify_dataset(filepath, expected_p)
        results[name] = (success, msg, meta)
        if success:
            print(f"    [PASS] Shape: {meta['shape']} | Dtype: {meta['dtype']} | Size: {meta['size_mb']:.1f} MB")
            dimensions[name] = meta["D"]
        else:
            print(f"    [FAIL] {msg}")
            all_passed = False
        print()

    # Validate optional datasets
    for name, (filename, _) in optional_datasets.items():
        filepath = RAW_DIR / filename
        if filepath.exists():
            print(f"[*] Checking optional dataset {name} ({filename})...")
            # For FASTA, just check it exists and has content
            size = filepath.stat().st_size
            if size > 0:
                print(f"    [PASS] Exists, size: {size / 1024:.1f} KB")
            else:
                print(f"    [FAIL] File is empty")
        else:
            print(f"[*] Optional dataset {name} ({filename}) is not present (skipping).")
        print()

    # Cross-dataset consistency check
    if len(dimensions) == len(required_datasets):
        dims = list(dimensions.values())
        if len(set(dims)) > 1:
            print("=" * 70)
            print("  [FAIL] CROSS-DATASET INCONSISTENCY DETECTED!")
            print("  Mixing feature encodings will corrupt training and evaluation:")
            for name, d in dimensions.items():
                print(f"    - {name}: {d} features")
            print("=" * 70)
            all_passed = False
        else:
            print("=" * 70)
            print(f"  [PASS] Cross-dataset consistency verified. Feature dimension: {dims[0]}")
            print("=" * 70)
    else:
        print("=" * 70)
        print("  [FAIL] Missing required datasets. Cannot verify consistency.")
        print("=" * 70)
        all_passed = False

    print("\n  Summary Table:")
    print("  " + "-" * 75)
    print(f"  | {'Dataset':<10} | {'Status':<6} | {'Proteins':<8} | {'Features':<8} | {'Size (MB)':<10} | {'Detail':<16} |")
    print("  " + "-" * 75)
    for name, (success, msg, meta) in results.items():
        status = "PASS" if success else "FAIL"
        if success:
            proteins = str(meta["N"])
            features = str(meta["D"])
            size = f"{meta['size_mb']:.1f}"
            detail = "OK"
        else:
            proteins = "-"
            features = "-"
            size = "-"
            detail = msg[:16]
        print(f"  | {name:<10} | {status:<6} | {proteins:<8} | {features:<8} | {size:<10} | {detail:<16} |")
    print("  " + "-" * 75)

    if all_passed:
        print("\n  >>> ALL REQUIRED DATASETS PASSED VERIFICATION. <<<\n")
        sys.exit(0)
    else:
        print("\n  >>> DATASET VERIFICATION FAILED! <<<\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
