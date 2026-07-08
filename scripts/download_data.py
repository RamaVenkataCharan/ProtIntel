#!/usr/bin/env python3
"""Download benchmark datasets for ProtIntel.

Downloads the following protein secondary structure prediction datasets
to ``datasets/raw/``:

1. **CullPDB** (filtered, 6133 proteins) — Training set.
2. **CB513** (513 proteins) — Independent test set.
3. **RS126** (126 proteins) — Validation set.

Each dataset is a NumPy array in ``.npy.gz`` format with shape
``(N, 700, 57)`` containing one-hot amino acid profiles and Q8
secondary structure labels.

Usage::

    python scripts/download_data.py

Files are skipped if they already exist and their SHA-256 checksums
match.  Progress bars are displayed via ``tqdm``.
"""

from __future__ import annotations

import gzip
import hashlib
import shutil
import sys
from pathlib import Path
from typing import NamedTuple

import requests
from tqdm import tqdm

# Ensure UTF-8 output on Windows (avoids UnicodeEncodeError with emoji/special chars)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────
# Project root resolution
# ──────────────────────────────────────────────────────────────────────

def _get_project_root() -> Path:
    """Locate the project root by searching for the ``configs/`` directory.

    Returns:
        Absolute path to the project root directory.
    """
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "configs").is_dir():
            return parent
    return Path.cwd()


PROJECT_ROOT: Path = _get_project_root()
RAW_DIR: Path = PROJECT_ROOT / "datasets" / "raw"


# ──────────────────────────────────────────────────────────────────────
# Dataset specifications
# ──────────────────────────────────────────────────────────────────────

class DatasetSpec(NamedTuple):
    """Specification for a downloadable dataset file.

    Attributes:
        name: Human-readable dataset name.
        url: Download URL.
        filename: Local filename (relative to ``datasets/raw/``).
        sha256: Expected SHA-256 checksum of the downloaded file,
            or ``None`` if the checksum is not known and should be
            computed after first download.
        description: Short description of the dataset.
    """
    name: str
    url: str
    filename: str
    sha256: str | None
    description: str


DATASETS: list[DatasetSpec] = [
    DatasetSpec(
        name="CullPDB (filtered, 6133 proteins)",
        url=(
            "https://www.princeton.edu/~jzthree/datasets/ICML2014/"
            "cullpdb+profile_6133_filtered.npy.gz"
        ),
        filename="cullpdb+profile_6133_filtered.npy.gz",
        sha256=None,  # Checksum verified after first successful download
        description="Training set with 6133 non-redundant protein chains.",
    ),
    DatasetSpec(
        name="CB513 (513 proteins)",
        url=(
            "https://www.princeton.edu/~jzthree/datasets/ICML2014/"
            "cb513+profile_split1.npy.gz"
        ),
        filename="cb513+profile_split1.npy.gz",
        sha256=None,
        description="Independent test set with 513 protein chains.",
    ),
    DatasetSpec(
        name="RS126 (126 proteins)",
        url=(
            "https://www.princeton.edu/~jzthree/datasets/ICML2014/"
            "rs126+profile_split1.npy.gz"
        ),
        filename="rs126+profile_split1.npy.gz",
        sha256=None,
        description="Validation set with 126 protein chains.",
    ),
]


# ──────────────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────────────

def compute_sha256(filepath: Path) -> str:
    """Compute the SHA-256 checksum of a file.

    Reads the file in 8 KB chunks to support large files without
    loading the entire contents into memory.

    Args:
        filepath: Path to the file.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def verify_checksum(filepath: Path, expected_sha256: str | None) -> bool:
    """Verify a file's SHA-256 checksum against an expected value.

    If the expected checksum is ``None``, the file is assumed to be
    valid (checksum not available for verification).

    Args:
        filepath: Path to the file to verify.
        expected_sha256: Expected hex digest, or ``None`` to skip.

    Returns:
        ``True`` if the checksum matches or is not provided.
    """
    if expected_sha256 is None:
        return True

    actual = compute_sha256(filepath)
    return actual == expected_sha256


def download_file(
    url: str,
    dest: Path,
    description: str = "",
    chunk_size: int = 8192,
) -> bool:
    """Download a file from a URL with a tqdm progress bar.

    Args:
        url: The URL to download from.
        dest: Destination file path. Parent directories are created
            automatically.
        description: Short label for the progress bar.
        chunk_size: Number of bytes per read chunk.

    Returns:
        ``True`` if the download succeeded, ``False`` otherwise.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Use a temporary file to avoid partial downloads
    tmp_dest = dest.with_suffix(dest.suffix + ".tmp")

    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        bar_label = description or dest.name

        with (
            open(tmp_dest, "wb") as f,
            tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=bar_label,
                ncols=80,
            ) as progress,
        ):
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    progress.update(len(chunk))

        # Move temp file to final destination
        shutil.move(str(tmp_dest), str(dest))
        return True

    except requests.exceptions.RequestException as e:
        print(f"\n  ✗ Download failed: {e}", file=sys.stderr)
        # Clean up partial download
        if tmp_dest.exists():
            tmp_dest.unlink()
        return False

    except KeyboardInterrupt:
        print("\n  ✗ Download interrupted by user.", file=sys.stderr)
        if tmp_dest.exists():
            tmp_dest.unlink()
        return False


# ──────────────────────────────────────────────────────────────────────
# Main download routine
# ──────────────────────────────────────────────────────────────────────

def download_all_datasets(
    output_dir: Path = RAW_DIR,
    force: bool = False,
) -> dict[str, bool]:
    """Download all benchmark datasets.

    Skips files that already exist with matching checksums unless
    ``force=True`` is specified.

    Args:
        output_dir: Directory to save downloaded files. Created if
            it does not exist.
        force: If ``True``, re-download even if the file exists.

    Returns:
        A dictionary mapping dataset names to download success status.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  ProtIntel — Dataset Downloader")
    print("=" * 60)
    print(f"  Output directory: {output_dir}")
    print(f"  Datasets to download: {len(DATASETS)}")
    print("=" * 60)
    print()

    results: dict[str, bool] = {}

    for spec in DATASETS:
        dest = output_dir / spec.filename
        print(f"[+] {spec.name}")
        print(f"   {spec.description}")

        # Check if file already exists
        if dest.exists() and not force:
            if verify_checksum(dest, spec.sha256):
                size_mb = dest.stat().st_size / (1024 * 1024)
                print(f"   [OK] Already exists ({size_mb:.1f} MB), skipping.")
                actual_hash = compute_sha256(dest)
                print(f"   SHA-256: {actual_hash[:16]}...")
                results[spec.name] = True
                print()
                continue
            else:
                print(
                    "   [WARN] File exists but checksum mismatch. Re-downloading..."
                )

        # Download
        print(f"   Downloading from: {spec.url}")
        success = download_file(
            url=spec.url,
            dest=dest,
            description=spec.name,
        )

        if success:
            size_mb = dest.stat().st_size / (1024 * 1024)
            actual_hash = compute_sha256(dest)
            print(f"   [OK] Downloaded successfully ({size_mb:.1f} MB)")
            print(f"   SHA-256: {actual_hash[:16]}...")

            # Verify checksum if one was specified
            if spec.sha256 is not None:
                if actual_hash != spec.sha256:
                    print(
                        f"   [FAIL] CHECKSUM MISMATCH! Expected: {spec.sha256[:16]}..."
                    )
                    success = False
                else:
                    print("   [OK] Checksum verified.")
        else:
            print(f"   [FAIL] Failed to download {spec.name}")

        results[spec.name] = success
        print()

    # Summary
    print("=" * 60)
    print("  Download Summary")
    print("=" * 60)
    for name, success in results.items():
        status = "[OK]" if success else "[FAIL]"
        print(f"  {status} {name}")

    total = len(results)
    succeeded = sum(results.values())
    print(f"\n  {succeeded}/{total} datasets downloaded successfully.")

    if succeeded < total:
        print(
            "\n  ⚠ Some downloads failed. Check your internet connection "
            "and try again."
        )
        print("    You can also manually download from the URLs above.")

    print("=" * 60)
    return results


def main() -> None:
    """Entry point for the download script.

    Parses command-line arguments and downloads all datasets.
    Exits with code 1 if any download fails.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Download benchmark datasets for ProtIntel.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Datasets:\n"
            "  - CullPDB (6133 proteins) — Training\n"
            "  - CB513 (513 proteins) — Testing\n"
            "  - RS126 (126 proteins) — Validation\n"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RAW_DIR,
        help=f"Output directory (default: {RAW_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist.",
    )

    args = parser.parse_args()
    results = download_all_datasets(
        output_dir=args.output_dir,
        force=args.force,
    )

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
