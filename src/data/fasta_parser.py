"""FASTA file parsing, writing, and validation utilities for ProtIntel.

Provides functions to read protein sequences from FASTA-formatted files or
raw strings, write them back to disk, and validate sequence content against
the canonical 20 amino acid alphabet.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import TextIO

from src.utils.logger import get_logger

logger = get_logger(__name__)

# The 20 canonical amino acids
CANONICAL_AA: frozenset[str] = frozenset("ACDEFGHIKLMNPQRSTVWY")

# Extended alphabet including common non-standard residues
EXTENDED_AA: frozenset[str] = frozenset("ACDEFGHIKLMNPQRSTVWYBZXJOU")

# Maximum characters per line when writing FASTA
FASTA_LINE_WIDTH: int = 60


def parse_fasta(path: str | Path) -> list[dict[str, str]]:
    """Parse a FASTA file and return a list of sequence records.

    Handles multi-line sequences, blank lines, and duplicate sequence IDs.
    Duplicate IDs are logged as warnings but still included in the output.

    Args:
        path: Path to the FASTA file. Must exist and be readable.

    Returns:
        A list of dictionaries, each containing:
            - ``"id"``: The sequence identifier (text before first whitespace
              on the header line, excluding the ``>`` character).
            - ``"description"``: The full header line text after the ``>``
              character (including the ID).
            - ``"sequence"``: The concatenated amino acid sequence string.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file is empty or contains no valid FASTA records.

    Example:
        >>> records = parse_fasta("proteins.fasta")
        >>> records[0]["id"]
        'sp|P12345|MYG_HUMAN'
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    records = parse_fasta_string(text)

    if not records:
        raise ValueError(f"No valid FASTA records found in: {file_path}")

    logger.info(f"Parsed {len(records)} sequences from {file_path.name}")
    return records


def parse_fasta_string(text: str) -> list[dict[str, str]]:
    """Parse FASTA-formatted text from a raw string.

    This is used by the FastAPI upload endpoint to parse sequences
    submitted directly as text payloads without writing to disk.

    Handles multi-line sequences, blank lines, trailing whitespace,
    and duplicate ID warnings identically to :func:`parse_fasta`.

    Args:
        text: A string containing one or more FASTA records.

    Returns:
        A list of dictionaries with keys ``"id"``, ``"description"``,
        and ``"sequence"``.

    Example:
        >>> records = parse_fasta_string(">seq1 example\\nACDEFG\\nHIKLMN\\n")
        >>> records[0]["sequence"]
        'ACDEFGHIKLMN'
    """
    if not text or not text.strip():
        return []

    records: list[dict[str, str]] = []
    seen_ids: set[str] = set()

    current_id: str | None = None
    current_description: str = ""
    sequence_parts: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()

        # Skip blank lines
        if not stripped:
            continue

        # Skip comment lines
        if stripped.startswith(";"):
            continue

        if stripped.startswith(">"):
            # Flush the previous record if one exists
            if current_id is not None:
                records.append({
                    "id": current_id,
                    "description": current_description,
                    "sequence": "".join(sequence_parts),
                })

            # Parse the new header line
            header = stripped[1:].strip()
            if not header:
                current_id = f"unnamed_{len(records)}"
                current_description = ""
                warnings.warn(
                    f"FASTA header at record {len(records)} is empty; "
                    f"assigned ID '{current_id}'.",
                    stacklevel=2,
                )
            else:
                # ID is everything before the first whitespace
                parts = header.split(None, 1)
                current_id = parts[0]
                current_description = header

            # Check for duplicate IDs
            if current_id in seen_ids:
                warnings.warn(
                    f"Duplicate sequence ID detected: '{current_id}'. "
                    f"All occurrences will be included.",
                    stacklevel=2,
                )
                logger.warning(f"Duplicate sequence ID: {current_id}")
            seen_ids.add(current_id)

            sequence_parts = []
        else:
            # Sequence line: strip whitespace and digits
            cleaned = re.sub(r"[\s\d]", "", stripped)
            if cleaned:
                sequence_parts.append(cleaned.upper())

    # Flush the last record
    if current_id is not None:
        records.append({
            "id": current_id,
            "description": current_description,
            "sequence": "".join(sequence_parts),
        })

    return records


def write_fasta(
    records: list[dict[str, str]],
    path: str | Path,
    line_width: int = FASTA_LINE_WIDTH,
) -> None:
    """Write sequence records to a FASTA-formatted file.

    Each sequence is wrapped at ``line_width`` characters per line
    (default 60), following the standard FASTA convention.

    Args:
        records: A list of dictionaries, each with at least ``"id"``
            and ``"sequence"`` keys. An optional ``"description"`` key
            provides additional header text.
        path: Destination file path. Parent directories are created
            automatically if they do not exist.
        line_width: Maximum number of residue characters per line.
            Must be a positive integer.

    Raises:
        ValueError: If ``records`` is empty or ``line_width`` is not positive.
    """
    if not records:
        raise ValueError("Cannot write an empty list of FASTA records.")
    if line_width < 1:
        raise ValueError(f"line_width must be positive, got {line_width}.")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            _write_single_record(f, record, line_width)

    logger.info(f"Wrote {len(records)} sequences to {out_path.name}")


def _write_single_record(
    f: TextIO,
    record: dict[str, str],
    line_width: int,
) -> None:
    """Write a single FASTA record to an open file handle.

    Args:
        f: Writable text file handle.
        record: Dictionary with ``"id"``, optional ``"description"``,
            and ``"sequence"`` keys.
        line_width: Characters per sequence line.
    """
    description = record.get("description", "")
    seq_id = record.get("id", "unknown")

    # Use the full description if available, otherwise just the ID
    header = description if description else seq_id
    f.write(f">{header}\n")

    sequence = record.get("sequence", "")
    for i in range(0, len(sequence), line_width):
        f.write(sequence[i : i + line_width] + "\n")


def validate_fasta(records: list[dict[str, str]]) -> list[str]:
    """Validate a list of FASTA records and return warning messages.

    Checks for the following issues:
        - Empty sequences (zero length after parsing).
        - Sequences shorter than 10 residues.
        - Non-standard amino acid characters (not in the canonical 20).

    Args:
        records: A list of dictionaries with ``"id"`` and ``"sequence"``
            keys, as returned by :func:`parse_fasta`.

    Returns:
        A list of human-readable warning strings. An empty list indicates
        all records passed validation.

    Example:
        >>> warnings = validate_fasta([{"id": "s1", "sequence": "ACBX"}])
        >>> len(warnings)
        2
    """
    if not records:
        return ["No records to validate."]

    warning_messages: list[str] = []

    for idx, record in enumerate(records):
        seq_id = record.get("id", f"record_{idx}")
        sequence = record.get("sequence", "")

        # Check for empty sequences
        if not sequence:
            warning_messages.append(
                f"[{seq_id}] Sequence is empty (0 residues)."
            )
            continue

        # Check for short sequences
        if len(sequence) < 10:
            warning_messages.append(
                f"[{seq_id}] Sequence is very short ({len(sequence)} residues, "
                f"minimum recommended is 10)."
            )

        # Check for non-standard characters
        non_standard = set(sequence.upper()) - CANONICAL_AA
        if non_standard:
            sorted_chars = sorted(non_standard)
            warning_messages.append(
                f"[{seq_id}] Contains non-standard amino acid characters: "
                f"{', '.join(sorted_chars)}."
            )

    if warning_messages:
        logger.warning(
            f"FASTA validation produced {len(warning_messages)} warning(s)."
        )
    else:
        logger.info(
            f"All {len(records)} FASTA records passed validation."
        )

    return warning_messages
