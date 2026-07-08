"""CSV/JSON export service for prediction results."""

from __future__ import annotations

import csv
import io
import json
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


def export_to_csv(result: dict[str, Any]) -> str:
    """Export a prediction result to CSV format.

    Creates a table with one row per residue containing position,
    amino acid, Q3 prediction, Q8 prediction, confidence, and
    per-class probabilities.

    Args:
        result: Prediction result dictionary.

    Returns:
        CSV-formatted string.
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    header = [
        "Position", "Amino Acid", "Q3 Prediction", "Q8 Prediction",
        "Confidence", "P(H)", "P(E)", "P(C)",
    ]
    writer.writerow(header)

    sequence = result["sequence"]
    q3_preds = result["q3_prediction"]
    q8_preds = result["q8_prediction"]
    confidence = result["confidence"]
    q3_probs = result["q3_probabilities"]

    for i in range(len(sequence)):
        row = [
            i + 1,
            sequence[i],
            q3_preds[i],
            q8_preds[i],
            f"{confidence[i]:.4f}",
            f"{q3_probs[i][0]:.4f}",
            f"{q3_probs[i][1]:.4f}",
            f"{q3_probs[i][2]:.4f}",
        ]
        writer.writerow(row)

    return output.getvalue()


def export_to_json(result: dict[str, Any], indent: int = 2) -> str:
    """Export a prediction result to formatted JSON.

    Args:
        result: Prediction result dictionary.
        indent: JSON indentation level.

    Returns:
        JSON-formatted string.
    """
    return json.dumps(result, indent=indent, ensure_ascii=False)
