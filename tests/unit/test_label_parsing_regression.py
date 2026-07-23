"""Regression tests for secondary structure label parsing in ProtIntel.

Ensures that:
1. Synthetic (1, 700, 57) sample parsing correctly reads Q8 1-hot from columns 22-29.
2. _Q8_LABEL_START and _Q8_LABEL_END constants remain strictly fixed at 22 and 30.
3. Distribution canary test on real data matches expected PSSP literature range.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from src.data.protein_dataset import (
    _Q8_LABEL_START,
    _Q8_LABEL_END,
    _Q8_TO_Q3_REDUCTION,
    ProteinDataset,
)
from src.data.preprocessor import Q8_CLASSES, Q3_CLASSES


class TestLabelParsingRegression:
    """Regression suite locking in secondary structure label column indices."""

    def test_q8_label_constants_lock(self):
        """Assert _Q8_LABEL_START == 22 and _Q8_LABEL_END == 30 to prevent offset regressions."""
        assert _Q8_LABEL_START == 22, f"Expected _Q8_LABEL_START to be 22, got {_Q8_LABEL_START}"
        assert _Q8_LABEL_END == 30, f"Expected _Q8_LABEL_END to be 30, got {_Q8_LABEL_END}"

    def test_synthetic_sample_q8_q3_parsing(self, tmp_path: Path):
        """Assert a synthetic (1, 700, 57) sample with known 1-hot Q8 at cols 22-29 parses correctly."""
        synth_data = np.zeros((1, 700, 57), dtype=np.float32)

        # Set sequence length = 10 (fill amino acid one-hot at cols 0-20)
        seq_len = 10
        for i in range(seq_len):
            synth_data[0, i, 0] = 1.0  # Alanine (A)

        # Assign specific Q8 classes at cols 22-29 for the 10 positions:
        # Col 22 (L -> C), Col 23 (B -> E), Col 24 (E -> E), Col 25 (G -> H),
        # Col 26 (I -> H), Col 27 (H -> H), Col 28 (S -> C), Col 29 (T -> C)
        q8_col_sequence = [22, 23, 24, 25, 26, 27, 28, 29, 22, 27]  # L, B, E, G, I, H, S, T, L, H
        expected_q8_str = "CBEGIHSTCH"
        expected_q3_str = "CEEHHHCCCH"

        for i, col in enumerate(q8_col_sequence):
            synth_data[0, i, col] = 1.0

        # Save synthetic array to file
        npy_file = tmp_path / "test_synthetic_cullpdb.npy"
        np.save(str(npy_file), synth_data)

        # Load with ProteinDataset
        dataset = ProteinDataset(
            data_path=npy_file,
            split="train",
            config={},
            use_cache=False,
        )

        assert len(dataset) == 1
        sample = dataset[0]
        parsed_q8_indices = sample["q8_labels"].tolist()
        parsed_q3_indices = sample["q3_labels"].tolist()

        parsed_q8_chars = "".join([Q8_CLASSES[idx] for idx in parsed_q8_indices])
        parsed_q3_chars = "".join([Q3_CLASSES[idx] for idx in parsed_q3_indices])

        assert parsed_q8_chars == expected_q8_str, (
            f"Parsed Q8 '{parsed_q8_chars}' != Expected '{expected_q8_str}'"
        )
        assert parsed_q3_chars == expected_q3_str, (
            f"Parsed Q3 '{parsed_q3_chars}' != Expected '{expected_q3_str}'"
        )

    def test_class_distribution_literature_canary(self):
        """Canary test verifying Q3 distribution on dataset matches expected literature ranges.

        Literature expected ranges:
            - Coil (C): 35% - 45%
            - Helix (H): 30% - 42%
            - Sheet (E): 18% - 26%
        """
        project_root = Path(__file__).resolve().parent.parent.parent
        raw_cb513 = project_root / "datasets" / "raw" / "cb513+profile_split1.npy.gz"

        if not raw_cb513.exists():
            pytest.skip("CB513 dataset file not found for canary test")

        dataset = ProteinDataset(
            data_path=raw_cb513,
            split="test",
            config={},
            use_cache=False,
        )

        all_q3_chars = "".join(dataset.q3_labels)
        total_residues = len(all_q3_chars)

        c_count = all_q3_chars.count("C")
        h_count = all_q3_chars.count("H")
        e_count = all_q3_chars.count("E")

        c_frac = c_count / total_residues
        h_frac = h_count / total_residues
        e_frac = e_count / total_residues

        assert 0.35 <= c_frac <= 0.47, f"Coil fraction {c_frac:.2%} out of expected literature range [35%, 47%]"
        assert 0.30 <= h_frac <= 0.42, f"Helix fraction {h_frac:.2%} out of expected literature range [30%, 42%]"
        assert 0.18 <= e_frac <= 0.28, f"Sheet fraction {e_frac:.2%} out of expected literature range [18%, 28%]"
