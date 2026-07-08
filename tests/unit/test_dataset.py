"""Unit tests for the ProtIntel data pipeline.

Covers the FASTA parser, sequence preprocessor, augmentation utilities,
and the protein dataset collation function with comprehensive edge-case
handling and round-trip verification.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from src.data.augmentation import (
    AugmentationPipeline,
    random_mask,
    reverse_complement_protein,
    subsequence_crop,
)
from src.data.fasta_parser import (
    parse_fasta,
    parse_fasta_string,
    validate_fasta,
    write_fasta,
)
from src.data.preprocessor import (
    chunk_sequence,
    clean_sequence,
    compute_class_weights,
    decode_q3,
    decode_q8,
    encode_q3,
    encode_q8,
    handle_nonstandard,
)
from src.data.protein_dataset import collate_fn


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def sample_fasta_text() -> str:
    """Multi-line FASTA string with two sequences."""
    return textwrap.dedent("""\
        >sp|P12345|MYG_HUMAN Myoglobin
        ACDEFGHIKL
        MNPQRSTVWY
        >sp|P67890|HBB_HUMAN Hemoglobin beta
        MVLSPADKTN
        VKAAWGKVGA
        HAGEYGAEAL
    """)


@pytest.fixture
def sample_records() -> list[dict[str, str]]:
    """Pre-built FASTA records for write/validate tests."""
    return [
        {
            "id": "protein_1",
            "description": "protein_1 test protein alpha",
            "sequence": "ACDEFGHIKLMNPQRSTVWY",
        },
        {
            "id": "protein_2",
            "description": "protein_2 test protein beta",
            "sequence": "MVLSPADKTNVKAAWGKVGAHAGEYGAEAL",
        },
    ]


@pytest.fixture
def short_sequence() -> str:
    """A short canonical amino acid sequence."""
    return "ACDEFGHIKL"


@pytest.fixture
def short_labels_q3() -> str:
    """Q3 labels matching the short sequence."""
    return "HHEECCCHHE"


@pytest.fixture
def short_labels_q8() -> str:
    """Q8 labels matching the short sequence."""
    return "HHEEGBTSCH"


@pytest.fixture
def long_sequence() -> str:
    """A 600-residue sequence for chunking tests."""
    return "ACDEFGHIKL" * 60


@pytest.fixture
def long_labels() -> str:
    """Q3 labels for the 600-residue sequence."""
    return "HHEECCHHEC" * 60


@pytest.fixture
def mock_tokenizer() -> MagicMock:
    """Mock ESM-2 tokenizer that returns ASCII-based token IDs."""
    tokenizer = MagicMock()

    def _tokenize(
        sequence: str,
        return_tensors: str = "pt",
        padding: bool = False,
        truncation: bool = True,
        max_length: int = 514,
    ) -> dict[str, torch.Tensor]:
        # Generate token IDs from ASCII values (simple mock)
        token_ids = [ord(c) for c in sequence]
        attention = [1] * len(token_ids)
        return {
            "input_ids": torch.tensor([token_ids], dtype=torch.long),
            "attention_mask": torch.tensor([attention], dtype=torch.long),
        }

    tokenizer.side_effect = _tokenize
    tokenizer.__call__ = _tokenize
    return tokenizer


def _make_sample(
    sequence: str,
    q3_labels: str,
    q8_labels: str,
    protein_id: str,
) -> dict[str, Any]:
    """Create a mock dataset sample dictionary for collate_fn tests.

    Simulates the output of ProteinDataset.__getitem__ using ASCII-based
    token IDs instead of a real ESM-2 tokenizer.

    Args:
        sequence: Amino acid sequence string.
        q3_labels: Q3 label string.
        q8_labels: Q8 label string.
        protein_id: Identifier string.

    Returns:
        A sample dictionary matching the collate_fn expected format.
    """
    input_ids = torch.tensor([ord(c) for c in sequence], dtype=torch.long)
    attention_mask = torch.ones(len(sequence), dtype=torch.long)
    q3_encoded = encode_q3(q3_labels)
    q8_encoded = encode_q8(q8_labels)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "q3_labels": torch.tensor(q3_encoded, dtype=torch.long),
        "q8_labels": torch.tensor(q8_encoded, dtype=torch.long),
        "sequence": sequence,
        "protein_id": protein_id,
        "length": len(sequence),
    }


# ======================================================================
# FASTA Parser Tests
# ======================================================================

class TestFastaParser:
    """Tests for the FASTA parsing module."""

    def test_parse_fasta_multiline(
        self, sample_fasta_text: str, tmp_path: Path
    ) -> None:
        """Multi-line sequences are assembled correctly into a single string."""
        fasta_file = tmp_path / "test.fasta"
        fasta_file.write_text(sample_fasta_text, encoding="utf-8")

        records = parse_fasta(fasta_file)

        assert len(records) == 2
        assert records[0]["id"] == "sp|P12345|MYG_HUMAN"
        assert records[0]["sequence"] == "ACDEFGHIKLMNPQRSTVWY"
        assert len(records[0]["sequence"]) == 20
        assert records[1]["id"] == "sp|P67890|HBB_HUMAN"
        assert records[1]["sequence"] == "MVLSPADKTNVKAAWGKVGAHAGEYGAEAL"
        assert len(records[1]["sequence"]) == 30

    def test_parse_fasta_string(self, sample_fasta_text: str) -> None:
        """Parsing from a raw string produces the same result as file parsing."""
        records = parse_fasta_string(sample_fasta_text)

        assert len(records) == 2
        assert records[0]["sequence"] == "ACDEFGHIKLMNPQRSTVWY"
        assert records[1]["sequence"] == "MVLSPADKTNVKAAWGKVGAHAGEYGAEAL"
        assert "description" in records[0]
        assert "MYG_HUMAN" in records[0]["description"]

    def test_parse_fasta_empty_sequence(self) -> None:
        """Empty sequence generates a record with zero-length sequence string."""
        text = ">empty_protein\n\n>valid_protein\nACDEFGHIKL\n"
        records = parse_fasta_string(text)

        assert len(records) == 2
        empty_rec = records[0]
        assert empty_rec["id"] == "empty_protein"
        assert empty_rec["sequence"] == ""

        warnings = validate_fasta(records)
        # Should warn about the empty sequence
        assert any("empty" in w.lower() or "0 residues" in w for w in warnings)

    def test_parse_fasta_blank_lines_and_comments(self) -> None:
        """Blank lines and comment lines (starting with ';') are skipped."""
        text = textwrap.dedent("""\
            ; This is a comment
            >seq1

            ACDEF

            ; another comment
            GHIKL
        """)
        records = parse_fasta_string(text)

        assert len(records) == 1
        assert records[0]["sequence"] == "ACDEFGHIKL"

    def test_parse_fasta_duplicate_ids(self) -> None:
        """Duplicate IDs produce warnings but all records are kept."""
        text = ">dup_id\nACDEF\n>dup_id\nGHIKL\n"

        with pytest.warns(UserWarning, match="Duplicate"):
            records = parse_fasta_string(text)

        assert len(records) == 2
        assert records[0]["id"] == "dup_id"
        assert records[1]["id"] == "dup_id"

    def test_parse_fasta_file_not_found(self, tmp_path: Path) -> None:
        """FileNotFoundError raised for non-existent file."""
        with pytest.raises(FileNotFoundError):
            parse_fasta(tmp_path / "nonexistent.fasta")

    def test_write_fasta_roundtrip(
        self, sample_records: list[dict[str, str]], tmp_path: Path
    ) -> None:
        """Write then read back produces identical records."""
        out_file = tmp_path / "output.fasta"
        write_fasta(sample_records, out_file)

        reloaded = parse_fasta(out_file)

        assert len(reloaded) == len(sample_records)
        for original, loaded in zip(sample_records, reloaded):
            assert loaded["id"] == original["id"]
            assert loaded["sequence"] == original["sequence"]

    def test_write_fasta_line_wrapping(
        self, tmp_path: Path
    ) -> None:
        """Sequences are wrapped at 60 characters per line."""
        long_seq = "A" * 150
        records = [{"id": "long", "sequence": long_seq}]
        out_file = tmp_path / "wrapped.fasta"

        write_fasta(records, out_file, line_width=60)

        content = out_file.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        # Header + 3 sequence lines (60 + 60 + 30)
        assert len(lines) == 4
        assert len(lines[1]) == 60
        assert len(lines[2]) == 60
        assert len(lines[3]) == 30

    def test_write_fasta_empty_records(self) -> None:
        """ValueError raised when writing an empty record list."""
        with pytest.raises(ValueError, match="empty"):
            write_fasta([], Path("dummy.fasta"))


# ======================================================================
# Preprocessor Tests
# ======================================================================

class TestPreprocessor:
    """Tests for the sequence preprocessing module."""

    def test_clean_sequence(self) -> None:
        """Uppercase, strip whitespace, remove dashes and digits."""
        assert clean_sequence("  acD-EF 12gH.I  ") == "ACDEFGHI"
        assert clean_sequence("") == ""
        assert clean_sequence("  ") == ""
        assert clean_sequence("ACDEF") == "ACDEF"

    def test_clean_sequence_preserves_amino_acids(self) -> None:
        """All 20 canonical AAs are preserved after cleaning."""
        canonical = "ACDEFGHIKLMNPQRSTVWY"
        assert clean_sequence(canonical) == canonical

    def test_handle_nonstandard_mask(self) -> None:
        """Mask policy replaces B, J, O, U, X, Z with X."""
        result = handle_nonstandard("ABCJOXUZ", policy="mask")
        assert result == "AXCXXXXX"

    def test_handle_nonstandard_replace(self) -> None:
        """Replace policy maps B→D, Z→E, X→A, J→L, O→K, U→C."""
        result = handle_nonstandard("ABZXJOU", policy="replace")
        assert result == "ADEALKC"  # A stays, B→D, Z→E, X→A, J→L, O→K, U→C

    def test_handle_nonstandard_replace_specific(self) -> None:
        """Each non-standard residue is individually mapped correctly."""
        assert handle_nonstandard("B", policy="replace") == "D"
        assert handle_nonstandard("Z", policy="replace") == "E"
        assert handle_nonstandard("X", policy="replace") == "A"
        assert handle_nonstandard("J", policy="replace") == "L"
        assert handle_nonstandard("O", policy="replace") == "K"
        assert handle_nonstandard("U", policy="replace") == "C"

    def test_handle_nonstandard_invalid_policy(self) -> None:
        """ValueError raised for unknown policy."""
        with pytest.raises(ValueError, match="Invalid nonstandard_policy"):
            handle_nonstandard("ACDEF", policy="unknown")

    def test_handle_nonstandard_empty(self) -> None:
        """Empty input returns empty string for both policies."""
        assert handle_nonstandard("", policy="mask") == ""
        assert handle_nonstandard("", policy="replace") == ""

    def test_chunk_sequence_overlap(
        self, long_sequence: str, long_labels: str
    ) -> None:
        """Overlapping chunks cover the full sequence with correct overlap."""
        chunks = chunk_sequence(
            long_sequence, long_labels, max_len=512, overlap=64
        )

        assert len(chunks) >= 2

        # First chunk is full length
        assert len(chunks[0][0]) == 512
        assert len(chunks[0][1]) == 512

        # Verify overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            seq_a = chunks[i][0]
            seq_b = chunks[i + 1][0]
            # The last `overlap` chars of chunk i should match
            # the first `overlap` chars of chunk i+1
            overlap_a = seq_a[-64:]
            overlap_b = seq_b[:64]
            assert overlap_a == overlap_b, (
                f"Overlap mismatch between chunk {i} and {i+1}"
            )

    def test_chunk_sequence_short(
        self, short_sequence: str, short_labels_q3: str
    ) -> None:
        """Sequence shorter than max_len produces a single chunk."""
        chunks = chunk_sequence(
            short_sequence, short_labels_q3, max_len=512, overlap=64
        )

        assert len(chunks) == 1
        assert chunks[0][0] == short_sequence
        assert chunks[0][1] == short_labels_q3

    def test_chunk_sequence_exact_length(self) -> None:
        """Sequence exactly equal to max_len produces a single chunk."""
        seq = "A" * 512
        labels = "H" * 512
        chunks = chunk_sequence(seq, labels, max_len=512, overlap=64)
        assert len(chunks) == 1

    def test_chunk_sequence_mismatched_lengths(self) -> None:
        """ValueError raised when seq and labels have different lengths."""
        with pytest.raises(ValueError, match="must match"):
            chunk_sequence("ACDEF", "HH", max_len=512, overlap=64)

    def test_chunk_sequence_invalid_params(self) -> None:
        """ValueError for invalid max_len and overlap values."""
        with pytest.raises(ValueError, match="positive"):
            chunk_sequence("ACDEF", "HHECC", max_len=0, overlap=0)
        with pytest.raises(ValueError, match="strictly less"):
            chunk_sequence("ACDEF", "HHECC", max_len=5, overlap=5)

    def test_encode_q3_valid(self) -> None:
        """H, E, C are correctly mapped to 0, 1, 2."""
        assert encode_q3("H") == [0]
        assert encode_q3("E") == [1]
        assert encode_q3("C") == [2]
        assert encode_q3("HEC") == [0, 1, 2]
        assert encode_q3("HHHEEECCC") == [0, 0, 0, 1, 1, 1, 2, 2, 2]

    def test_encode_q3_invalid(self) -> None:
        """ValueError raised for non-Q3 characters."""
        with pytest.raises(ValueError, match="Invalid Q3"):
            encode_q3("HEX")
        with pytest.raises(ValueError, match="Invalid Q3"):
            encode_q3("G")

    def test_encode_q3_empty(self) -> None:
        """Empty string returns empty list."""
        assert encode_q3("") == []

    def test_encode_decode_q3_roundtrip(self) -> None:
        """Encoding then decoding Q3 labels returns the original string."""
        original = "HHEECCHHEC"
        encoded = encode_q3(original)
        decoded = decode_q3(encoded)
        assert decoded == original

    def test_encode_decode_q8_roundtrip(self) -> None:
        """Encoding then decoding Q8 labels returns the original string."""
        original = "HEGIBTSC"
        encoded = encode_q8(original)
        decoded = decode_q8(encoded)
        assert decoded == original

    def test_decode_q3_invalid(self) -> None:
        """ValueError raised for out-of-range Q3 indices."""
        with pytest.raises(ValueError, match="Invalid Q3 index"):
            decode_q3([0, 1, 5])

    def test_decode_q8_invalid(self) -> None:
        """ValueError raised for out-of-range Q8 indices."""
        with pytest.raises(ValueError, match="Invalid Q8 index"):
            decode_q8([0, 1, 10])

    def test_compute_class_weights_shape(self) -> None:
        """Output tensor has shape (num_classes,) and dtype float32."""
        labels = [[0, 0, 1, 2], [1, 1, 2, 0]]
        weights = compute_class_weights(labels, num_classes=3)

        assert weights.shape == (3,)
        assert weights.dtype == torch.float32

    def test_compute_class_weights_balanced(self) -> None:
        """Balanced classes produce approximately equal weights."""
        labels = [[0, 1, 2] * 100]
        weights = compute_class_weights(labels, num_classes=3)

        # All weights should be approximately 1.0
        for w in weights:
            assert abs(w.item() - 1.0) < 0.01

    def test_compute_class_weights_imbalanced(self) -> None:
        """Rare classes get higher weights than common classes."""
        labels = [[0] * 100 + [1] * 10 + [2] * 1]
        weights = compute_class_weights(labels, num_classes=3)

        assert weights[0] < weights[1] < weights[2]

    def test_compute_class_weights_empty(self) -> None:
        """ValueError raised for empty label list."""
        with pytest.raises(ValueError, match="empty"):
            compute_class_weights([], num_classes=3)


# ======================================================================
# Augmentation Tests
# ======================================================================

class TestAugmentation:
    """Tests for the sequence augmentation module."""

    def test_random_mask_preserves_length(
        self, short_sequence: str, short_labels_q3: str
    ) -> None:
        """Output sequence has the same length as the input."""
        masked_seq, masked_labels = random_mask(
            short_sequence, short_labels_q3, mask_prob=0.5
        )
        assert len(masked_seq) == len(short_sequence)
        assert len(masked_labels) == len(short_labels_q3)

    def test_random_mask_labels_unchanged(
        self, short_sequence: str, short_labels_q3: str
    ) -> None:
        """Labels are not modified by random masking."""
        _, labels_out = random_mask(
            short_sequence, short_labels_q3, mask_prob=0.5
        )
        assert labels_out == short_labels_q3

    def test_random_mask_zero_prob(
        self, short_sequence: str, short_labels_q3: str
    ) -> None:
        """Zero mask probability leaves the sequence unchanged."""
        masked_seq, _ = random_mask(
            short_sequence, short_labels_q3, mask_prob=0.0
        )
        assert masked_seq == short_sequence

    def test_random_mask_full_prob(
        self, short_sequence: str, short_labels_q3: str
    ) -> None:
        """Mask probability of 1.0 replaces all residues."""
        masked_seq, _ = random_mask(
            short_sequence, short_labels_q3, mask_prob=1.0
        )
        assert all(c == "X" for c in masked_seq)

    def test_random_mask_invalid_prob(
        self, short_sequence: str, short_labels_q3: str
    ) -> None:
        """ValueError for mask_prob outside [0, 1]."""
        with pytest.raises(ValueError, match="mask_prob"):
            random_mask(short_sequence, short_labels_q3, mask_prob=-0.1)
        with pytest.raises(ValueError, match="mask_prob"):
            random_mask(short_sequence, short_labels_q3, mask_prob=1.5)

    def test_random_mask_mismatched_lengths(self) -> None:
        """ValueError when sequence and labels have different lengths."""
        with pytest.raises(ValueError, match="must match"):
            random_mask("ACDEF", "HH", mask_prob=0.1)

    def test_subsequence_crop_min_frac(
        self, short_sequence: str, short_labels_q3: str
    ) -> None:
        """Cropped sequence is not shorter than min_frac * original length."""
        min_frac = 0.8
        min_expected = int(len(short_sequence) * min_frac)

        # Run multiple times to account for randomness
        for _ in range(50):
            cropped_seq, cropped_labels = subsequence_crop(
                short_sequence, short_labels_q3, min_frac=min_frac
            )
            assert len(cropped_seq) >= min_expected
            assert len(cropped_seq) == len(cropped_labels)

    def test_subsequence_crop_alignment(self) -> None:
        """Cropped labels correspond to the correct residues."""
        seq = "ABCDEFGHIJ"
        labels = "0123456789"

        for _ in range(50):
            cropped_seq, cropped_labels = subsequence_crop(
                seq, labels, min_frac=0.5
            )
            # Find where the crop starts in the original
            start = seq.index(cropped_seq[0])
            expected_labels = labels[start : start + len(cropped_seq)]
            assert cropped_labels == expected_labels

    def test_subsequence_crop_full_frac(
        self, short_sequence: str, short_labels_q3: str
    ) -> None:
        """min_frac=1.0 returns the full sequence."""
        cropped_seq, cropped_labels = subsequence_crop(
            short_sequence, short_labels_q3, min_frac=1.0
        )
        assert cropped_seq == short_sequence
        assert cropped_labels == short_labels_q3

    def test_reverse_complement_protein(self) -> None:
        """Reversal produces reversed sequence and labels."""
        seq, labels = reverse_complement_protein("ACDEF", "HHECC")
        assert seq == "FEDCA"
        assert labels == "CCEHH"

    def test_reverse_complement_empty(self) -> None:
        """Reversal of empty strings returns empty strings."""
        seq, labels = reverse_complement_protein("", "")
        assert seq == ""
        assert labels == ""

    def test_augmentation_pipeline(
        self, short_sequence: str, short_labels_q3: str
    ) -> None:
        """Pipeline applies transforms in order and respects probabilities."""
        # Create a deterministic pipeline with probability 1.0
        call_order: list[str] = []

        def aug_a(seq: str, labels: str) -> tuple[str, str]:
            call_order.append("a")
            return (seq.lower(), labels)

        def aug_b(seq: str, labels: str) -> tuple[str, str]:
            call_order.append("b")
            return (seq + "Z", labels + "C")

        aug_a.__name__ = "aug_a"  # type: ignore[attr-defined]
        aug_b.__name__ = "aug_b"  # type: ignore[attr-defined]

        pipeline = AugmentationPipeline([(aug_a, 1.0), (aug_b, 1.0)])
        result_seq, result_labels = pipeline(short_sequence, short_labels_q3)

        assert call_order == ["a", "b"]
        assert result_seq == short_sequence.lower() + "Z"
        assert result_labels == short_labels_q3 + "C"

    def test_augmentation_pipeline_skip(
        self, short_sequence: str, short_labels_q3: str
    ) -> None:
        """Pipeline with probability 0 skips all augmentations."""
        def aug_never(seq: str, labels: str) -> tuple[str, str]:
            return ("SHOULD_NOT_APPEAR", labels)

        aug_never.__name__ = "aug_never"  # type: ignore[attr-defined]

        pipeline = AugmentationPipeline([(aug_never, 0.0)])
        result_seq, _ = pipeline(short_sequence, short_labels_q3)

        assert result_seq == short_sequence

    def test_augmentation_pipeline_len(self) -> None:
        """Pipeline length matches the number of augmentations."""
        pipeline = AugmentationPipeline([
            (random_mask, 0.5),
            (reverse_complement_protein, 0.3),
        ])
        assert len(pipeline) == 2

    def test_augmentation_pipeline_repr(self) -> None:
        """Pipeline repr includes augmentation names and probabilities."""
        pipeline = AugmentationPipeline([
            (random_mask, 0.5),
        ])
        assert "random_mask" in repr(pipeline)
        assert "0.50" in repr(pipeline)


# ======================================================================
# Dataset Collation Tests
# ======================================================================

class TestCollation:
    """Tests for the collate_fn batching function."""

    def test_collate_fn_padding(self) -> None:
        """Shorter sequences are padded to the max length in the batch."""
        sample_short = _make_sample("ACDE", "HHEC", "HEGT", "prot_1")
        sample_long = _make_sample("ACDEFGHIKL", "HHEECCHHEC", "HHEEGBTSCH", "prot_2")

        batch = collate_fn([sample_short, sample_long])

        # Padded to max length (10 for input_ids, 10 for labels)
        assert batch["input_ids"].shape == (2, 10)
        assert batch["q3_labels"].shape == (2, 10)
        assert batch["q8_labels"].shape == (2, 10)

        # Verify padding values
        # The shorter sample should have pad tokens (1) after position 4
        assert batch["input_ids"][1, 4:].tolist() == [1] * 6

        # Labels should have -100 for padded positions
        assert batch["q3_labels"][1, 4:].tolist() == [-100] * 6
        assert batch["q8_labels"][1, 4:].tolist() == [-100] * 6

    def test_collate_fn_sort_by_length(self) -> None:
        """Batch is sorted by sequence length in descending order."""
        sample_a = _make_sample("ACD", "HHE", "HHE", "short")
        sample_b = _make_sample("ACDEFGHIKL", "HHEECCHHEC", "HHEEGBTSCH", "long")
        sample_c = _make_sample("ACDEF", "HHECC", "HHECC", "medium")

        batch = collate_fn([sample_a, sample_b, sample_c])

        # Should be sorted: long (10) > medium (5) > short (3)
        assert batch["protein_ids"] == ["long", "medium", "short"]
        assert batch["lengths"].tolist() == [10, 5, 3]

    def test_collate_fn_attention_mask(self) -> None:
        """Attention mask is 0 for padded positions."""
        sample_short = _make_sample("ACD", "HHE", "HHE", "p1")
        sample_long = _make_sample("ACDEFG", "HHECCH", "HHECCH", "p2")

        batch = collate_fn([sample_short, sample_long])

        # Longer sample (index 0 after sorting) should have all 1s
        assert batch["attention_mask"][0].tolist() == [1] * 6

        # Shorter sample (index 1 after sorting) should have 1s then 0s
        assert batch["attention_mask"][1, :3].tolist() == [1, 1, 1]
        assert batch["attention_mask"][1, 3:].tolist() == [0, 0, 0]

    def test_collate_fn_preserves_sequences(self) -> None:
        """Raw sequence strings are preserved in the batch output."""
        sample = _make_sample("ACDEF", "HHECC", "HHECC", "test_prot")
        batch = collate_fn([sample])

        assert batch["sequences"] == ["ACDEF"]
        assert batch["protein_ids"] == ["test_prot"]

    def test_collate_fn_empty_batch(self) -> None:
        """ValueError raised for an empty batch."""
        with pytest.raises(ValueError, match="empty"):
            collate_fn([])

    def test_collate_fn_single_sample(self) -> None:
        """Single-sample batch has correct shapes."""
        sample = _make_sample("ACDEF", "HHECC", "HHECC", "solo")
        batch = collate_fn([sample])

        assert batch["input_ids"].shape == (1, 5)
        assert batch["q3_labels"].shape == (1, 5)
        assert batch["lengths"].shape == (1,)
        assert batch["lengths"][0].item() == 5


# ======================================================================
# Edge Case Tests
# ======================================================================

class TestEdgeCases:
    """Tests for boundary conditions and unusual inputs."""

    def test_parse_fasta_string_empty(self) -> None:
        """Empty string returns empty list."""
        assert parse_fasta_string("") == []
        assert parse_fasta_string("   \n  \n  ") == []

    def test_validate_fasta_empty_records(self) -> None:
        """Validation of empty list returns a warning."""
        warnings = validate_fasta([])
        assert len(warnings) == 1
        assert "No records" in warnings[0]

    def test_validate_fasta_nonstandard_chars(self) -> None:
        """Non-standard characters are detected in validation."""
        records = [{"id": "test", "sequence": "ACBZX"}]
        warnings = validate_fasta(records)

        assert len(warnings) >= 1
        # Should mention the non-standard characters
        assert any("non-standard" in w.lower() for w in warnings)

    def test_validate_fasta_short_sequence(self) -> None:
        """Short sequences (<10 residues) produce warnings."""
        records = [{"id": "tiny", "sequence": "ACD"}]
        warnings = validate_fasta(records)

        assert any("short" in w.lower() for w in warnings)

    def test_clean_sequence_only_whitespace(self) -> None:
        """Whitespace-only input returns empty string."""
        assert clean_sequence("   \t\n  ") == ""

    def test_compute_class_weights_single_class(self) -> None:
        """Single-class labels still produce valid weights."""
        labels = [[0, 0, 0, 0]]
        weights = compute_class_weights(labels, num_classes=3)
        assert weights.shape == (3,)
        # The present class should have the lowest weight
        assert weights[0] < weights[1]
