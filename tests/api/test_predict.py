"""API tests for /predict and /predict_batch endpoints.

All tests use the fixture model from conftest.py (tiny nn.Embedding replacing
ESM-2).  See conftest.py for the full rationale.

Test matrix
-----------
Valid paths
  - Single valid sequence → 200, correct output shape and type
  - Q3 and Q8 lengths must match input sequence length
  - Q3 probabilities must sum to 1.0 per residue
  - Q8 probabilities must sum to 1.0 per residue
  - Confidence values in [0, 1]
  - FASTA-formatted input (with header line) is accepted
  - Batch prediction with multiple sequences → 200
  - Batch total_sequences matches input count

Error paths (all must be 422 Unprocessable Entity unless noted)
  - Empty string sequence
  - Sequence shorter than min_length=5
  - Sequence longer than max_length=2048
  - Sequence containing invalid characters (digits, punctuation)
  - Empty payload (missing `sequence` field entirely)
  - Batch with empty list
  - Batch exceeding 50 sequences
  - /predict with no body
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.api.conftest import MEDIUM_SEQ, SHORT_SEQ, VALID_SEQS


# ── valid single prediction ────────────────────────────────────────────────────

class TestPredictValid:
    """Tests for valid single-sequence prediction requests."""

    def test_status_200(self, client: TestClient) -> None:
        """A valid sequence returns HTTP 200."""
        resp = client.post("/predict", json={"sequence": MEDIUM_SEQ})
        assert resp.status_code == 200, resp.text

    def test_response_has_required_fields(self, client: TestClient) -> None:
        """All required response fields are present."""
        resp = client.post("/predict", json={"sequence": MEDIUM_SEQ})
        data = resp.json()
        required = {
            "protein_id", "sequence", "length",
            "q3_prediction", "q8_prediction",
            "q3_probabilities", "q8_probabilities",
            "confidence", "processing_time_ms",
        }
        assert required.issubset(data.keys()), (
            f"Missing fields: {required - data.keys()}"
        )

    def test_q3_length_matches_sequence(self, client: TestClient) -> None:
        """len(q3_prediction) == len(input sequence)."""
        seq = MEDIUM_SEQ
        resp = client.post("/predict", json={"sequence": seq})
        data = resp.json()
        assert len(data["q3_prediction"]) == len(seq)

    def test_q8_length_matches_sequence(self, client: TestClient) -> None:
        """len(q8_prediction) == len(input sequence)."""
        seq = MEDIUM_SEQ
        resp = client.post("/predict", json={"sequence": seq})
        data = resp.json()
        assert len(data["q8_prediction"]) == len(seq)

    def test_sequence_field_is_cleaned_uppercase(self, client: TestClient) -> None:
        """Lowercase input is uppercased in the response."""
        resp = client.post("/predict", json={"sequence": MEDIUM_SEQ.lower()})
        data = resp.json()
        assert data["sequence"] == MEDIUM_SEQ.upper()

    def test_q3_labels_are_valid_chars(self, client: TestClient) -> None:
        """Every residue in q3_prediction is one of H, E, C."""
        resp = client.post("/predict", json={"sequence": MEDIUM_SEQ})
        data = resp.json()
        invalid = [c for c in data["q3_prediction"] if c not in ("H", "E", "C")]
        assert not invalid, f"Invalid Q3 chars: {invalid}"

    def test_q8_labels_are_valid_chars(self, client: TestClient) -> None:
        """Every residue in q8_prediction is one of H, E, G, I, B, T, S, C."""
        valid_q8 = set("HEGIBTSC")
        resp = client.post("/predict", json={"sequence": MEDIUM_SEQ})
        data = resp.json()
        invalid = [c for c in data["q8_prediction"] if c not in valid_q8]
        assert not invalid, f"Invalid Q8 chars: {invalid}"

    def test_q3_probabilities_sum_to_one(self, client: TestClient) -> None:
        """Per-residue Q3 probabilities sum to 1.0 ± 1e-4."""
        resp = client.post("/predict", json={"sequence": MEDIUM_SEQ})
        data = resp.json()
        for i, probs in enumerate(data["q3_probabilities"]):
            s = sum(probs)
            assert abs(s - 1.0) < 1e-4, (
                f"Residue {i}: Q3 probs sum to {s:.6f}, expected ~1.0"
            )

    def test_q8_probabilities_sum_to_one(self, client: TestClient) -> None:
        """Per-residue Q8 probabilities sum to 1.0 ± 1e-4."""
        resp = client.post("/predict", json={"sequence": MEDIUM_SEQ})
        data = resp.json()
        for i, probs in enumerate(data["q8_probabilities"]):
            s = sum(probs)
            assert abs(s - 1.0) < 1e-4, (
                f"Residue {i}: Q8 probs sum to {s:.6f}, expected ~1.0"
            )

    def test_q3_probabilities_shape(self, client: TestClient) -> None:
        """q3_probabilities is a list of 3-element sublists."""
        resp = client.post("/predict", json={"sequence": MEDIUM_SEQ})
        data = resp.json()
        for i, probs in enumerate(data["q3_probabilities"]):
            assert len(probs) == 3, (
                f"Residue {i}: expected 3 Q3 probs, got {len(probs)}"
            )

    def test_q8_probabilities_shape(self, client: TestClient) -> None:
        """q8_probabilities is a list of 8-element sublists."""
        resp = client.post("/predict", json={"sequence": MEDIUM_SEQ})
        data = resp.json()
        for i, probs in enumerate(data["q8_probabilities"]):
            assert len(probs) == 8, (
                f"Residue {i}: expected 8 Q8 probs, got {len(probs)}"
            )

    def test_confidence_values_in_unit_interval(self, client: TestClient) -> None:
        """All confidence scores are in [0, 1]."""
        resp = client.post("/predict", json={"sequence": MEDIUM_SEQ})
        data = resp.json()
        for i, c in enumerate(data["confidence"]):
            assert 0.0 <= c <= 1.0, (
                f"Residue {i}: confidence {c:.4f} out of [0, 1]"
            )

    def test_length_field_matches_sequence(self, client: TestClient) -> None:
        """The `length` field in the response equals the sequence length."""
        seq = MEDIUM_SEQ
        resp = client.post("/predict", json={"sequence": seq})
        data = resp.json()
        assert data["length"] == len(seq)

    def test_processing_time_ms_is_positive(self, client: TestClient) -> None:
        """processing_time_ms is a positive number."""
        resp = client.post("/predict", json={"sequence": MEDIUM_SEQ})
        data = resp.json()
        assert data["processing_time_ms"] > 0

    def test_fasta_input_accepted(self, client: TestClient) -> None:
        """FASTA-formatted input (with >header line) is handled correctly."""
        fasta_input = f">sp|TEST|PROTEIN Human test protein\n{MEDIUM_SEQ}\n"
        resp = client.post("/predict", json={"sequence": fasta_input})
        # Should succeed: the validator strips the header
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert len(data["q3_prediction"]) == len(MEDIUM_SEQ)

    def test_minimum_valid_sequence(self, client: TestClient) -> None:
        """A 5-residue sequence (the minimum) is accepted."""
        resp = client.post("/predict", json={"sequence": "ACDEF"})
        assert resp.status_code == 200, resp.text

    def test_optional_fields_absent_by_default(self, client: TestClient) -> None:
        """attention_map and residue_importance are None when not requested."""
        resp = client.post("/predict", json={"sequence": MEDIUM_SEQ})
        data = resp.json()
        assert data.get("attention_map") is None
        assert data.get("residue_importance") is None

    def test_return_attention_flag(self, client: TestClient) -> None:
        """When return_attention=True, attention_map is included in the response."""
        resp = client.post(
            "/predict",
            json={"sequence": MEDIUM_SEQ, "return_attention": True},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data.get("attention_map") is not None
        # Should be an L×L matrix
        L = len(MEDIUM_SEQ)
        assert len(data["attention_map"]) == L
        assert all(len(row) == L for row in data["attention_map"])


# ── error paths — single prediction ───────────────────────────────────────────

class TestPredictErrors:
    """Tests that verify correct error responses for invalid inputs."""

    def test_empty_string_sequence(self, client: TestClient) -> None:
        """Empty string fails validation (min_length=5)."""
        resp = client.post("/predict", json={"sequence": ""})
        assert resp.status_code == 422, resp.text

    def test_sequence_too_short(self, client: TestClient) -> None:
        """Sequence shorter than 5 residues fails (min_length constraint)."""
        resp = client.post("/predict", json={"sequence": "ACD"})
        assert resp.status_code == 422, resp.text

    def test_sequence_too_long(self, client: TestClient) -> None:
        """Sequence longer than 2048 residues fails (max_length constraint)."""
        resp = client.post("/predict", json={"sequence": "A" * 2049})
        assert resp.status_code == 422, resp.text

    def test_invalid_characters_digits(self, client: TestClient) -> None:
        """Digits in sequence trigger validation error."""
        resp = client.post("/predict", json={"sequence": "ACDEF123GH"})
        assert resp.status_code == 422, resp.text

    def test_invalid_characters_special(self, client: TestClient) -> None:
        """Special characters trigger validation error."""
        resp = client.post("/predict", json={"sequence": "ACDEF!@#GH"})
        assert resp.status_code == 422, resp.text

    def test_invalid_characters_error_message(self, client: TestClient) -> None:
        """The 422 detail mentions invalid amino acid characters."""
        resp = client.post("/predict", json={"sequence": "ACDEF123GH"})
        body = resp.text.lower()
        assert "invalid" in body or "amino acid" in body or "value error" in body

    def test_empty_payload(self, client: TestClient) -> None:
        """Empty JSON payload (missing required `sequence` field) returns 422."""
        resp = client.post("/predict", json={})
        assert resp.status_code == 422, resp.text

    def test_no_body(self, client: TestClient) -> None:
        """Request with no body at all returns 422."""
        resp = client.post("/predict")
        assert resp.status_code == 422, resp.text

    def test_sequence_boundary_exactly_at_max_length(self, client: TestClient) -> None:
        """Sequence of exactly 2048 residues is accepted (boundary condition)."""
        resp = client.post("/predict", json={"sequence": "A" * 2048})
        assert resp.status_code == 200, resp.text

    def test_sequence_boundary_exactly_at_min_length(self, client: TestClient) -> None:
        """Sequence of exactly 5 residues is accepted (boundary condition)."""
        resp = client.post("/predict", json={"sequence": "ACDEF"})
        assert resp.status_code == 200, resp.text

    def test_sequence_boundary_one_below_min(self, client: TestClient) -> None:
        """Sequence of 4 residues (one below min) is rejected."""
        resp = client.post("/predict", json={"sequence": "ACDE"})
        assert resp.status_code == 422, resp.text


# ── batch prediction ───────────────────────────────────────────────────────────

class TestBatchPredict:
    """Tests for the /predict_batch endpoint."""

    def test_batch_status_200(self, client: TestClient) -> None:
        """Valid batch request returns HTTP 200."""
        resp = client.post("/predict_batch", json={"sequences": VALID_SEQS})
        assert resp.status_code == 200, resp.text

    def test_batch_total_sequences_matches_input(self, client: TestClient) -> None:
        """total_sequences in response equals the number of input sequences."""
        seqs = [SHORT_SEQ, MEDIUM_SEQ]
        resp = client.post("/predict_batch", json={"sequences": seqs})
        data = resp.json()
        assert data["total_sequences"] == len(seqs)

    def test_batch_results_count(self, client: TestClient) -> None:
        """len(results) matches the number of input sequences."""
        seqs = [SHORT_SEQ, MEDIUM_SEQ]
        resp = client.post("/predict_batch", json={"sequences": seqs})
        data = resp.json()
        assert len(data["results"]) == len(seqs)

    def test_batch_each_result_has_required_fields(self, client: TestClient) -> None:
        """Each result in the batch response has all required fields."""
        required = {
            "protein_id", "sequence", "length",
            "q3_prediction", "q8_prediction",
            "q3_probabilities", "q8_probabilities",
            "confidence", "processing_time_ms",
        }
        resp = client.post("/predict_batch", json={"sequences": VALID_SEQS})
        for i, result in enumerate(resp.json()["results"]):
            missing = required - result.keys()
            assert not missing, f"Result {i} missing fields: {missing}"

    def test_batch_result_q3_lengths(self, client: TestClient) -> None:
        """Each result's q3_prediction length matches its sequence length."""
        seqs = [SHORT_SEQ, MEDIUM_SEQ]
        resp = client.post("/predict_batch", json={"sequences": seqs})
        for i, (seq, result) in enumerate(zip(seqs, resp.json()["results"])):
            assert len(result["q3_prediction"]) == len(seq), (
                f"Result {i}: q3_prediction length mismatch"
            )

    def test_batch_total_time_is_positive(self, client: TestClient) -> None:
        """total_processing_time_ms is a positive number."""
        resp = client.post("/predict_batch", json={"sequences": VALID_SEQS})
        assert resp.json()["total_processing_time_ms"] > 0

    def test_batch_single_sequence(self, client: TestClient) -> None:
        """Batch with a single sequence works correctly."""
        resp = client.post("/predict_batch", json={"sequences": [MEDIUM_SEQ]})
        assert resp.status_code == 200, resp.text
        assert resp.json()["total_sequences"] == 1

    # ── batch error paths ──────────────────────────────────────────────────────

    def test_batch_empty_sequences_list(self, client: TestClient) -> None:
        """Empty sequences list fails validation (min_length=1 on list)."""
        resp = client.post("/predict_batch", json={"sequences": []})
        assert resp.status_code == 422, resp.text

    def test_batch_oversized(self, client: TestClient) -> None:
        """51 sequences exceeds the 50-item batch limit — must be 422."""
        seqs = ["ACDEFGHIKL"] * 51
        resp = client.post("/predict_batch", json={"sequences": seqs})
        assert resp.status_code == 422, resp.text

    def test_batch_exactly_50_sequences_accepted(self, client: TestClient) -> None:
        """Exactly 50 sequences is at the limit and must be accepted."""
        seqs = ["ACDEFGHIKL"] * 50
        resp = client.post("/predict_batch", json={"sequences": seqs})
        assert resp.status_code == 200, resp.text

    def test_batch_missing_sequences_field(self, client: TestClient) -> None:
        """Missing `sequences` field returns 422."""
        resp = client.post("/predict_batch", json={})
        assert resp.status_code == 422, resp.text


class TestUploadPredict:
    """Tests for the /upload endpoint."""

    def test_upload_fasta_valid(self, client: TestClient) -> None:
        """Uploading a valid FASTA file returns 200 with batch predictions."""
        fasta_content = ">seq1\nACDEFGHIKL\n>seq2\nMNPQRSTVWY\n"
        files = {"file": ("proteins.fasta", fasta_content, "text/plain")}
        resp = client.post("/upload", files=files)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["total_sequences"] == 2
        assert len(data["results"]) == 2
        assert data["results"][0]["protein_id"] == "seq1"
        assert data["results"][0]["sequence"] == "ACDEFGHIKL"

    def test_upload_fasta_invalid_extension(self, client: TestClient) -> None:
        """Uploading a file with an invalid extension returns 400."""
        files = {"file": ("proteins.pdf", "some content", "application/pdf")}
        resp = client.post("/upload", files=files)
        assert resp.status_code == 400, resp.text

    def test_upload_fasta_empty(self, client: TestClient) -> None:
        """Uploading an empty file returns 400."""
        files = {"file": ("proteins.fasta", "", "text/plain")}
        resp = client.post("/upload", files=files)
        assert resp.status_code == 400, resp.text
