"""API tests for /model_info, /health, and /metrics endpoints.

All tests use the fixture model from conftest.py (tiny nn.Embedding replacing
ESM-2).  See conftest.py for the full rationale.

Test matrix
-----------
/health
  - Returns HTTP 200
  - model_loaded is True when service is wired
  - status is "healthy" when model is loaded
  - device field is present and a non-empty string

/model_info
  - Returns HTTP 200
  - All required fields present
  - total_parameters > 0 (real model layers)
  - trainable_parameters <= total_parameters
  - q3_classes has exactly 3 entries
  - q8_classes has exactly 8 entries

/metrics
  - Returns HTTP 200 even when no evaluation file exists
  - All metric fields default to null when no evaluation has been run
  - dataset field defaults to "CB513"

/ (root)
  - Returns HTTP 200
  - name and version fields present
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealth:
    """Tests for the GET /health endpoint."""

    def test_health_status_200(self, client: TestClient) -> None:
        """Health endpoint returns HTTP 200."""
        resp = client.get("/health")
        assert resp.status_code == 200, resp.text

    def test_health_model_loaded_true(self, client: TestClient) -> None:
        """model_loaded is True because the fixture service is pre-loaded."""
        resp = client.get("/health")
        assert resp.json()["model_loaded"] is True

    def test_health_status_healthy(self, client: TestClient) -> None:
        """status field is 'healthy' when model is loaded."""
        resp = client.get("/health")
        assert resp.json()["status"] == "healthy"

    def test_health_device_field_present(self, client: TestClient) -> None:
        """device field is present and non-empty."""
        resp = client.get("/health")
        device = resp.json().get("device", "")
        assert isinstance(device, str) and device != ""

    def test_health_required_fields(self, client: TestClient) -> None:
        """All required fields are present in the response."""
        required = {"status", "model_loaded", "device"}
        data = client.get("/health").json()
        assert required.issubset(data.keys()), (
            f"Missing fields: {required - data.keys()}"
        )


# ── /model_info ───────────────────────────────────────────────────────────────

class TestModelInfo:
    """Tests for the GET /model_info endpoint."""

    def test_model_info_status_200(self, client: TestClient) -> None:
        """model_info endpoint returns HTTP 200."""
        resp = client.get("/model_info")
        assert resp.status_code == 200, resp.text

    def test_model_info_required_fields(self, client: TestClient) -> None:
        """All required fields are present in the response."""
        required = {
            "model_name", "version", "architecture",
            "esm2_model", "total_parameters", "trainable_parameters",
            "q3_classes", "q8_classes",
        }
        data = client.get("/model_info").json()
        assert required.issubset(data.keys()), (
            f"Missing fields: {required - data.keys()}"
        )

    def test_model_info_total_parameters_positive(self, client: TestClient) -> None:
        """total_parameters > 0 — the fixture model has real weight tensors."""
        data = client.get("/model_info").json()
        assert data["total_parameters"] > 0

    def test_model_info_trainable_lte_total(self, client: TestClient) -> None:
        """trainable_parameters <= total_parameters."""
        data = client.get("/model_info").json()
        assert data["trainable_parameters"] <= data["total_parameters"]

    def test_model_info_q3_classes_count(self, client: TestClient) -> None:
        """q3_classes contains exactly 3 entries (H, E, C)."""
        data = client.get("/model_info").json()
        assert len(data["q3_classes"]) == 3, (
            f"Expected 3 Q3 classes, got {len(data['q3_classes'])}: "
            f"{data['q3_classes']}"
        )

    def test_model_info_q8_classes_count(self, client: TestClient) -> None:
        """q8_classes contains exactly 8 entries."""
        data = client.get("/model_info").json()
        assert len(data["q8_classes"]) == 8, (
            f"Expected 8 Q8 classes, got {len(data['q8_classes'])}: "
            f"{data['q8_classes']}"
        )

    def test_model_info_version_string(self, client: TestClient) -> None:
        """version is a non-empty string."""
        data = client.get("/model_info").json()
        assert isinstance(data["version"], str) and data["version"] != ""

    def test_model_info_model_name(self, client: TestClient) -> None:
        """model_name is 'ProtIntel'."""
        data = client.get("/model_info").json()
        assert data["model_name"] == "ProtIntel"


# ── /metrics ──────────────────────────────────────────────────────────────────

class TestMetrics:
    """Tests for the GET /metrics endpoint.

    These tests verify the *no-evaluation* state: no trained checkpoint exists
    yet, so the metrics file ``logs/evaluation/cb513_results.json`` is absent.
    The endpoint must return HTTP 200 with null metric fields in this state.
    """

    def test_metrics_status_200(self, client: TestClient) -> None:
        """metrics endpoint returns HTTP 200 even with no evaluation file."""
        resp = client.get("/metrics")
        assert resp.status_code == 200, resp.text

    def test_metrics_dataset_field(self, client: TestClient) -> None:
        """dataset field defaults to 'CB513'."""
        data = client.get("/metrics").json()
        assert data.get("dataset") == "CB513"

    def test_metrics_null_when_no_evaluation(self, client: TestClient) -> None:
        """q3_accuracy, q8_accuracy, and q3_mcc are null when no eval file exists.

        This is the expected state before training + evaluate.py have been run.
        """
        import os
        results_path = "logs/evaluation/cb513_results.json"
        if os.path.exists(results_path):
            pytest.skip("Evaluation file exists — null-field test not applicable")

        data = client.get("/metrics").json()
        assert data.get("q3_accuracy") is None, (
            f"Expected q3_accuracy=null, got {data.get('q3_accuracy')}"
        )
        assert data.get("q8_accuracy") is None, (
            f"Expected q8_accuracy=null, got {data.get('q8_accuracy')}"
        )
        assert data.get("q3_mcc") is None, (
            f"Expected q3_mcc=null, got {data.get('q3_mcc')}"
        )

    def test_metrics_required_fields(self, client: TestClient) -> None:
        """All expected fields are present in the metrics response."""
        required = {"dataset", "q3_accuracy", "q8_accuracy", "q3_mcc"}
        data = client.get("/metrics").json()
        assert required.issubset(data.keys()), (
            f"Missing fields: {required - data.keys()}"
        )


# ── / (root) ──────────────────────────────────────────────────────────────────

class TestRoot:
    """Tests for the GET / root endpoint."""

    def test_root_status_200(self, client: TestClient) -> None:
        """Root endpoint returns HTTP 200."""
        resp = client.get("/")
        assert resp.status_code == 200, resp.text

    def test_root_has_name_and_version(self, client: TestClient) -> None:
        """Root response contains name and version fields."""
        data = client.get("/").json()
        assert "name" in data
        assert "version" in data

    def test_root_name_value(self, client: TestClient) -> None:
        """name field identifies the API."""
        data = client.get("/").json()
        assert "ProtIntel" in data["name"]
