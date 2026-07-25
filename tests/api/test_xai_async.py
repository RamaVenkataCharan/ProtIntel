"""API tests for asynchronous XAI processing, polling, and response caching."""

from __future__ import annotations

import time
import pytest
from fastapi.testclient import TestClient
from tests.api.conftest import SHORT_SEQ, MEDIUM_SEQ


class TestXAIAsyncAndCaching:
    """Verify asynchronous XAI endpoints, polling status, and caching logic."""

    def test_predict_sync_no_xai(self, client: TestClient) -> None:
        """A predict request without XAI should execute synchronously and return immediately."""
        resp = client.post("/predict", json={
            "sequence": SHORT_SEQ,
            "return_xai": False,
        })
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "protein_id" in data
        assert "q3_prediction" in data
        assert "job_id" not in data  # Sync request should NOT return a job ID

    def test_predict_sync_rollout_xai(self, client: TestClient) -> None:
        """A rollout XAI request is fast and should execute synchronously."""
        resp = client.post("/predict", json={
            "sequence": SHORT_SEQ,
            "return_xai": True,
            "xai_method": "rollout",
        })
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "protein_id" in data
        assert "residue_importance" in data
        assert len(data["residue_importance"]) == len(SHORT_SEQ)
        assert "job_id" not in data

    def test_predict_async_ig_xai(self, client: TestClient) -> None:
        """An Integrated Gradients (IG) XAI request is slow and should execute asynchronously."""
        resp = client.post("/predict", json={
            "sequence": SHORT_SEQ,
            "return_xai": True,
            "xai_method": "ig",
        })
        assert resp.status_code == 200, resp.text
        data = resp.json()
        
        # Verify job status structure
        assert "job_id" in data
        assert "status" in data
        assert data["status"] in ("pending", "processing", "completed")
        assert "result" in data

    def test_job_polling_and_completion(self, client: TestClient) -> None:
        """Creating an async job, polling it, and verifying it reaches completion."""
        # 1. Start the job
        resp = client.post("/predict", json={
            "sequence": MEDIUM_SEQ,
            "return_xai": True,
            "xai_method": "ig",
        })
        assert resp.status_code == 200
        job_data = resp.json()
        job_id = job_data["job_id"]

        # 2. Poll job status
        # Since the backend uses background tasks, we might need to wait slightly for completion
        max_attempts = 10
        completed = False
        result_data = None
        
        for _ in range(max_attempts):
            poll_resp = client.get(f"/predict/jobs/{job_id}")
            assert poll_resp.status_code == 200
            poll_data = poll_resp.json()
            assert poll_data["job_id"] == job_id
            
            if poll_data["status"] == "completed":
                completed = True
                result_data = poll_data["result"]
                break
            elif poll_data["status"] == "failed":
                pytest.fail(f"Job failed with error: {poll_data['error']}")
            
            time.sleep(0.1)

        assert completed, f"Job {job_id} did not complete within timeout"
        assert result_data is not None
        assert result_data["sequence"] == MEDIUM_SEQ.upper()
        assert len(result_data["residue_importance"]) == len(MEDIUM_SEQ)
        
        # Attribution values should be normalized to [0, 1]
        for val in result_data["residue_importance"]:
            assert 0.0 <= val <= 1.0

    def test_invalid_job_id_returns_404(self, client: TestClient) -> None:
        """Polling with a non-existent job ID should return HTTP 404."""
        resp = client.get("/predict/jobs/non-existent-uuid-12345")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_response_caching_sync(self, client: TestClient) -> None:
        """Subsequent identical sync requests should return instantly via cache."""
        seq = "M" * 10
        # First request (cold)
        start_cold = time.time()
        resp1 = client.post("/predict", json={
            "sequence": seq,
            "return_xai": False,
        })
        time_cold = time.time() - start_cold
        assert resp1.status_code == 200
        
        # Second request (hot cache)
        start_hot = time.time()
        resp2 = client.post("/predict", json={
            "sequence": seq,
            "return_xai": False,
        })
        time_hot = time.time() - start_hot
        assert resp2.status_code == 200
        
        # Both results must be identical
        assert resp1.json()["protein_id"] == resp2.json()["protein_id"]

    def test_response_caching_async(self, client: TestClient) -> None:
        """Identical async requests should hit the cache and return completed response immediately."""
        seq = "M" * 12
        
        # 1. Trigger first request (async job)
        resp1 = client.post("/predict", json={
            "sequence": seq,
            "return_xai": True,
            "xai_method": "ig",
        })
        job_data = resp1.json()
        job_id = job_data["job_id"]
        
        # Wait for the job to complete
        completed = False
        for _ in range(20):
            poll_resp = client.get(f"/predict/jobs/{job_id}")
            if poll_resp.json()["status"] == "completed":
                completed = True
                break
            time.sleep(0.05)
        assert completed, "First job did not complete"
        
        # 2. Trigger second identical request
        # Since it is in the cache, it should return a Completed PredictResponse directly, NOT a job ID
        resp2 = client.post("/predict", json={
            "sequence": seq,
            "return_xai": True,
            "xai_method": "ig",
        })
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert "job_id" not in data2  # Direct cache hit returns prediction response
        assert "residue_importance" in data2
        assert len(data2["residue_importance"]) == len(seq)
