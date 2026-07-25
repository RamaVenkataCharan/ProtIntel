import sys
from unittest.mock import MagicMock

# Mock the redis module in sys.modules to prevent ModuleNotFoundError
mock_redis_module = MagicMock()
sys.modules["redis"] = mock_redis_module

import os
import time
from unittest.mock import patch
import pytest
from backend.services.job_manager import XAIJobManager
from backend.schemas.response import PredictResponse


class MockRedis:
    """Mock Redis client that stores data in a shared class-level dictionary."""

    _shared_store: dict[str, str] = {}

    def __init__(self, **kwargs: any) -> None:
        pass

    def ping(self) -> bool:
        return True

    def get(self, key: str) -> str | None:
        return self._shared_store.get(key)

    def set(self, key: str, value: str, ex: int | None = None) -> bool:
        self._shared_store[key] = value
        return True


# Hook the mock module's Redis class to our MockRedis implementation
mock_redis_module.Redis = MockRedis
mock_redis_module.Redis.from_url = lambda url, **kwargs: MockRedis()



@patch("backend.services.job_manager.REDIS_AVAILABLE", True)
@patch("redis.Redis", MockRedis)
class TestRedisIntegration:
    """Verify that multiple workers can share job and cache states via Redis."""

    @pytest.fixture(autouse=True)
    def clean_mock_store(self) -> None:
        """Reset the shared mock Redis store before each test."""
        MockRedis._shared_store.clear()

    def test_multi_worker_job_state_sharing(self) -> None:
        """Verify that a job created by Worker 1 is visible and modifiable by Worker 2."""
        # Configure environment variables to trigger Redis initialization
        with patch.dict(os.environ, {"REDIS_HOST": "localhost", "REDIS_PORT": "6379"}):
            worker1_manager = XAIJobManager()
            worker2_manager = XAIJobManager()

            # Ensure both workers successfully connected to our MockRedis
            assert worker1_manager.redis_client is not None
            assert worker2_manager.redis_client is not None

            # 1. Worker 1 creates a job
            sequence = "M" * 20
            job_id = worker1_manager.create_job(sequence)
            assert job_id is not None

            # 2. Worker 2 retrieves the job and checks its pending status
            job_state_w2 = worker2_manager.get_job(job_id)
            assert job_state_w2 is not None
            assert job_state_w2.job_id == job_id
            assert job_state_w2.status == "pending"
            assert job_state_w2.sequence == sequence

            # 3. Worker 2 updates the job to "processing"
            worker2_manager.update_job(job_id, "processing")

            # 4. Worker 1 reads the updated state from Redis
            job_state_w1 = worker1_manager.get_job(job_id)
            assert job_state_w1 is not None
            assert job_state_w1.status == "processing"

    def test_multi_worker_cache_sharing(self) -> None:
        """Verify that cache hits are shared across different workers."""
        with patch.dict(os.environ, {"REDIS_HOST": "localhost"}):
            worker1_manager = XAIJobManager()
            worker2_manager = XAIJobManager()

            # Mock prediction result
            mock_prediction = PredictResponse(
                protein_id="test-id-123",
                sequence="MKFLILLFN",
                length=9,
                q3_prediction=["C"] * 9,
                q8_prediction=["C"] * 9,
                q3_probabilities=[[0.1, 0.2, 0.7]] * 9,
                q8_probabilities=[[0.1] * 8] * 9,
                confidence=[0.95] * 9,
                processing_time_ms=10.5,
            )

            # 1. Worker 1 caches the prediction
            worker1_manager.cache_prediction(
                sequence="MKFLILLFN",
                return_attention=False,
                return_xai=True,
                xai_method="rollout",
                result=mock_prediction,
            )

            # 2. Worker 2 gets a cache hit for the same request options
            cached_result = worker2_manager.get_cached_prediction(
                sequence="MKFLILLFN",
                return_attention=False,
                return_xai=True,
                xai_method="rollout",
            )

            assert cached_result is not None
            assert cached_result.protein_id == "test-id-123"
            assert cached_result.sequence == "MKFLILLFN"
            assert cached_result.confidence == [0.95] * 9
