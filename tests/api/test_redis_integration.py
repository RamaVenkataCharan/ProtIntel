import sys
from unittest.mock import MagicMock, patch
import os
import time
import pytest

# 1. Detect if the real redis library is installed
try:
    import redis
    REAL_REDIS_AVAILABLE = True
except ImportError:
    REAL_REDIS_AVAILABLE = False
    # Fallback mock for local runs without the redis package installed
    mock_redis_module = MagicMock()
    sys.modules["redis"] = mock_redis_module
    mock_redis_module.Redis = MagicMock
    mock_redis_module.Redis.from_url = lambda url, **kwargs: MagicMock()
    import redis  # type: ignore

from backend.services.job_manager import XAIJobManager
from backend.schemas.response import PredictResponse

# 2. Check if a real Redis server is running and reachable
TEST_REDIS_URL = os.environ.get("TEST_REDIS_URL")
HAS_REAL_REDIS_SERVER = False

if REAL_REDIS_AVAILABLE and TEST_REDIS_URL:
    try:
        client = redis.Redis.from_url(TEST_REDIS_URL)
        client.ping()
        HAS_REAL_REDIS_SERVER = True
    except Exception:
        HAS_REAL_REDIS_SERVER = False


class MockRedis:
    """Mock Redis client for standard multi-worker tests."""

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


@patch("backend.services.job_manager.REDIS_AVAILABLE", True)
class TestRedisIntegration:
    """Verify that multiple workers can share job and cache states via Mocked Redis."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self) -> any:
        """Reset the shared mock Redis store and patch the Redis client constructor."""
        MockRedis._shared_store.clear()
        with patch("redis.Redis", MockRedis):
            yield

    def test_multi_worker_job_state_sharing(self) -> None:
        """Verify that a job created by Worker 1 is visible and modifiable by Worker 2."""
        with patch.dict(os.environ, {"REDIS_HOST": "localhost", "REDIS_PORT": "6379"}):
            worker1_manager = XAIJobManager()
            worker2_manager = XAIJobManager()

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


@pytest.mark.skipif(not HAS_REAL_REDIS_SERVER, reason="Real Redis server not available")
class TestRealRedisIntegration:
    """Verify connection, operations, and key expiration on a live Redis instance."""

    def test_real_redis_operations(self) -> None:
        """Test real round-trips to live Redis container."""
        with patch.dict(os.environ, {"TEST_REDIS_URL": TEST_REDIS_URL, "REDIS_URL": TEST_REDIS_URL}):
            with patch("backend.services.job_manager.REDIS_AVAILABLE", True):
                manager = XAIJobManager()
                assert manager.redis_client is not None

                sequence = "MKFLILLFN"
                job_id = manager.create_job(sequence)
                assert job_id is not None

                # Test basic retrieve
                job = manager.get_job(job_id)
                assert job is not None
                assert job.sequence == sequence
                assert job.status == "pending"

                # Test update status
                manager.update_job(job_id, "processing")
                job = manager.get_job(job_id)
                assert job.status == "processing"


class TestGracefulDegradation:
    """Verify that job_manager handles Redis connection failures gracefully."""

    def test_graceful_degradation_on_redis_failure(self) -> None:
        """Verify fallback to in-memory state without crashing request."""
        # 1. Create a manager and inject a failing mock client
        with patch("backend.services.job_manager.REDIS_AVAILABLE", True):
            manager = XAIJobManager()
            
            mock_failing_redis = MagicMock()
            # Simulate Redis connection/network failures on all operations using built-in ConnectionError
            mock_failing_redis.get.side_effect = ConnectionError("Redis connection lost")
            mock_failing_redis.set.side_effect = ConnectionError("Redis connection lost")
            
            manager.redis_client = mock_failing_redis

            # 2. Create job - should fall back to memory store and not raise exception
            sequence = "MKFLILLFN"
            job_id = manager.create_job(sequence)
            assert job_id is not None
            assert job_id in manager._jobs  # verify saved in local memory fallback dict

            # 3. Get job - should return from memory fallback
            job = manager.get_job(job_id)
            assert job is not None
            assert job.sequence == sequence
            assert job.status == "pending"

            # 4. Update job - should execute in memory fallback
            manager.update_job(job_id, "processing")
            assert manager._jobs[job_id].status == "processing"
