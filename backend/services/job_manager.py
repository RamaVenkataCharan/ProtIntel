"""Job manager for handling long-running XAI tasks and response caching."""

from __future__ import annotations

import uuid
import asyncio
import time
from typing import Any, Dict, Optional, Literal
from threading import Lock
from pydantic import BaseModel
from backend.schemas.response import PredictResponse
from src.utils.io_utils import compute_sequence_hash
from src.utils.logger import get_logger

logger = get_logger(__name__)

class JobState(BaseModel):
    job_id: str
    sequence: str
    status: Literal["pending", "processing", "completed", "failed"]
    result: Optional[PredictResponse] = None
    error: Optional[str] = None
    created_at: float
    completed_at: Optional[float] = None


import os
import json

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class XAIJobManager:
    """Manages asynchronous XAI background tasks and response caching."""

    def __init__(self) -> None:
        self._jobs: Dict[str, JobState] = {}
        self._cache: Dict[str, PredictResponse] = {}
        self._lock = Lock()

        self.redis_client: Optional[redis.Redis] = None
        redis_host = os.environ.get("REDIS_HOST")
        redis_url = os.environ.get("REDIS_URL")

        if (redis_host or redis_url) and REDIS_AVAILABLE:
            try:
                import redis as redis_lib
                if redis_url:
                    self.redis_client = redis_lib.Redis.from_url(redis_url)
                else:
                    port = int(os.environ.get("REDIS_PORT", 6379))
                    db = int(os.environ.get("REDIS_DB", 0))
                    self.redis_client = redis_lib.Redis(
                        host=redis_host, port=port, db=db, decode_responses=True
                    )
                self.redis_client.ping()
                logger.info("Successfully connected to Redis. Multi-worker safe state active.")
            except Exception as e:
                logger.warning(
                    f"Failed to connect to Redis ({e}). Falling back to single-worker in-memory state."
                )
                self.redis_client = None
        elif (redis_host or redis_url) and not REDIS_AVAILABLE:
            logger.warning(
                "Redis host configured but 'redis' package is not installed. "
                "Falling back to single-worker in-memory state."
            )

    def _get_cache_key(
        self,
        sequence: str,
        return_attention: bool,
        return_xai: bool,
        xai_method: str,
    ) -> str:
        """Generate a unique cache key based on sequence hash and request options."""
        seq_hash = compute_sequence_hash(sequence)
        return f"{seq_hash}_{return_attention}_{return_xai}_{xai_method}"

    def get_cached_prediction(
        self,
        sequence: str,
        return_attention: bool,
        return_xai: bool,
        xai_method: str,
    ) -> Optional[PredictResponse]:
        """Retrieve a cached prediction if available."""
        key = self._get_cache_key(sequence, return_attention, return_xai, xai_method)

        if self.redis_client is not None:
            try:
                cached_json = self.redis_client.get(f"cache:{key}")
                if cached_json:
                    data = json.loads(cached_json)
                    logger.info(
                        json.dumps({"event": "cache_hit", "key": key, "store": "redis"})
                    )
                    return PredictResponse(**data)
            except Exception as e:
                logger.error(
                    json.dumps(
                        {"event": "cache_read_error", "key": key, "error": str(e)}
                    )
                )

        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                logger.info(
                    json.dumps({"event": "cache_hit", "key": key, "store": "memory"})
                )
                return cached

        logger.info(json.dumps({"event": "cache_miss", "key": key}))
        return None

    def cache_prediction(
        self,
        sequence: str,
        return_attention: bool,
        return_xai: bool,
        xai_method: str,
        result: PredictResponse,
    ) -> None:
        """Store a completed prediction in the cache."""
        key = self._get_cache_key(sequence, return_attention, return_xai, xai_method)

        if self.redis_client is not None:
            try:
                # 24 hours TTL (86400 seconds)
                self.redis_client.set(f"cache:{key}", json.dumps(result.model_dump()), ex=86400)
                logger.info(
                    json.dumps({"event": "cache_save", "key": key, "store": "redis"})
                )
                return
            except Exception as e:
                logger.error(
                    json.dumps(
                        {"event": "cache_write_error", "key": key, "error": str(e)}
                    )
                )

        with self._lock:
            self._cache[key] = result
            logger.info(
                json.dumps({"event": "cache_save", "key": key, "store": "memory"})
            )

    def create_job(self, sequence: str) -> str:
        """Create a new job and return its job ID."""
        job_id = str(uuid.uuid4())
        job = JobState(
            job_id=job_id,
            sequence=sequence,
            status="pending",
            created_at=time.time(),
        )

        if self.redis_client is not None:
            try:
                # 30 minutes TTL
                self.redis_client.set(f"job:{job_id}", json.dumps(job.model_dump()), ex=1800)
                logger.info(
                    json.dumps(
                        {
                            "event": "job_created",
                            "job_id": job_id,
                            "store": "redis",
                            "sequence_length": len(sequence),
                        }
                    )
                )
                return job_id
            except Exception as e:
                logger.error(
                    json.dumps(
                        {"event": "job_create_error", "job_id": job_id, "error": str(e)}
                    )
                )

        with self._lock:
            self._jobs[job_id] = job
        logger.info(
            json.dumps(
                {
                    "event": "job_created",
                    "job_id": job_id,
                    "store": "memory",
                    "sequence_length": len(sequence),
                }
            )
        )
        return job_id

    def get_job(self, job_id: str) -> Optional[JobState]:
        """Get the state of a job."""
        if self.redis_client is not None:
            try:
                job_json = self.redis_client.get(f"job:{job_id}")
                if job_json:
                    data = json.loads(job_json)
                    return JobState(**data)
            except Exception as e:
                logger.error(f"Error reading job from Redis: {e}")

        with self._lock:
            return self._jobs.get(job_id)

    def update_job(
        self,
        job_id: str,
        status: Literal["pending", "processing", "completed", "failed"],
        result: Optional[PredictResponse] = None,
        error: Optional[str] = None,
    ) -> None:
        """Update a job's status and results."""
        job = self.get_job(job_id)
        if job is None:
            logger.warning(f"Attempted to update non-existent job {job_id}")
            return

        job.status = status
        if result is not None:
            job.result = result
        if error is not None:
            job.error = error
        if status in ("completed", "failed"):
            job.completed_at = time.time()

        if self.redis_client is not None:
            try:
                # Keep 30 min TTL on update
                self.redis_client.set(f"job:{job_id}", json.dumps(job.model_dump()), ex=1800)
                logger.info(
                    json.dumps(
                        {
                            "event": "job_updated",
                            "job_id": job_id,
                            "store": "redis",
                            "status": status,
                        }
                    )
                )
                return
            except Exception as e:
                logger.error(
                    json.dumps(
                        {"event": "job_update_error", "job_id": job_id, "error": str(e)}
                    )
                )

        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id] = job
                logger.info(
                    json.dumps(
                        {
                            "event": "job_updated",
                            "job_id": job_id,
                            "store": "memory",
                            "status": status,
                        }
                    )
                )

    async def execute_job_async(
        self,
        job_id: str,
        inference_service: Any,
        sequence: str,
        return_attention: bool,
        return_xai: bool,
        xai_method: str,
    ) -> None:
        """Background worker function that runs the prediction and updates the job."""
        self.update_job(job_id, "processing")
        try:
            # Run the CPU-intensive prediction in a separate thread to avoid blocking the event loop
            loop = asyncio.get_running_loop()
            result_dict = await loop.run_in_executor(
                None,  # Uses default thread pool executor
                lambda: inference_service.predict(
                    sequence=sequence,
                    return_attention=return_attention,
                    return_xai=return_xai,
                    xai_method=xai_method,
                )
            )
            response = PredictResponse(**result_dict)
            
            # Cache the result
            self.cache_prediction(sequence, return_attention, return_xai, xai_method, response)
            
            # Update the job state
            self.update_job(job_id, "completed", result=response)
        except Exception as e:
            logger.error(f"Error executing job {job_id}: {e}", exc_info=True)
            self.update_job(job_id, "failed", error=str(e))

    def get_queue_depth(self) -> int:
        """Get the number of jobs currently pending or processing."""
        if self.redis_client is not None:
            try:
                # Retrieve all job keys
                keys = self.redis_client.keys("job:*")
                count = 0
                for k in keys:
                    job_json = self.redis_client.get(k)
                    if job_json:
                        data = json.loads(job_json)
                        if data.get("status") in ("pending", "processing"):
                            count += 1
                return count
            except Exception as e:
                logger.error(f"Error checking queue depth in Redis: {e}")

        with self._lock:
            return sum(1 for j in self._jobs.values() if j.status in ("pending", "processing"))
