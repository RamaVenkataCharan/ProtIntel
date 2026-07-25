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


class XAIJobManager:
    """Manages asynchronous XAI background tasks and response caching."""

    def __init__(self) -> None:
        self._jobs: Dict[str, JobState] = {}
        self._cache: Dict[str, PredictResponse] = {}
        self._lock = Lock()

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
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                logger.info(f"Cache hit for key: {key}")
                return cached
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
        with self._lock:
            self._cache[key] = result
            logger.info(f"Cached prediction under key: {key}")

    def create_job(self, sequence: str) -> str:
        """Create a new job and return its job ID."""
        job_id = str(uuid.uuid4())
        job = JobState(
            job_id=job_id,
            sequence=sequence,
            status="pending",
            created_at=time.time(),
        )
        with self._lock:
            self._jobs[job_id] = job
        logger.info(f"Created XAI job {job_id}")
        return job_id

    def get_job(self, job_id: str) -> Optional[JobState]:
        """Get the state of a job."""
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
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.status = status
                if result is not None:
                    job.result = result
                if error is not None:
                    job.error = error
                if status in ("completed", "failed"):
                    job.completed_at = time.time()
                logger.info(f"Updated job {job_id} to status: {status}")
            else:
                logger.warning(f"Attempted to update non-existent job {job_id}")

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
