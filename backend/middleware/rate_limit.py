import os
import sys
import time
from fastapi import Request, HTTPException, status
from backend.services.job_manager import XAIJobManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Thread-safe in-memory sliding logs for rate-limiting fallback: key -> list of timestamps
_in_memory_limits: dict[str, list[float]] = {}


def rate_limit(limit: int, period: int):
    """FastAPI dependency for rate limiting endpoints.

    Args:
        limit: Number of requests allowed in the period.
        period: Period in seconds.
    """
    async def dependency(request: Request) -> None:
        # Automatically bypass rate limiting when running tests
        if "pytest" in sys.modules or os.environ.get("TESTING") == "true":
            return

        # Resolve client IP address
        client_ip = request.client.host if request.client else "unknown"
        path = request.url.path
        key = f"rate:{path}:{client_ip}"

        # Import job_manager reference locally to prevent circular imports
        from backend.routers.predict import job_manager

        # 1. Redis rate-limiting (using sliding window sorted set)
        if job_manager.redis_client is not None:
            try:
                now = time.time()
                pipeline = job_manager.redis_client.pipeline()
                # Clear values older than window period
                pipeline.zremrangebyscore(key, 0, now - period)
                # Add current request timestamp
                pipeline.zadd(key, {str(now): now})
                # Count requests in window
                pipeline.zcard(key)
                # Set TTL on rate-limit log to avoid persistent bloat
                pipeline.expire(key, period)
                # Execute pipeline transactionally
                results = pipeline.execute()
                current_count = results[2]

                if current_count > limit:
                    logger.warning(
                        f"Rate limit exceeded for client {client_ip} on {path} (Redis). "
                        f"Count: {current_count}/{limit}"
                    )
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="Rate limit exceeded. Please try again later.",
                    )
                return
            except HTTPException:
                raise
            except Exception as e:
                # Log error and fall back gracefully to in-memory tracking
                logger.warning(f"Redis rate limiting failed: {e}. Falling back to memory.")

        # 2. In-memory fallback rate-limiting
        now = time.time()
        with job_manager._lock:
            if key not in _in_memory_limits:
                _in_memory_limits[key] = []

            # Filter out expired requests
            _in_memory_limits[key] = [t for t in _in_memory_limits[key] if t > now - period]

            current_count = len(_in_memory_limits[key])
            if current_count >= limit:
                logger.warning(
                    f"Rate limit exceeded for client {client_ip} on {path} (Memory). "
                    f"Count: {current_count + 1}/{limit}"
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded. Please try again later.",
                )

            _in_memory_limits[key].append(now)

    return dependency
