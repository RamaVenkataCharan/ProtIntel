import json
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from src.utils.logger import get_logger

logger = get_logger("structured_logger")


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware to log request lifecycles in structured JSON format."""

    async def dispatch(self, request: Request, call_next: any) -> any:
        # Capture start metrics
        start_time = time.time()
        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        path = request.url.path
        user_agent = request.headers.get("user-agent", "unknown")

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            # Handle unhandled exceptions during request processing
            status_code = 500
            latency = (time.time() - start_time) * 1000
            log_record = {
                "event": "request_failed",
                "client_ip": client_ip,
                "method": method,
                "path": path,
                "status_code": status_code,
                "latency_ms": round(latency, 2),
                "user_agent": user_agent,
                "error": str(e),
            }
            logger.error(json.dumps(log_record))
            raise e

        # Calculate latency
        latency = (time.time() - start_time) * 1000
        log_record = {
            "event": "request_completed",
            "client_ip": client_ip,
            "method": method,
            "path": path,
            "status_code": status_code,
            "latency_ms": round(latency, 2),
            "user_agent": user_agent,
        }
        # Record metrics (excluding static files, evaluation assets, health, and prometheus endpoints)
        if not path.startswith(("/static", "/health", "/prometheus", "/evaluation-images")):
            try:
                from backend.routers.info import record_request
                record_request(latency / 1000.0)
            except Exception:
                pass
        if path == "/health":
            logger.debug(json.dumps(log_record))
        else:
            logger.info(json.dumps(log_record))

        return response
