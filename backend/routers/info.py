"""Info router for model metadata, metrics, and health check."""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from backend.schemas.response import HealthResponse, MetricsResponse, ModelInfoResponse
from backend.routers.predict import get_inference_service, job_manager

router = APIRouter(tags=["Info"])

# Prometheus metrics trackers (in-memory)
REQUEST_COUNT = 0
LATENCY_SUM = 0.0
LATENCY_COUNT = 0


def record_request(latency: float) -> None:
    """Record API request count and latency for Prometheus metrics."""
    global REQUEST_COUNT, LATENCY_SUM, LATENCY_COUNT
    REQUEST_COUNT += 1
    LATENCY_SUM += latency
    LATENCY_COUNT += 1


@router.get("/model_info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    """Return model architecture and parameter information."""
    svc = get_inference_service()
    if svc is not None and svc.is_loaded:
        info = svc.get_model_info()
        return ModelInfoResponse(**info)

    return ModelInfoResponse(total_parameters=0, trainable_parameters=0)


@router.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    """Return benchmark metrics from the last evaluation."""
    from pathlib import Path
    from src.utils.io_utils import load_json

    results_path = Path("logs/evaluation/cb513_results.json")
    if results_path.exists():
        data = load_json(results_path)
        return MetricsResponse(
            dataset="CB513",
            q3_accuracy=data.get("q3_accuracy"),
            q8_accuracy=data.get("q8_accuracy"),
            q3_mcc=data.get("q3_mcc"),
        )

    return MetricsResponse()


@router.get("/health", response_model=HealthResponse)
async def health(response: Response) -> HealthResponse:
    """Health check endpoint. Returns 503 if model is not loaded or Redis is down."""
    svc = get_inference_service()
    model_loaded = svc is not None and svc.is_loaded
    device = svc.device if svc is not None else "unknown"

    redis_connected = None
    redis_healthy = True

    if job_manager.redis_client is not None:
        redis_connected = False
        try:
            # Check if Redis is responsive
            if job_manager.redis_client.ping():
                redis_connected = True
        except Exception:
            redis_healthy = False

    is_healthy = model_loaded and redis_healthy

    if not is_healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status="healthy" if is_healthy else "unhealthy",
        model_loaded=model_loaded,
        device=device,
        redis_connected=redis_connected,
    )


@router.get("/prometheus")
async def prometheus_metrics() -> Response:
    """Return system and job queue metrics in Prometheus text exposition format."""
    queue_depth = job_manager.get_queue_depth()
    avg_latency = LATENCY_SUM / LATENCY_COUNT if LATENCY_COUNT > 0 else 0.0

    metrics_text = (
        f"# HELP protintel_http_requests_total Total number of HTTP requests processed.\n"
        f"# TYPE protintel_http_requests_total counter\n"
        f"protintel_http_requests_total {REQUEST_COUNT}\n\n"
        f"# HELP protintel_http_request_duration_seconds_sum Sum of request durations in seconds.\n"
        f"# TYPE protintel_http_request_duration_seconds_sum counter\n"
        f"protintel_http_request_duration_seconds_sum {LATENCY_SUM:.4f}\n\n"
        f"# HELP protintel_http_request_duration_seconds_avg Average request duration in seconds.\n"
        f"# TYPE protintel_http_request_duration_seconds_avg gauge\n"
        f"protintel_http_request_duration_seconds_avg {avg_latency:.4f}\n\n"
        f"# HELP protintel_xai_job_queue_depth Number of active async XAI jobs (pending or processing).\n"
        f"# TYPE protintel_xai_job_queue_depth gauge\n"
        f"protintel_xai_job_queue_depth {queue_depth}\n"
    )

    return Response(content=metrics_text, media_type="text/plain; version=0.0.4")
