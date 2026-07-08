"""Info router for model metadata, metrics, and health check."""

from __future__ import annotations

from fastapi import APIRouter

from backend.schemas.response import HealthResponse, MetricsResponse, ModelInfoResponse
from backend.routers.predict import get_inference_service

router = APIRouter(tags=["Info"])


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
async def health() -> HealthResponse:
    """Health check endpoint."""
    svc = get_inference_service()
    model_loaded = svc is not None and svc.is_loaded
    device = svc.device if svc is not None else "unknown"

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        device=device,
    )
