"""Prediction router for /predict and /predict_batch endpoints."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends

from backend.schemas.request import BatchPredictRequest, PredictRequest
from backend.schemas.response import BatchPredictResponse, PredictResponse, JobStatusResponse
from backend.services.job_manager import XAIJobManager
from backend.middleware.rate_limit import rate_limit

router = APIRouter(tags=["Prediction"])

# Thread pool for CPU-intensive inference
_executor = ThreadPoolExecutor(max_workers=2)

# Global inference service reference (set by main.py)
_inference_service = None

# Job manager for async execution and caching
job_manager = XAIJobManager()


def set_inference_service(service: Any) -> None:
    """Set the global inference service reference."""
    global _inference_service
    _inference_service = service


def get_inference_service() -> Any:
    """Return the current inference service (may be None if not yet loaded)."""
    return _inference_service


@router.post(
    "/predict",
    response_model=PredictResponse | JobStatusResponse,
    dependencies=[Depends(rate_limit(limit=10, period=60))],
)
async def predict(
    request: PredictRequest,
    background_tasks: BackgroundTasks,
) -> PredictResponse | JobStatusResponse:
    """Predict secondary structure for a single protein sequence.

    Runs synchronously for fast operations (no XAI or attention rollout only),
    and asynchronously via BackgroundTasks for heavy XAI computation (IG, SHAP).

    Args:
        request: PredictRequest with sequence and options.
        background_tasks: FastAPI BackgroundTasks manager.

    Returns:
        PredictResponse if completed synchronously, or JobStatusResponse if queued.
    """
    if _inference_service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    sequence = request.sequence

    # Check cache first
    cached = job_manager.get_cached_prediction(
        sequence=sequence,
        return_attention=request.return_attention,
        return_xai=request.return_xai,
        xai_method=request.xai_method,
    )
    if cached is not None:
        return cached

    # IG and SHAP attributions take ~13 seconds on CPU, so they run asynchronously
    is_slow_xai = request.return_xai and request.xai_method in ("ig", "shap")

    if is_slow_xai:
        job_id = job_manager.create_job(sequence)
        background_tasks.add_task(
            job_manager.execute_job_async,
            job_id=job_id,
            inference_service=_inference_service,
            sequence=sequence,
            return_attention=request.return_attention,
            return_xai=request.return_xai,
            xai_method=request.xai_method,
        )
        return JobStatusResponse(
            job_id=job_id,
            status="pending",
        )

    # Fast prediction runs synchronously
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        _executor,
        lambda: _inference_service.predict(
            sequence=sequence,
            return_attention=request.return_attention,
            return_xai=request.return_xai,
            xai_method=request.xai_method,
        ),
    )
    response = PredictResponse(**result)

    # Cache sync result
    job_manager.cache_prediction(
        sequence=sequence,
        return_attention=request.return_attention,
        return_xai=request.return_xai,
        xai_method=request.xai_method,
        result=response,
    )

    return response


@router.get(
    "/predict/jobs/{job_id}",
    response_model=JobStatusResponse,
    dependencies=[Depends(rate_limit(limit=120, period=60))],
)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """Poll the status of an asynchronous prediction/XAI job."""
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        result=job.result,
        error=job.error,
    )



@router.post("/predict_batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """Predict secondary structure for a batch of sequences.

    Args:
        request: BatchPredictRequest with sequences.

    Returns:
        BatchPredictResponse with results for all sequences.
    """
    if _inference_service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    import time
    start = time.time()

    results = []
    for seq in request.sequences:
        loop = asyncio.get_running_loop()
        def run_single(s: str = seq) -> dict[str, Any]:
            return _inference_service.predict(
                sequence=s,
                return_attention=request.return_attention,
                return_xai=request.return_xai,
                xai_method=request.xai_method,
            )

        result = await loop.run_in_executor(
            _executor,
            run_single,
        )
        results.append(PredictResponse(**result))

    total_ms = (time.time() - start) * 1000
    return BatchPredictResponse(
        results=results,
        total_sequences=len(results),
        total_processing_time_ms=round(total_ms, 2),
    )
