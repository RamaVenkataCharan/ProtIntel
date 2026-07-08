"""Prediction router for /predict and /predict_batch endpoints."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, HTTPException

from backend.schemas.request import BatchPredictRequest, PredictRequest
from backend.schemas.response import BatchPredictResponse, PredictResponse

router = APIRouter(tags=["Prediction"])

# Thread pool for CPU-intensive inference
_executor = ThreadPoolExecutor(max_workers=2)

# Global inference service reference (set by main.py)
_inference_service = None


def set_inference_service(service: Any) -> None:
    """Set the global inference service reference."""
    global _inference_service
    _inference_service = service


def get_inference_service() -> Any:
    """Return the current inference service (may be None if not yet loaded)."""
    return _inference_service


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Predict secondary structure for a single protein sequence.

    Args:
        request: PredictRequest with sequence and options.

    Returns:
        PredictResponse with predictions, probabilities, and confidence.
    """
    if _inference_service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        _executor,
        lambda: _inference_service.predict(
            sequence=request.sequence,
            return_attention=request.return_attention,
            return_xai=request.return_xai,
            xai_method=request.xai_method,
        ),
    )

    return PredictResponse(**result)


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
        result = await loop.run_in_executor(
            _executor,
            lambda s=seq: _inference_service.predict(
                sequence=s,
                return_attention=request.return_attention,
                return_xai=request.return_xai,
                xai_method=request.xai_method,
            ),
        )
        results.append(PredictResponse(**result))

    total_ms = (time.time() - start) * 1000
    return BatchPredictResponse(
        results=results,
        total_sequences=len(results),
        total_processing_time_ms=round(total_ms, 2),
    )
