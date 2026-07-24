"""Upload router for FASTA file processing."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, UploadFile, File

from backend.schemas.response import BatchPredictResponse, PredictResponse
from backend.routers.predict import get_inference_service

router = APIRouter(tags=["Upload"])


@router.post("/upload", response_model=BatchPredictResponse)
async def upload_fasta(file: UploadFile = File(...)) -> BatchPredictResponse:
    """Upload a FASTA file and predict structures for all sequences.

    Args:
        file: Uploaded FASTA file.

    Returns:
        BatchPredictResponse with predictions for all sequences.
    """
    svc = get_inference_service()
    if svc is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.filename or not file.filename.endswith((".fasta", ".fa", ".faa", ".txt")):
        raise HTTPException(
            status_code=400,
            detail="File must be a FASTA file (.fasta, .fa, .faa, .txt)",
        )

    content = await file.read()
    text = content.decode("utf-8")

    # Parse FASTA
    from src.data.fasta_parser import parse_fasta_string
    records = parse_fasta_string(text)

    if not records:
        raise HTTPException(status_code=400, detail="No valid sequences found in file")

    if len(records) > 50:
        raise HTTPException(
            status_code=400,
            detail=f"Too many sequences ({len(records)}). Maximum is 50.",
        )

    import time
    start = time.time()

    results: list[PredictResponse] = []
    for record in records:
        result = svc.predict(sequence=record["sequence"])
        result["protein_id"] = record["id"]
        results.append(PredictResponse(**result))

    total_ms = (time.time() - start) * 1000
    return BatchPredictResponse(
        results=results,
        total_sequences=len(results),
        total_processing_time_ms=round(total_ms, 2),
    )
