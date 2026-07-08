"""FastAPI application entry point for ProtIntel.

Usage:
    python backend/main.py
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.middleware.cors import configure_cors
from backend.routers import info, predict, upload
from backend.services.inference_service import InferenceService
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="ProtIntel API",
    description=(
        "Explainable Protein Secondary Structure Prediction using "
        "ESM-2, CNN-BiLSTM, and Attention"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
configure_cors(app)

# Mount evaluation static files
evaluation_dir = PROJECT_ROOT / "logs" / "evaluation"
evaluation_dir.mkdir(parents=True, exist_ok=True)
app.mount("/evaluation-images", StaticFiles(directory=str(evaluation_dir)), name="evaluation-images")

# Include routers
app.include_router(predict.router)
app.include_router(upload.router)
app.include_router(info.router)


@app.on_event("startup")
async def startup_event() -> None:
    """Load the model on application startup."""
    logger.info("Starting ProtIntel API server...")

    config = load_config()

    device = os.environ.get("DEVICE", config.inference.device)
    checkpoint = os.environ.get("MODEL_PATH", config.inference.checkpoint_path)

    service = InferenceService(
        checkpoint_path=checkpoint,
        device=device,
        model_config=config.model,
    )

    try:
        service.load_model()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.warning("API running without model — predictions will fail")

    predict.set_inference_service(service)
    logger.info("ProtIntel API ready!")


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint with API information."""
    return {
        "name": "ProtIntel API",
        "version": "1.0.0",
        "description": "Explainable Protein Secondary Structure Prediction",
        "docs": "/docs",
    }


def main() -> None:
    """Run the API server."""
    import uvicorn

    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))

    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
