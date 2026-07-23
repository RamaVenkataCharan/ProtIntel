import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app
from backend.routers.predict import set_inference_service
from backend.services.inference_service import InferenceService
from src.utils.config_loader import load_config

@pytest.fixture(scope="module")
def real_client():
    config = load_config()
    checkpoint_path = Path(PROJECT_ROOT / "models" / "best_checkpoint.pt")
    if not checkpoint_path.exists():
        pytest.skip("Trained checkpoint not found. Skipping real API test.")
        
    service = InferenceService(
        checkpoint_path=str(checkpoint_path),
        device="cpu",
        model_config=config.model,
    )
    service.load_model()
    set_inference_service(service)
    
    with TestClient(app) as client:
        yield client

def test_real_prediction_non_trivial(real_client):
    # Test with a real valid sequence (human myoglobin fragment or similar)
    sequence = "MGLSDGEWQLVLNVWGKVEADIPGHGQEVLIRLFKGHPETLEKFDKFKHLKSEDEMKASE"
    resp = real_client.post("/predict", json={"sequence": sequence})
    assert resp.status_code == 200
    
    data = resp.json()
    assert data["sequence"] == sequence
    
    q3_preds = data["q3_prediction"]
    q8_preds = data["q8_prediction"]
    
    assert len(q3_preds) == len(sequence)
    assert len(q8_preds) == len(sequence)
    
    # Assert predictions are non-trivial (not collapsed to majority-class only)
    # The collapsed model predicts only Coil ('C') for Q3 or Turn ('T')/Coil ('C') for Q8
    q3_set = set(q3_preds)
    q8_set = set(q8_preds)
    
    print(f"Real Q3 prediction: {''.join(q3_preds)}")
    print(f"Real Q8 prediction: {''.join(q8_preds)}")
    
    # Ensure prediction contains Helix ('H') or Sheet ('E')
    assert any(c in q3_set for c in ("H", "E")), f"Q3 prediction collapsed: {q3_preds}"
    assert any(c in q8_set for c in ("H", "E")), f"Q8 prediction collapsed: {q8_preds}"
