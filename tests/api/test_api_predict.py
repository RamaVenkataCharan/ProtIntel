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
    resp = real_client.post("/predict", json={
        "sequence": sequence,
        "return_xai": True,
        "xai_method": "ig"
    })
    assert resp.status_code == 200
    
    data = resp.json()
    assert data["sequence"] == sequence
    
    q3_preds = data["q3_prediction"]
    q8_preds = data["q8_prediction"]
    
    assert len(q3_preds) == len(sequence)
    assert len(q8_preds) == len(sequence)
    
    # Assert predictions are non-trivial (not collapsed to majority-class only)
    q3_set = set(q3_preds)
    q8_set = set(q8_preds)
    
    print(f"Real Q3 prediction: {''.join(q3_preds)}")
    print(f"Real Q8 prediction: {''.join(q8_preds)}")
    
    # Ensure prediction contains Helix ('H') or Sheet ('E')
    assert any(c in q3_set for c in ("H", "E")), f"Q3 prediction collapsed: {q3_preds}"
    assert any(c in q8_set for c in ("H", "E")), f"Q8 prediction collapsed: {q8_preds}"

    # Verify XAI Attributions
    assert "residue_importance" in data
    importance = data["residue_importance"]
    assert importance is not None
    assert len(importance) == len(sequence)
    
    # Check that attributions are non-degenerate (not all identical, not all zero)
    assert len(set(importance)) > 1, "Attribution scores are degenerate (all identical)"
    assert any(score > 0 for score in importance), "Attribution scores are all zero"
    
    # Check they are normalized within 0 to 1 range
    assert all(0.0 <= score <= 1.0 for score in importance), "Attribution scores not normalized to [0, 1]"

