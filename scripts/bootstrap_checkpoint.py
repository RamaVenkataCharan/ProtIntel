"""Bootstrap a fresh 480-dim checkpoint so the API can start immediately.

This saves a randomly-initialized ProtIntelModel (esm2_t12_35M / 480-dim)
as best_checkpoint.pt so the backend loads without errors.  Weights are
random — actual training via train.py will overwrite this.
"""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.models.protintel_model import ProtIntelModel
from src.utils.config_loader import load_config

print("Loading config...")
config = load_config()

print(f"ESM-2 model : {config.model.esm2.model_name}")
print(f"Embedding dim: {config.model.esm2.embedding_dim}")

print("\nBuilding model with fresh random weights (no ESM-2 download needed)...")
model = ProtIntelModel(config=config.model, device="cpu")

# We DON'T call _load_model() on the ESM-2 backbone — we just save
# the downstream weights (CNN, BiLSTM, attention, heads) which are
# already randomly initialized.  The ESM-2 weights are loaded lazily
# at inference time.
save_dir = PROJECT_ROOT / "models"
save_dir.mkdir(exist_ok=True)

# Remove old incompatible checkpoints
removed = []
for old in save_dir.glob("*.pt"):
    old.unlink()
    removed.append(old.name)
if removed:
    print(f"\nRemoved old checkpoints: {removed}")

checkpoint = {
    "epoch": 0,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": {},
    "val_q3_accuracy": 0.0,
    "val_q8_accuracy": 0.0,
    "config": config.model.model_dump(),
    "note": "Bootstrap checkpoint — random weights, replace with trained model.",
}

out = save_dir / "best_checkpoint.pt"
torch.save(checkpoint, str(out))
print(f"\n✅ Saved bootstrap checkpoint → {out}")
print("   Size:", round(out.stat().st_size / 1e6, 1), "MB")
print("\nBackend will now start without checkpoint errors.")
print("Run train.py to replace this with a trained checkpoint.")
