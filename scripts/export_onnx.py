"""ONNX Model Exporter for ProtIntel.

Exports the downstream prediction pipeline to ONNX format for accelerated CPU/GPU serving.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.models.protintel_model import ProtIntelModel
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def export_onnx(output_path: str = "models/protintel_downstream.onnx") -> Path:
    config = load_config()
    model = ProtIntelModel(config=config.model, device="cpu")
    model.eval()

    dummy_embeddings = torch.randn(1, 100, 1280)
    dummy_mask = torch.ones(1, 100, dtype=torch.long)
    dummy_lengths = torch.tensor([100], dtype=torch.long)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting ProtIntel downstream model to ONNX: {out_file}")
    torch.onnx.export(
        model,
        (None, dummy_embeddings, dummy_mask, dummy_lengths),
        str(out_file),
        input_names=["embeddings", "attention_mask", "seq_lengths"],
        output_names=["q3_logits", "q8_logits", "q3_probs", "q8_probs"],
        dynamic_axes={
            "embeddings": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "q3_logits": {0: "batch_size", 1: "seq_len"},
            "q8_logits": {0: "batch_size", 1: "seq_len"},
        },
        opset_version=14,
    )
    logger.info("ONNX export completed successfully!")
    return out_file


if __name__ == "__main__":
    export_onnx()
