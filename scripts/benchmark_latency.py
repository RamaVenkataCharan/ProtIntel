"""Latency and throughput benchmarking script for ProtIntel.

Measures CPU/GPU inference latency (ms per sequence) and peak memory usage across sequence lengths.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.models.protintel_model import ProtIntelModel
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def benchmark_latency(device: str = "cpu", num_runs: int = 50) -> None:
    logger.info(f"Benchmarking ProtIntel inference latency on {device.upper()}...")
    config = load_config()
    model = ProtIntelModel(config=config.model, device=device)
    model.eval()

    seq_lengths = [100, 256, 512]
    for L in seq_lengths:
        dummy_embeddings = torch.randn(1, L, 1280, device=device)
        dummy_mask = torch.ones(1, L, dtype=torch.long, device=device)
        dummy_lens = torch.tensor([L], dtype=torch.long, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(embeddings=dummy_embeddings, attention_mask=dummy_mask, seq_lengths=dummy_lens)

        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(embeddings=dummy_embeddings, attention_mask=dummy_mask, seq_lengths=dummy_lens)
        elapsed_ms = ((time.time() - start_time) / num_runs) * 1000.0

        logger.info(f"  Sequence Length L={L:3d} | Avg Latency: {elapsed_ms:.2f} ms / sequence")


if __name__ == "__main__":
    benchmark_latency()
