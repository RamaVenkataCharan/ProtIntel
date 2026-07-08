# ProtIntel — Explainable Protein Secondary Structure Prediction

<p align="center">
  <strong>ESM-2 · CNN-BiLSTM · Attention-Based Learning</strong>
</p>

## Overview

ProtIntel is a production-grade deep learning system for predicting protein secondary structure (Q3 and Q8) from amino acid sequences. It combines:

- **ESM-2** (650M parameter protein language model) for rich per-residue embeddings
- **Multi-scale CNN** for local residue pattern detection
- **Bidirectional LSTM** for long-range sequential dependencies
- **Multi-head self-attention** with explainable attention weights
- **Explainable AI** via Integrated Gradients, SHAP, and attention rollout

### Team
- B. Murali Gopi
- M. Kumar Siva Sai
- M. Rama Venkata Charan
- Yashwanth Prakash

**Guide:** Mrs. V. Aruna  
**Academic Context:** Final Year B.Tech (Computer Science) Project

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/RamaVenkataCharan/ProtIntel
cd ProtIntel
pip install -r requirements.txt

# 2. Download datasets
python scripts/download_data.py

# 3. Preprocess
python scripts/preprocess.py

# 4. Generate ESM-2 embeddings (requires ~2.5 GB download)
python scripts/generate_embeddings.py --device cuda

# 5. Train
python train.py --device cuda

# 6. Evaluate on CB513
python evaluate.py

# 7. Run API server
python backend/main.py

# 8. Run frontend (separate terminal)
cd frontend && npm install && npm run dev
```

---

## Architecture

```
Raw Amino Acid Sequence
        │
        ▼
┌─────────────────────┐
│  ESM-2 (650M)       │  Per-residue embeddings: (L × 1280)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Multi-Scale CNN     │  Kernels: 3, 5, 7 + Residual blocks
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Bidirectional LSTM  │  2 layers, captures long-range deps
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Multi-Head Attention│  8 heads, returns weights for XAI
└─────────────────────┘
        │
     ┌──┴──┐
     ▼     ▼
   Q3     Q8
  Head   Head
```

**Q3 classes:** Helix (H), Sheet (E), Coil (C)  
**Q8 classes:** H, E, G, I, B, T, S, C

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 4 GB (inference) | 12+ GB (training) |
| RAM | 8 GB | 16+ GB |
| Disk | 15 GB | 30 GB |
| Python | 3.10+ | 3.11 |

---

## Project Structure

```
ProtIntel/
├── configs/          # YAML configuration files
├── src/
│   ├── data/         # Dataset, preprocessing, augmentation
│   ├── models/       # ESM-2, CNN, BiLSTM, attention, full model
│   ├── training/     # Losses, metrics, callbacks, trainer
│   ├── evaluation/   # Evaluator, visualizer
│   ├── xai/          # Integrated Gradients, SHAP, attention rollout
│   └── utils/        # Config, logging, I/O, reproducibility
├── backend/          # FastAPI REST API
├── frontend/         # React/TypeScript UI
├── scripts/          # Data download, preprocessing, benchmarking
├── tests/            # Unit, integration, API tests
├── docs/             # Documentation
└── docker/           # Dockerfiles and docker-compose
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/predict` | Single sequence prediction |
| POST | `/predict_batch` | Batch prediction (up to 50) |
| POST | `/upload` | Upload FASTA file |
| GET | `/model_info` | Architecture info |
| GET | `/metrics` | Benchmark metrics |
| GET | `/health` | Health check |

Example:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sequence": "MKFLILLFNILCLFPVLAADNHGVSMNAS", "return_xai": true}'
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v
```

---

## Citation

```bibtex
@software{protintel2024,
  title={ProtIntel: Explainable Protein Secondary Structure Prediction},
  author={Gopi, B. Murali and Sai, M. Kumar Siva and Charan, M. Rama Venkata and Prakash, Yashwanth},
  year={2024},
  url={https://github.com/RamaVenkataCharan/ProtIntel}
}
```

---

## License

MIT License
