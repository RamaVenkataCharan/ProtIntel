# ProtIntel

**Explainable Protein Secondary Structure Prediction using ESM-2, CNN-BiLSTM, and Multi-Head Attention**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)

---

## Overview

**ProtIntel** is a production-grade deep learning system for predicting protein secondary structure (Q3 and Q8) from amino acid sequences. It achieves interpretable predictions by combining a state-of-the-art protein language model (ESM-2) with explainable AI techniques.

### Key Features

- **ESM-2 Embeddings**: Leverages the 650M parameter protein language model for rich per-residue representations (1280D)
- **Multi-Scale Convolutional Architecture**: Captures local residue patterns at multiple temporal scales (kernels 3, 5, 7)
- **Bidirectional LSTM**: Models long-range sequential dependencies with 2 stacked layers
- **Multi-Head Self-Attention**: 8-head attention mechanism with interpretable attention weights
- **Explainable AI (XAI)**: Built-in attribution methods:
  - **Integrated Gradients**: Gradient-based input attribution
  - **SHAP**: Game-theoretic feature importance
  - **Attention Rollout**: Hierarchical attention visualization
- **Dual Task Learning**: Simultaneous Q3 (3-class) and Q8 (8-class) prediction for richer supervision
- **Production-Ready API**: FastAPI backend with batch processing, FASTA upload, and model introspection endpoints
- **Comprehensive Evaluation**: Confusion matrices, per-class metrics, and benchmark comparisons on CB513

---

## Model Architecture

```
Input Amino Acid Sequence
        ↓
    [ESM-2 650M]
    Per-residue embeddings: L × 1280
        ↓
  [Multi-Scale CNN]
  Kernels: 3, 5, 7 + Residual blocks
        ↓
[Bidirectional LSTM]
2 layers, hidden=256
        ↓
[Multi-Head Attention]
8 heads, returns weight matrices
        ↓
    ┌────┴─────┐
    ↓          ↓
  Q3 Head    Q8 Head
   (×3)       (×8)
```

### Q3 Classification (3 classes)
- **H** (Helix)
- **E** (Extended/Sheet)
- **C** (Coil)

### Q8 Classification (8 classes)
- **H** (α-Helix)
- **E** (β-Sheet)
- **G** (3₁₀ Helix)
- **I** (π-Helix)
- **B** (β-Bridge)
- **T** (Turn)
- **S** (Bend)
- **C** (Coil)

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU VRAM** | 4 GB (inference only) | 12+ GB (training) |
| **RAM** | 8 GB | 16+ GB |
| **Disk Space** | 15 GB | 30 GB |
| **Python** | 3.10 | 3.11+ |
| **CUDA** | Optional | 11.8+ (for GPU) |

---

## Installation & Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/RamaVenkataCharan/ProtIntel
cd ProtIntel
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download & Preprocess Datasets
```bash
python scripts/download_data.py
python scripts/preprocess.py
```

### 4. Generate ESM-2 Embeddings (Critical Step)
This pre-computes embeddings for all training sequences (~2.5 GB download):
```bash
python scripts/generate_embeddings.py --device cuda
```
> **Note**: CPU embedding generation is possible but very slow (~24+ hours). GPU highly recommended.

### 5. Run Pre-Flight Diagnostics
Validate your environment before training:
```bash
python preflight_checks.py
```
This runs 6 sanity checks:
- Data loading & tensor shape verification
- Label distribution analysis
- Embedding cache coverage
- Single-batch overfit test
- Dataset file integrity
- Checkpoint validity

### 6. Train Model
```bash
python train.py --device cuda --epochs 50 --batch-size 16
```

### 7. Evaluate on CB513 Benchmark
```bash
python evaluate.py --device cuda --output-dir logs/evaluation
```

### 8. Launch API Server
```bash
python backend/main.py
# Server running at http://localhost:8000
```

### 9. Run Frontend (in separate terminal)
```bash
cd frontend
npm install
npm run dev
# UI running at http://localhost:5173
```

---

## Usage

### Single-Sequence Prediction (CLI)
```bash
python infer.py "MKFLILLFNILCLFPVLAADNHGVSMNAS"
python infer.py --file proteins.fasta --device cuda --xai
```

**Output includes:**
- Q3 and Q8 predictions with per-residue confidence scores
- Secondary structure composition (H%, E%, C%)
- Average prediction confidence
- (Optional) XAI attribution scores for top residues

### Batch Prediction via API
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": "MKFLILLFNILCLFPVLAADNHGVSMNAS",
    "return_xai": true,
    "xai_method": "ig"
  }'
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/predict` | Single sequence prediction |
| `POST` | `/predict_batch` | Batch prediction (up to 50 sequences) |
| `POST` | `/upload` | Upload and predict from FASTA file |
| `GET` | `/model_info` | Architecture and parameter count |
| `GET` | `/metrics` | Benchmark metrics (Q3, Q8 accuracy, etc.) |
| `GET` | `/health` | Health check |

---

## Project Structure

```
ProtIntel/
├── configs/                    # YAML configuration files (model, training, data)
├── src/
│   ├── data/
│   │   ├── data_module.py     # PyTorch Lightning DataModule
│   │   ├── dataset.py         # Custom Dataset with ESM-2 caching
│   │   ├── preprocessor.py    # Label encoding, class weight computation
│   │   └── fasta_parser.py    # FASTA I/O utilities
│   ├── models/
│   │   ├── protintel_model.py # Full model pipeline
│   │   ├── esm2_encoder.py    # ESM-2 integration
│   │   ├── cnn_layer.py       # Multi-scale CNN with residuals
│   │   ├── lstm_layer.py      # Bidirectional LSTM
│   │   └── attention.py       # Multi-head self-attention
│   ├── training/
│   │   ├── trainer.py         # Training loop, checkpointing, metrics
│   │   ├── losses.py          # Cross-entropy with label smoothing & class weights
│   │   ├── metrics.py         # Accuracy, F1, MCC per-class
│   │   └── callbacks.py       # Early stopping, learning rate scheduling
│   ├── evaluation/
│   │   ├── evaluator.py       # Benchmark evaluation on test sets
│   │   └── visualizer.py      # Confusion matrices, per-class plots
│   ├── xai/
│   │   ├── integrated_gradients.py  # Gradient-based attribution
│   │   ├── shap_explainer.py        # SHAP kernel explainer
│   │   └── attention_rollout.py     # Attention weight visualization
│   └── utils/
│       ├── config_loader.py   # YAML config parsing
│       ├── logger.py          # Structured logging
│       ├── reproducibility.py # Seed setting, device management
│       └── io_utils.py        # File I/O, sequence hashing
├── backend/
│   ├── main.py                # FastAPI application
│   ├── routes.py              # Endpoint handlers
│   ├── schemas.py             # Pydantic request/response models
│   └── services/
│       ├── inference_service.py        # Prediction orchestration
│       └── explanation_service.py      # XAI computation
├── frontend/                  # React + TypeScript + Vite UI
│   ├── src/
│   │   ├── components/        # React components
│   │   ├── pages/             # Page layouts
│   │   └── services/          # API client
│   └── package.json
├── scripts/                   # Utility scripts
│   ├── download_data.py       # CullPDB & CB513 download
│   ├── preprocess.py          # Data parsing & label encoding
│   └── generate_embeddings.py # ESM-2 embedding pre-computation
├── tests/                     # Unit & integration tests
│   ├── unit/                  # Model, data loader, loss function tests
│   └── integration/           # End-to-end, API tests
├── docs/                      # Documentation
│   ├── README.md              # Extended usage guide
│   └── ARCHITECTURE.md        # Detailed design decisions
├── docker/                    # Dockerfiles
│   ├── Dockerfile.train       # Training image
│   └── docker-compose.yml     # Multi-container setup
├── train.py                   # Top-level training entry point
├── infer.py                   # CLI inference script
├── evaluate.py                # Benchmark evaluation script
├── preflight_checks.py        # Pre-training diagnostics (6 checks)
├── pyproject.toml             # Project metadata & tool config
├── requirements.txt           # Production dependencies
├── requirements-dev.txt       # Development dependencies
└── LICENSE                    # MIT License
```

---

## Training Pipeline

### 1. Configuration
All hyperparameters are in `configs/default.yaml`:
```yaml
model:
  esm2_model_id: "facebook/esm2_t33_650M_UR50D"
  cnn_hidden: 128
  lstm_hidden: 256
  attention_heads: 8
  dropout: 0.3

training:
  epochs: 50
  batch_size: 16
  learning_rate: 1e-4
  warmup_steps: 500
  max_grad_norm: 1.0
  loss:
    type: "cross_entropy"
    label_smoothing: 0.1
    use_class_weights: true

data:
  train_ratio: 0.9
  val_ratio: 0.1
  max_length: 700
  pad_idx: -100
```

### 2. Data Loading
- **CullPDB 6133**: ~5,600 training sequences (processed to 4,600 after filtering)
- **CB513**: ~500 test sequences (benchmark)
- **Caching**: ESM-2 embeddings cached as `.pt` files to avoid recomputation
- **Data Augmentation**: Length-based sampling, sequence masking available

### 3. Loss Functions
```
Total Loss = Q3 Loss + 0.5 × Q8 Loss
```
- **Cross-entropy** with optional label smoothing (ε=0.1)
- **Class weights** computed from training set distribution to handle class imbalance
- **Focal loss** option for harder examples

### 4. Optimization
- **Optimizer**: Adam (β₁=0.9, β₂=0.999)
- **Learning Rate**: 1e-4 with linear warmup (500 steps) + cosine annealing decay
- **Gradient Clipping**: max norm = 1.0
- **Regularization**: L2 weight decay = 1e-5, dropout = 0.3

### 5. Validation & Checkpointing
- Validation every epoch on 10% held-out training data
- Metrics tracked: Q3/Q8 accuracy, precision, recall, F1, Matthews correlation coefficient
- Best checkpoint saved based on Q3 validation accuracy
- Early stopping if no improvement for 5 epochs

---

## Evaluation Metrics

### Per-Class Metrics
```
Q3 Accuracy = (TP_H + TP_E + TP_C) / Total
Q8 Accuracy = (TP_H + TP_E + ... + TP_C) / Total

Per-class F1 = 2 × (Precision × Recall) / (Precision + Recall)
MCC = (TP×TN - FP×FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

### Benchmark Results
Expected performance on CB513 test set:
- **Q3 Accuracy**: ~78-82%
- **Q8 Accuracy**: ~68-72%
- **Inference Speed**: ~10-50 ms per protein (CPU)

### Confusion Matrix Visualization
The evaluator generates confusion matrices for both Q3 and Q8 tasks, saved as PNG plots in `logs/evaluation/`.

---

## Explainable AI (XAI)

### Integrated Gradients
Computes gradient-based attribution by integrating gradients along a straight-line path from baseline to input:
```python
result = service.predict(sequence="...", return_xai=True, xai_method="ig")
# Returns: residue_importance scores (0.0 to 1.0)
```

### SHAP (SHapley Additive exPlanations)
Game-theoretic approach using the Kernel explainer:
```python
result = service.predict(sequence="...", return_xai=True, xai_method="shap")
```

### Attention Rollout
Visualizes hierarchical attention patterns across all layers and heads by averaging and composing attention matrices.

### Interpretation
- **High attribution**: Residues strongly influence the model's prediction
- **Low attribution**: Residues are near-neutral or strongly contradicted by context
- **Useful for**:
  - Identifying active sites
  - Validating model predictions
  - Comparing model reasoning across sequence variants

---

## Development & Testing

### Run Tests
```bash
# All tests
pytest tests/ -v

# Unit tests only (fast)
pytest tests/unit/ -v

# Integration tests (requires checkpoints)
pytest tests/integration/ -v

# API tests
pytest tests/api/ -v
```

### Code Quality
```bash
# Format with Black
black src/ backend/ tests/

# Type checking with mypy
mypy src/ backend/ --ignore-missing-imports

# Linting with isort
isort src/ backend/ tests/
```

---

## Docker Deployment

### Build Training Image
```bash
docker build -f docker/Dockerfile.train -t protintel:latest .
```

### Run in Container
```bash
docker run --gpus all -it \
  -v $(pwd)/datasets:/workspace/datasets \
  -v $(pwd)/models:/workspace/models \
  protintel:latest \
  python train.py --device cuda --epochs 50
```

### Multi-Container Setup
```bash
docker-compose -f docker/docker-compose.yml up --build
# Launches: training, API server, frontend, TensorBoard
```

---

## Troubleshooting

### Dataset not found
```bash
python scripts/download_data.py
```

### Slow training (CPU-only)
Pre-compute embeddings with GPU access:
```bash
python scripts/generate_embeddings.py --device cuda
```

### Out of Memory (OOM)
Reduce batch size or max sequence length in `configs/default.yaml`:
```yaml
training:
  batch_size: 8  # was 16

data:
  max_length: 400  # was 700
```

### Validation metrics not improving
Check:
1. Dataset label distribution (run `preflight_checks.py` Check 2)
2. Learning rate (try 5e-5 or 5e-4)
3. Embedding quality (visualize sample embeddings)

---

## Performance Benchmarks

### Hardware
- **GPU**: NVIDIA A100 (40GB)
- **CPU**: Intel Xeon (16 cores)
- **RAM**: 128 GB

### Metrics
| Task | GPU | CPU |
|------|-----|-----|
| Single inference | 15 ms | 50 ms |
| Batch (16) inference | 200 ms | 800 ms |
| Training (1 epoch) | 2.5 min | 45 min |
| ESM-2 embedding generation | 0.5s/seq | 3-5s/seq |

---

## Citation

If you use ProtIntel in your research, please cite:

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

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) file for details.

---

## Team

- **B. Murali Gopi**
- **M. Kumar Siva Sai**
- **M. Rama Venkata Charan**
- **Yashwanth Prakash**

**Academic Advisor:** Mrs. V. Aruna  
**Context:** Final Year B.Tech (Computer Science) Project

---

## Acknowledgments

- **ESM-2 Model**: [Meta AI, Facebook](https://github.com/facebookresearch/esm)
- **Datasets**: [Princeton ICML 2014](https://www.princeton.edu/~jzthree/datasets/ICML2014/)
- **PyTorch Lightning**: Simplified training loop abstraction
- **Hugging Face Transformers**: Model hosting and utilities

---

## Support & Contribution

For issues, feature requests, or contributions:
1. Open an [issue](https://github.com/RamaVenkataCharan/ProtIntel/issues)
2. Fork and submit a [pull request](https://github.com/RamaVenkataCharan/ProtIntel/pulls)
3. Follow existing code style (Black, isort, type hints)

---

**Last Updated:** July 2024  
**Status:** Active development
