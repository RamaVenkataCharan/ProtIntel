# ProtIntel Documentation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/) [![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-teal)](https://fastapi.tiangolo.com/) [![React](https://img.shields.io/badge/React-18-blue)](https://react.dev/) [![Three.js](https://img.shields.io/badge/Three.js-r128-black)](https://threejs.org/)

This folder contains project-level documentation and reference material for ProtIntel, an explainable protein secondary structure prediction system powered by ESM-2 embeddings and a CNN–BiLSTM–Attention architecture.

---

## Overview

ProtIntel predicts protein secondary structure from amino acid sequences in two complementary settings:

- Q3 prediction: Helix, Strand, Coil
- Q8 prediction: DSSP 8-state labels (H, E, G, I, B, T, S, C)

The system is designed for both scientific research and practical deployment. It produces high-level confidence estimates and interpretable residue-level explanations using XAI techniques such as Integrated Gradients and SHAP.

---

## Why this project matters

Protein structure prediction remains a foundational problem in bioinformatics and computational biology. Traditional sequence-based methods often struggle to capture global structural context. ProtIntel addresses this by combining:

- ESM-2 protein embeddings for rich contextual sequence representation
- multi-scale CNNs for local motif detection
- bidirectional LSTM for long-range sequence reasoning
- attention modules for interpretability and dependency modeling
- XAI visualizations to explain model decisions

---

## System architecture

```text
Amino Acid Sequence (FASTA)
          ↓
[ESM-2 650M Transformer]
   Per-residue embeddings: L × 1280
          ↓
[Multi-scale 1D CNN]
   kernels: 3, 5, 7 + residual blocks
          ↓
[Bidirectional LSTM]
   long-range context modeling
          ↓
[Multi-head self-attention]
   explainable sequence dependencies
          ↓
   ┌───────────────┴───────────────┐
   ↓                               ↓
Q3 Head                         Q8 Head
(H / E / C)                  (DSSP 8-state)
```

---

## Documentation structure

```text
docs/
├── README.md               # Project documentation overview
├── images/                 # UI screenshots and model visual assets
├── architecture/           # Architecture notes and diagrams (if added)
├── api/                    # API docs and usage examples (if added)
├── evaluation/             # Benchmark and metrics documentation (if added)
└── notebooks/             # Optional exploratory analysis notebooks
```

---

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/RamaVenkataCharan/ProtIntel.git
cd ProtIntel
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run backend

```bash
python backend/main.py
```

### 5. Launch frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend is typically served at `http://localhost:5173`, and the API runs at `http://localhost:8000`.

---

## API overview

The backend exposes a minimal REST API for inference and analysis.

| Method | Endpoint | Purpose |
|:---:|:---|:---|
| `POST` | `/predict` | Single sequence prediction |
| `POST` | `/predict_batch` | Batch FASTA or multi-sequence prediction |
| `POST` | `/upload` | Predict from uploaded FASTA file |
| `GET` | `/model_info` | Model metadata and runtime configuration |
| `GET` | `/metrics` | Evaluation metrics and benchmark summaries |
| `GET` | `/health` | Server health check |

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    "return_xai": true,
    "xai_method": "ig"
  }'
```

---

## Explainability and interpretability

ProtIntel is designed not only to predict secondary structure but also to explain it.

### Included explainability methods

- Integrated Gradients (IG)
- SHAP values
- Attention-based attribution and rollout analysis

These techniques enable residue-level explanation and help identify which amino acids drive a prediction.

---

## Evaluation workflow

The project includes evaluation utilities for benchmarking on protein secondary structure datasets, including CB513-style evaluation pipelines.

Typical outputs include:

- confusion matrices
- per-class precision/recall/F1
- MCC and accuracy summaries
- confidence distributions
- predicted class breakdowns

---

## Training and data flow

A typical workflow:

1. Download the benchmark dataset
2. Preprocess protein sequences
3. Generate ESM-2 embeddings
4. Train the model with sequence-label supervision
5. Evaluate on validation or benchmark splits
6. Run inference via CLI or web UI
7. Inspect explanations through the 3D visualization and XAI panels

---

## Team and academic context

- M. Rama Venkata Charan
- B. Murali Gopi
- M. Kumar Siva Sai
- Yashwanth Prakash

Guide: Mrs. V. Aruna

Academic context: Final Year B.Tech Computer Science Project

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

This project is licensed under the [MIT License](../LICENSE).

---

## Contributing

Contributions are welcome. Please see the repository-level [CONTRIBUTING.md](../CONTRIBUTING.md) for coding standards, workflow guidance, and pull request expectations.
