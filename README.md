<!-- Premium README — polished hero, badges, TOC, and clear sections -->

# ProtIntel 🧬 — Explainable Protein Secondary Structure Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <a href="#features"><img alt="Features" src="https://img.shields.io/badge/Features-ESM--2%20%7C%20XAI%20%7C%203D%20Viz-blue"/></a> [![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/) [![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-teal)](https://fastapi.tiangolo.com/) [![React](https://img.shields.io/badge/React-18-blue)](https://react.dev/) [![Three.js](https://img.shields.io/badge/Three.js-r128-black)](https://threejs.org/)

ProtIntel is a production-ready research platform that predicts protein secondary structure (Q3 & Q8) from amino acid sequences while providing residue-level interpretability using state-of-the-art XAI methods and an interactive 3D visualizer.

- Modern model stack: ESM‑2 embeddings → multi-scale CNN → BiLSTM → multi-head attention
- Explainability built-in: Integrated Gradients, SHAP, attention rollout
- Full product path: training, evaluation, REST API (FastAPI), and React + Three.js frontend

---

## Table of contents

- [Demo](#demo)
- [Highlights](#highlights)
- [Quick start](#quick-start)
- [Architecture](#architecture)
- [REST API](#rest-api)
- [Explainability (XAI)](#explainability-xai)
- [Evaluation & Benchmarking](#evaluation--benchmarking)
- [Contributing & Code of Conduct](#contributing--code-of-conduct)
- [License](#license)

---

## Demo

![ProtIntel Dashboard Overview](docs/images/dashboard_overview.png)

Interactive 3D ribbon visualization, residue-level XAI heatmaps, benchmark panels, and CSV/JSON export for batch analyses.

---

## Highlights

- ESM‑2 (650M) per-residue embeddings (L × 1280)
- Multi-scale 1D CNNs (kernels 3,5,7) with residual connections
- Bidirectional LSTM (2 layers, hidden=256) for long-range context
- 8-head multi-head self-attention with extraction for interpretability
- Joint Q3 (H/E/C) and Q8 (DSSP 8-state) prediction
- Interactive web UI (React + Three.js) with cinematic camera tours, XAI overlays, and batch mode
- Production-ready FastAPI backend with async inference and batch endpoints

---

## Quick start (local)

1. Clone the repo

```bash
git clone https://github.com/RamaVenkataCharan/ProtIntel.git
cd ProtIntel
```

2. Python environment & install

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

3. (Optional) Download data & precompute ESM-2 embeddings

```bash
python scripts/download_data.py
python scripts/preprocess.py
python scripts/generate_embeddings.py --device cuda
```

4. Run preflight checks

```bash
python preflight_checks.py
```

5. Start backend API

```bash
python backend/main.py
# API: http://localhost:8000
```

6. Start frontend (separate terminal)

```bash
cd frontend
npm install
npm run dev
# UI: http://localhost:5173
```

---

## Architecture

```text
Input FASTA sequence
       ↓
   ESM-2 (L × 1280)
       ↓
Multi-scale 1D CNN (3,5,7)
       ↓
   Bidirectional LSTM
       ↓
Multi-head Self-Attention (8 heads)
       ↓
   ┌───────────┴───────────┐
   ↓                       ↓
Q3 Head (H/E/C)        Q8 Head (DSSP 8-state)
```

### Targets
- Q3: Helix (H), Strand (E), Coil (C)
- Q8: H, E, G, I, B, T, S, C (DSSP labels)

---

## REST API (examples)

Endpoints

| Method | Path | Description |
|---|---:|---|
| POST | /predict | Single sequence prediction (optionally return XAI) |
| POST | /predict_batch | Batch prediction (FASTA upload or multiple sequences) |
| POST | /upload | Predict from uploaded FASTA |
| GET  | /model_info | Model metadata & device status |
| GET  | /metrics | Benchmark metrics (CB513) |
| GET  | /health | Health check |

Example (cURL):

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

## Explainability (XAI)

ProtIntel provides multiple complementary attribution methods to interpret residue influence:

- Integrated Gradients (IG) — path-integral gradients for per-residue attribution
- SHAP (KernelExplainer) — Shapley-value based contributions
- Attention extraction & rollout — layer-wise attention flow and contact-like signals

Visualizations are exposed in the UI as heatmap overlays on the 3D ribbon and 2D residue strips for quick scientific inspection.

---

## Evaluation & Benchmarking

The repo includes evaluation scripts for standard datasets (CB513, CullPDB workflows). Outputs include confusion matrices, per-class F1, MCC, and confidence histograms.

Run evaluation:

```bash
python evaluate.py
```

---

## Project layout

```text
ProtIntel/
├── backend/          # FastAPI server & inference services
├── frontend/         # React + TypeScript + Three.js UI
├── src/              # PyTorch models, training, evaluation, XAI
├── docs/             # Documentation and screenshots
├── scripts/          # Data download & preprocessing
├── tests/            # Unit & integration tests
├── train.py
├── infer.py
├── evaluate.py
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

---

## Contributing & Code of Conduct

We welcome contributions — small fixes, clarifications, tests, bug reports, and new features. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow and expectations. Maintain respectful and constructive communication.

---

## Authors

- M. Rama Venkata Charan
- B. Murali Gopi
- M. Kumar Siva Sai
- Yashwanth Prakash

Academic advisor: Mrs. V. Aruna

---

## Acknowledgments

- Meta AI Research — [ESM-2](https://github.com/facebookresearch/esm)
- CullPDB & CB513 datasets
- Three.js / React Three Fiber for visualization

---

## License

ProtIntel is MIT-licensed — see [LICENSE](LICENSE) for details.

---

If you'd like, I can:
- add an eye-catching hero/banner GIF or animated preview
- add a badge row with CI, code coverage, and PyPI/pip install status (where applicable)
- open a PR instead of committing directly to main
