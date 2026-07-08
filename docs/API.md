# ProtIntel API Documentation

## Base URL

```
http://localhost:8000
```

## Authentication

No authentication required for the current version.

---

## Endpoints

### POST /predict

Predict secondary structure for a single protein sequence.

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": "MKFLILLFNILCLFPVLAADNHGVSMNAS",
    "return_attention": false,
    "return_xai": true,
    "xai_method": "ig"
  }'
```

**Response:**
```json
{
  "protein_id": "a1b2c3d4e5f6g7h8",
  "sequence": "MKFLILLFNILCLFPVLAADNHGVSMNAS",
  "length": 29,
  "q3_prediction": ["H", "H", "H", "E", "E", "C", "C", "..."],
  "q8_prediction": ["H", "H", "G", "E", "E", "T", "C", "..."],
  "q3_probabilities": [[0.85, 0.10, 0.05], ...],
  "q8_probabilities": [[0.70, 0.05, 0.15, ...], ...],
  "confidence": [0.85, 0.92, 0.78, ...],
  "residue_importance": [0.12, 0.45, 0.89, ...],
  "xai_method": "ig",
  "processing_time_ms": 156.3
}
```

### POST /predict_batch

Batch prediction for up to 50 sequences.

```bash
curl -X POST http://localhost:8000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "sequences": ["MKFLILLFNILCLFPVLAAD", "ACDEFGHIKLMNPQRSTVWY"],
    "return_attention": false,
    "return_xai": false
  }'
```

### POST /upload

Upload a FASTA file for batch prediction.

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@proteins.fasta"
```

### GET /model_info

```bash
curl http://localhost:8000/model_info
```

Response:
```json
{
  "model_name": "ProtIntel",
  "version": "1.0.0",
  "architecture": "ESM-2 → CNN → BiLSTM → Attention → Q3/Q8",
  "esm2_model": "facebook/esm2_t33_650M_UR50D",
  "total_parameters": 660000000,
  "trainable_parameters": 5200000
}
```

### GET /metrics

```bash
curl http://localhost:8000/metrics
```

### GET /health

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

---

## Error Handling

All errors return JSON with a `detail` field:

```json
{
  "detail": "Invalid amino acid characters: ['1', '2']"
}
```

| Status Code | Meaning |
|------------|---------|
| 400 | Invalid input (bad sequence, too many sequences) |
| 422 | Validation error (missing required fields) |
| 503 | Model not loaded |
