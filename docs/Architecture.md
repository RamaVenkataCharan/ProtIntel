# ProtIntel Architecture

## Design Philosophy

ProtIntel uses a **hybrid architecture** that combines the strengths of four complementary paradigms:

1. **Pre-trained protein language model (ESM-2)** — Captures evolutionary and physicochemical context from training on 250M+ protein sequences
2. **Multi-scale CNN** — Detects local residue patterns at multiple spatial scales (3, 5, 7 residues)
3. **Bidirectional LSTM** — Captures long-range sequential dependencies across the full protein chain
4. **Self-attention** — Enables direct interaction between distant residues and produces interpretable attention weights

This combination addresses a key research gap: existing methods typically use only one or two of these paradigms.

## Architecture Diagram

```mermaid
graph TD
    A["Raw AA Sequence"] --> B["ESM-2 Tokenizer"]
    B --> C["ESM-2 (650M params)"]
    C --> D["Per-residue embeddings<br/>(L × 1280)"]
    D --> E["Multi-Scale Conv1D<br/>K=3, K=5, K=7"]
    E --> F["Concatenate + Project"]
    F --> G["Residual Blocks × 3"]
    G --> H["BiLSTM (2 layers)"]
    H --> I["Multi-Head Self-Attention<br/>(8 heads)"]
    I --> J["Feed-Forward Block"]
    J --> K["Q3 Head (H/E/C)"]
    J --> L["Q8 Head (8 classes)"]
    I --> M["Attention Weights<br/>(for XAI)"]

    style A fill:#0A0E1A,stroke:#00D4FF,color:#fff
    style C fill:#7C3AED,stroke:#fff,color:#fff
    style E fill:#00D4FF,stroke:#fff,color:#000
    style H fill:#FF6B6B,stroke:#fff,color:#fff
    style I fill:#4ECDC4,stroke:#fff,color:#000
    style K fill:#FF6B6B,stroke:#fff,color:#fff
    style L fill:#4ECDC4,stroke:#fff,color:#000
```

## Component Details

### ESM-2 Embedding Generator
- **Model:** `facebook/esm2_t33_650M_UR50D` (33 transformer layers, 650M parameters)
- **Output:** Per-residue embeddings of dimension 1280
- **Mode:** Frozen by default; optionally fine-tune last N layers
- **Optimization:** Disk caching via SHA-256 sequence hashing

### Multi-Scale CNN Encoder
- **Parallel convolutions:** Kernel sizes 3, 5, 7 capture local patterns at different scales
- **Residual blocks:** 3 stacked blocks with BatchNorm and dropout
- **All convolutions use `padding='same'`** to preserve sequence length
- **Output dimension:** Configurable (default 512)

### Bidirectional LSTM
- **2 bidirectional layers:** Each direction has 256 hidden units → 512 total
- **Variable-length handling:** Uses `pack_padded_sequence` for efficient computation
- **Dropout:** Applied between layers

### Multi-Head Self-Attention
- **8 attention heads** with pre-LayerNorm formulation
- **Key feature:** Returns attention weights for explainability
- **Residual connection** for training stability

### Prediction Heads
- **Q3 Head:** Predicts Helix (H), Sheet (E), Coil (C)
- **Q8 Head:** Predicts H, E, G, I, B, T, S, C
- **Architecture:** Linear → ReLU → Dropout → Linear → Softmax
- **Returns:** Logits, probabilities, and confidence (max probability)

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| ESM-2 over ProtBERT/ProtT5 | Higher downstream accuracy, efficient 650M size |
| Frozen ESM-2 by default | Reduces VRAM requirements; embeddings are already rich |
| Multi-scale CNN | Different kernel sizes capture different structural motifs |
| Pre-norm attention | More stable training than post-norm |
| Dual Q3/Q8 heads | Multi-task learning improves Q3 via Q8's finer labels |
| Label smoothing | Prevents overconfident predictions on ambiguous residues |
