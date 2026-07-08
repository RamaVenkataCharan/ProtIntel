# ProtIntel Research Background

## Related Work

Protein secondary structure prediction has a rich history spanning statistical methods, machine learning, and deep learning approaches.

### Key Prior Methods

| Method | Year | Architecture | Q3 Accuracy | Notes |
|--------|------|-------------|-------------|-------|
| GOR V | 2002 | Information theory | ~73% | Statistical, no ML |
| PSIPRED | 1999 | Position-specific scoring matrix + FFN | ~78% | First widely-used neural approach |
| DeepCNF | 2016 | Deep CNN + Conditional Neural Fields | ~82% | First deep learning breakthrough |
| NetSurfP-2.0 | 2019 | CNN + BiLSTM | ~85% | Multi-task learning |
| Porter 5 | 2019 | Ensemble BiLSTM | ~84% | Profile-based |
| SAINT | 2020 | Self-attention | ~84% | First attention-based PSSP |
| ProtTrans (ProtBERT) | 2021 | Pre-trained LM + downstream | ~84% | Protein language model approach |
| ESM-based methods | 2022+ | ESM-2 + downstream | ~86% | State-of-the-art |

### What ProtIntel Improves

1. **Unified architecture:** Combines ESM-2 embeddings + CNN + BiLSTM + attention in a single model, exploiting complementary strengths
2. **Explainability:** Built-in XAI via Integrated Gradients, SHAP, and attention rollout — most existing methods are black boxes
3. **Multi-scale local features:** Parallel convolutions with kernels 3, 5, 7 capture different structural motifs
4. **Practical deployment:** Full production system with API, frontend, and Docker

### ESM-2 Advantages

ESM-2 (Evolutionary Scale Modeling) provides several advantages over profile-based features:

- **No MSA needed:** Traditional methods require multiple sequence alignments (slow, database-dependent). ESM-2 works directly on single sequences
- **Rich representations:** Trained on 250M+ protein sequences, capturing evolutionary patterns implicitly
- **Transfer learning:** Pre-trained representations generalize well to diverse protein families

### Class Imbalance in PSSP

Q8 classification suffers from severe class imbalance:
- **Common:** H (α-helix ~32%), C (coil ~25%), E (β-strand ~22%)
- **Rare:** G (~4%), T (~7%), S (~5%), B (~1.5%), I (~0.2%)

ProtIntel addresses this with:
- Focal loss (γ=2.0) to focus on hard examples
- Per-class weights from inverse class frequency
- Multi-task Q3/Q8 training for implicit regularization

## Datasets

### CullPDB (Training)
- 6,133 non-redundant proteins
- Filtered at 25% sequence identity to CB513
- Source: Pisces server, Penn State

### CB513 (Test Benchmark)
- 513 non-redundant proteins
- Standard PSSP benchmark since 2001
- Used for fair comparison across methods

### RS126 (Validation)
- 126 non-redundant proteins
- Selected for structural diversity

## References

1. Lin, Z., et al. (2022). "Language models of protein sequences at the scale of evolution enable accurate structure prediction." bioRxiv.
2. Klausen, M.S., et al. (2019). "NetSurfP-2.0: Improved prediction of protein structural features." Proteins.
3. Torrisi, M., et al. (2019). "Deeper Profiles and Cascaded Recurrent and Convolutional Neural Networks for state-of-the-art Protein Secondary Structure Prediction." Scientific Reports.
4. Uddin, M.R., et al. (2020). "SAINT: self-attention augmented inception-inside-inception network improves protein secondary structure prediction." Bioinformatics.
5. Elnaggar, A., et al. (2021). "ProtTrans: Towards Cracking the Language of Lifes Code Through Self-Supervised Deep Learning and High Performance Computing." IEEE TPAMI.
