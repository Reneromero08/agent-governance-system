# Eigen-Spectrum Alignment Protocol

**Status:** Implementation (Phase E.X.1)
**Discovery:** 2026-01-08
**Goal:** Cross-model semantic alignment via eigenvalue spectrum invariance

---

## Overview

The eigenvalue spectrum of an anchor word distance matrix is **invariant across embedding models** (r = 0.99+), even when raw distance matrices are uncorrelated or inverted.

This discovery enables cross-model semantic alignment without neural network training - using only linear algebra (MDS + Procrustes).

### Key Results

| Model Pair | Raw Distance r | Eigenvalue r |
|------------|----------------|--------------|
| MiniLM ↔ E5-large | -0.05 (inverted!) | **0.9869** |
| MiniLM ↔ MPNET | 0.914 | 0.9954 |
| MiniLM ↔ BGE | 0.277 | 0.9895 |

**Alignment improvement:** Raw similarity -0.005 → Aligned similarity **0.838** (+0.843)

---

## Quick Start

### Installation

```bash
pip install numpy scipy sentence-transformers
```

### CLI Usage

```bash
# Build anchor set from word list
python -m eigen_alignment anchors build anchors.txt --output anchor_set.json

# Compute spectrum signature for a model
python -m eigen_alignment signature compute anchor_set.json --model all-MiniLM-L6-v2

# Fit alignment map between two models
python -m eigen_alignment map fit --source model_a.sig --target model_b.sig --output map.json

# Apply alignment to new points
python -m eigen_alignment map apply --input vectors.npy --map map.json --output aligned.npy
```

### Python API

```python
from eigen_alignment.lib import mds, procrustes, protocol

# Compute anchor embeddings
embeddings = get_embeddings(model, anchors)

# Build distance matrix and MDS coordinates
D2 = mds.squared_distance_matrix(embeddings)
X, eigenvalues, eigenvectors = mds.classical_mds(D2)

# Compute spectrum signature
signature = protocol.spectrum_signature(eigenvalues, k=8)

# Fit Procrustes alignment
R, residual = procrustes.procrustes_align(X_source, X_target)

# Project out-of-sample points
y = procrustes.out_of_sample_mds(d2_to_anchors, D2, eigenvectors, eigenvalues)
y_aligned = y @ R
```

---

## The Method

1. **Compute squared distance matrix D²** for anchor words
2. **Apply classical MDS:** B = -½ J D² J (double-centered Gram matrix)
3. **Eigendecompose:** B = VΛV^T
4. **Get MDS coordinates:** X = V√Λ
5. **Procrustes rotation:** R = argmin ||X₁R - X₂||
6. **Align new points** via Gower's out-of-sample formula

---

## Directory Structure

```
eigen-alignment/
├── README.md                    # This file
├── PROTOCOL_SPEC.md             # Normative protocol specification
├── lib/                         # Library implementation
│   ├── mds.py                   # Classical MDS
│   ├── procrustes.py            # Procrustes alignment
│   ├── protocol.py              # Message types, signatures
│   ├── schemas/                 # JSON schemas
│   └── adapters/                # Model adapters
├── cli/                         # CLI tool
├── benchmarks/                  # Benchmark harness
│   ├── anchor_sets/             # Anchor word lists (8, 16, 32, 64)
│   ├── held_out/                # Evaluation set
│   └── results/                 # Benchmark outputs
├── tests/                       # Test suite
└── receipts/                    # Receipts and reports
```

---

## Protocol Messages

| Message Type | Description |
|--------------|-------------|
| `ANCHOR_SET` | Anchor texts, IDs, and content hash |
| `EMBEDDER_DESCRIPTOR` | Model ID, weights hash, dimension |
| `SPECTRUM_SIGNATURE` | Eigenvalue spectrum (Λ_k) with hash |
| `ALIGNMENT_MAP` | Rotation matrix R with metadata |

---

## Related Work

- arXiv:2405.07987 - Platonic Representation Hypothesis
- arXiv:2505.12540 - vec2vec (neural approach)
- arXiv:2511.21038 - Semantic Anchors in ICL

---

## References

- **Proof of concept:** `../experiments/eigen_alignment_proof.py`
- **Hypothesis document:** `../research/cassette-network/01-08-2026_UNIVERSAL_SEMANTIC_ANCHOR_HYPOTHESIS.md`
- **Execution pack:** `../research/cassette-network/OPUS_EIGEN_SPECTRUM_ALIGNMENT_PROTOCOL_PACK.md`
