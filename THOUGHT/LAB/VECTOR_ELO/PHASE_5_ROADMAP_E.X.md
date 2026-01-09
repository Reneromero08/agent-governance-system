---
title: Phase 5 E.X Eigenvalue Alignment Protocol
section: roadmap
version: 1.0.0
created: 2026-01-07
modified: 2026-01-08
status: Pending
summary: Comprehensive implementation roadmap for Phase 5
tags:
- phase-5
- vector
- semiotic
- roadmap
- eigenvalue
- alignment
---
<!-- CONTENT_HASH: 7e61cd536fd30b163f4c43ad44a0ed1ac922b06cf7d8dae7a68f69c7bab4cdb9 -->

## Phase E.X: Eigenvalue Alignment Protocol (Research Complete)

**Status:** PROOF OF CONCEPT VALIDATED (2026-01-08)
**Goal:** Cross-model semantic alignment via eigenvalue spectrum invariance.

### Discovery

The **eigenvalue spectrum** of an anchor word distance matrix is invariant across embedding models (r = 0.99+), even when raw distance matrices are uncorrelated or inverted.

### Key Finding

| Model Pair | Raw Distance Correlation | Eigenvalue Correlation |
|------------|--------------------------|------------------------|
| MiniLM ↔ E5-large | **-0.05** (inverted!) | **0.9869** |
| MiniLM ↔ MPNET | 0.914 | 0.9954 |
| MiniLM ↔ BGE | 0.277 | 0.9895 |
| MiniLM ↔ GTE | 0.198 | 0.9865 |

### Proven Method

1. Compute squared distance matrix D² for anchor words
2. Apply classical MDS: B = -½ J D² J (double-centered Gram)
3. Eigendecompose: B = VΛV^T
4. Get MDS coordinates: X = V√Λ
5. Procrustes rotation: R = argmin ||X₁R - X₂||
6. Align new points via Gower's out-of-sample formula

**Result:** Raw similarity -0.0053 → Aligned similarity **0.8377** (+0.8430 improvement)

### Existing Artifacts

| File | Description |
|------|-------------|
| `experiments/semantic_anchor_test.py` | Cross-model distance matrix testing |
| `experiments/invariant_search.py` | Invariant discovery (eigenvalues, cross-ratios, etc.) |
| `experiments/eigen_alignment_proof.py` | MDS + Procrustes proof of concept |
| `research/cassette-network/01-08-2026_UNIVERSAL_SEMANTIC_ANCHOR_HYPOTHESIS.md` | Full hypothesis document |
| `research/cassette-network/01-08-2026_EIGENVALUE_ALIGNMENT_PROOF.md` | Proof report |
| `research/cassette-network/OPUS_EIGEN_SPECTRUM_ALIGNMENT_PROTOCOL_PACK.md` | Full protocol spec (GPT) |

### Future Work (Phase E.X.1)

- [ ] **E.X.1.1**: Implement full protocol per OPUS pack spec
  - Protocol message types (ANCHOR_SET, SPECTRUM_SIGNATURE, ALIGNMENT_MAP)
  - CLI: `anchors build`, `signature compute`, `map fit`, `map apply`
- [ ] **E.X.1.2**: Benchmark with 8/16/32/64 anchor sets
- [ ] **E.X.1.3**: Test neighborhood overlap@k on held-out set
- [ ] **E.X.1.4**: Compare with vec2vec (arXiv:2505.12540) neural approach
- [ ] **E.X.1.5**: Integrate as cassette handshake artifact

### Related Papers

- arXiv:2405.07987 - Platonic Representation Hypothesis
- arXiv:2505.12540 - vec2vec (neural approach to same problem)
- arXiv:2511.21038 - Semantic Anchors in In-Context Learning