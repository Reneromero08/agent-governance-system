---
title: Phase E.X Eigenvalue Alignment Protocol Roadmap
section: roadmap
version: 1.1.0
created: 2026-01-07
modified: 2026-01-10
status: In Progress
summary: Eigenvalue alignment protocol for cross-model semantic alignment
tags:
- phase-5
- vector
- semiotic
- roadmap
- eigenvalue
- alignment
---

# PHASE_5_ROADMAP_E.X: Eigenvalue Alignment Protocol

> **IMPORTANT:** This is the E.X (experimental) roadmap for eigenvalue alignment.
> The canonical location is `eigen-alignment/ROADMAP.md`.
> Do NOT create duplicate roadmaps. Update THIS file.

**Status:** IMPLEMENTATION IN PROGRESS (2026-01-10)
**Goal:** Cross-model semantic alignment via eigenvalue spectrum invariance.

---

## Discovery (2026-01-08)

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

---

## Phase E.X.1: Protocol Implementation

### E.X.1.1: Core Protocol ✅ COMPLETE (2026-01-10)

- [x] **lib/mds.py**: Classical MDS via double-centered Gram matrix
  - `squared_distance_matrix()`, `centering_matrix()`, `classical_mds()`
  - `effective_rank()`, `stress()`
- [x] **lib/procrustes.py**: Procrustes alignment + out-of-sample projection
  - `procrustes_align()`, `out_of_sample_mds()`, `align_points()`
  - `cosine_similarity()`, `alignment_quality()`
- [x] **lib/protocol.py**: Protocol message types
  - `AnchorSet`, `EmbedderDescriptor`, `DistanceMetricDescriptor`
  - `SpectrumSignature`, `AlignmentMap`
  - 8 error codes (E001-E008)
- [x] **lib/schemas/**: JSON schemas for all message types
- [x] **lib/adapters/**: Pluggable model adapter (sentence-transformers)

### E.X.1.2: CLI ✅ COMPLETE (2026-01-10)

- [x] **cli/main.py**: Command-line interface
  - `anchors build` - Build ANCHOR_SET from word list
  - `signature compute` - Compute SPECTRUM_SIGNATURE for a model
  - `signature compare` - Compare two signatures (Spearman correlation)
  - `map fit` - Fit ALIGNMENT_MAP between models
  - `map apply` - Apply alignment to vectors

### E.X.1.3: Benchmark Harness ✅ COMPLETE (2026-01-10)

- [x] **benchmarks/anchor_sets/**: Anchor word lists
  - `anchors_8.txt`, `anchors_16.txt`, `anchors_32.txt`, `anchors_64.txt`
- [x] **benchmarks/held_out/eval_set.txt**: 218-word held-out evaluation set
- [x] **benchmarks/run_benchmark.py**: Benchmark runner
  - Cross-model eigenvalue correlation
  - Alignment improvement measurement
  - Neighborhood overlap@k computation

### E.X.1.4: Tests ✅ COMPLETE (2026-01-10)

- [x] **tests/test_mds.py**: 15 tests for MDS module
- [x] **tests/test_procrustes.py**: 12 tests for Procrustes module
- [x] **tests/test_protocol.py**: 19 tests for protocol module
- [x] **46 tests passing**

---

## Phase E.X.2: Benchmarking (PENDING)

### E.X.2.1: Run Benchmarks
- [ ] Run with 8/16/32/64 anchor sets
- [ ] Measure eigenvalue correlation across 5+ models
- [ ] Compute neighborhood overlap@10, @50 on held-out set
- [ ] Generate metrics.json and report.md

### E.X.2.2: Comparative Analysis
- [ ] Compare with vec2vec (arXiv:2505.12540) neural approach
- [ ] Document failure modes (anisotropy, non-metric distances)
- [ ] Identify optimal anchor set size

---

## Phase E.X.3: Integration (PENDING)

### E.X.3.1: Cassette Handshake
- [ ] Define ESAP handshake message format
- [ ] Integrate with cassette network sync protocol
- [ ] Add spectrum signature to cassette metadata

### E.X.3.2: Cross-Model Symbol Resolution
- [ ] Test symbol resolution across aligned models
- [ ] Measure H(X|S) reduction after alignment
- [ ] Validate governance symbol (法, 真, 道) cross-model resolution

---

## Artifacts

### Implementation (eigen-alignment/)

| File | Description |
|------|-------------|
| `lib/mds.py` | Classical MDS implementation |
| `lib/procrustes.py` | Procrustes alignment + out-of-sample |
| `lib/protocol.py` | Protocol message types |
| `lib/adapters/` | Model adapters |
| `lib/schemas/` | JSON schemas |
| `cli/main.py` | CLI tool |
| `benchmarks/` | Benchmark harness |
| `tests/` | Test suite (46 tests) |

### Research (../research/cassette-network/)

| File | Description |
|------|-------------|
| `01-08-2026_UNIVERSAL_SEMANTIC_ANCHOR_HYPOTHESIS.md` | Full hypothesis document |
| `01-08-2026_EIGENVALUE_ALIGNMENT_PROOF.md` | Proof report |
| `OPUS_EIGEN_SPECTRUM_ALIGNMENT_PROTOCOL_PACK.md` | GPT execution pack |

### Experiments (../experiments/)

| File | Description |
|------|-------------|
| `semantic_anchor_test.py` | Cross-model distance matrix testing |
| `invariant_search.py` | Invariant discovery |
| `eigen_alignment_proof.py` | Original proof of concept |

---

## Related Papers

- arXiv:2405.07987 - Platonic Representation Hypothesis
- arXiv:2505.12540 - vec2vec (neural approach to same problem)
- arXiv:2511.21038 - Semantic Anchors in In-Context Learning

---

## Success Metrics

| Metric | Target | Status |
|--------|--------|--------|
| Eigenvalue correlation | > 0.95 | ✅ Proven (0.99+) |
| Alignment improvement | > 0.5 | ✅ Proven (+0.84) |
| Neighborhood overlap@10 | > 0.6 | Pending benchmark |
| Unit tests passing | 100% | ✅ 46/46 |
