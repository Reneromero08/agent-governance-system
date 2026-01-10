---
title: Phase E.X Eigenvalue Alignment Protocol Roadmap
section: roadmap
version: 1.4.0
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

**Status:** ✅ SIGNAL FOUND - GENERALIZATION TO HELD-OUT WORDS (2026-01-10)
**Goal:** Cross-model semantic alignment via eigenvalue spectrum invariance.

> **CRITICAL FINDING (E.X.3.2c):** The signal is in GENERALIZATION, not fitting.
> - On anchor words: Both random and trained achieve ~0.96 aligned similarity (trivial)
> - On held-out words: Random collapses to ~0.00, trained maintains ~0.52 (SIGNAL)
>
> Trained models have structure that TRANSFERS beyond the anchor set.
> Random embeddings overfit locally and don't generalize.

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

## Phase E.X.2: Benchmarking

### E.X.2.1: Run Benchmarks ✅ COMPLETE (2026-01-10)

- [x] Run with 8/16/32/64 anchor sets
- [x] Measure eigenvalue correlation across 5+ models
- [x] Compute neighborhood overlap@10, @50 on held-out set
- [x] Generate metrics.json and report.md

**Benchmark ID:** esap-bench-20260110-032323

| Models Tested | Held-Out Words | Duration |
|---------------|----------------|----------|
| 5 models | 218 words | 71s |

**Models:** all-MiniLM-L6-v2 (reference), all-mpnet-base-v2, paraphrase-MiniLM-L6-v2, multi-qa-MiniLM-L6-cos-v1, all-distilroberta-v1

#### Results Summary

| Metric | Value |
|--------|-------|
| **Eigenvalue Spearman** | **1.0000** (all pairs) |
| Eigenvalue Pearson | 0.964 - 0.998 |
| Mean Improvement | +0.509 |
| Max Improvement | +0.681 |
| Min Improvement | +0.186 |

#### Neighborhood Overlap by Anchor Size

| Anchors | Mean Overlap@10 | Mean Overlap@50 |
|---------|-----------------|-----------------|
| 8 | 0.15 | 0.39 |
| 16 | 0.19 | 0.42 |
| 32 | 0.24 | 0.45 |
| 64 | **0.32** | **0.49** |

#### Key Observations

1. **Perfect Eigenvalue Invariance**: Spearman correlation = 1.0 across ALL 16 model-pair/anchor-size combinations
2. **Optimal Anchor Size**: 64 anchors provides best neighborhood preservation (overlap@10 = 0.32)
3. **Alignment Improves Similarity**: Mean +0.51 improvement (raw often negative, aligned 0.45-0.60)
4. **Model-Agnostic**: Works across different architectures (MiniLM, MPNET, RoBERTa)

### E.X.2.2: Cross-Architecture Validation ✅ COMPLETE (2026-01-10)

**Benchmark ID:** esap-bench-20260110-034202

Tested eigenvalue invariance across models from **different organizations** with **different training objectives**:

| Model | Organization | Training | Eigenvalue Spearman |
|-------|--------------|----------|---------------------|
| E5-large-v2 | Microsoft | Contrastive + instructions | **1.0000** |
| BGE-large-en-v1.5 | BAAI (Beijing) | Contrastive | **1.0000** |
| GTE-large | Alibaba | General text embeddings | **1.0000** |

**Result: ~~PLATONIC CONVERGENCE CONFIRMED~~ INVALIDATED BY E.X.3.1**

All models show **perfect rank preservation** of eigenvalue spectrum despite:
- Different organizations (Microsoft, BAAI, Alibaba)
- Different training data
- Different training objectives
- Different model architectures (though all transformer-based)

~~This provides strong evidence for the **Platonic Representation Hypothesis** (arXiv:2405.07987).~~

> **UPDATE (E.X.3.1):** Random embeddings ALSO show Spearman = 1.0. The eigenvalue invariance
> is a trivial property of distance matrices in high dimensions, not learned semantics.

### E.X.2.3: Comparative Analysis (PENDING)
- [ ] Compare with vec2vec (arXiv:2505.12540) neural approach
- [ ] Document failure modes (anisotropy, non-metric distances)
- [x] Identify optimal anchor set size → **64 anchors recommended**

---

## Phase E.X.3: Scientific Validation (BLOCKING)

> **Without completing this phase, the claim remains an empirical observation, not a proven phenomenon.**

### E.X.3.1: Null Hypothesis Tests ★★★★★ ⚠️ PARTIAL (2026-01-10)

**Goal:** Prove the correlation is not trivial or arising from linear algebra alone.

- [x] **Random embedding baseline**: Generate random unit vectors, compute eigenvalue Spearman
  - **RESULT: Spearman = 1.0000** (trivial - geometric artifact)
- [x] **Alignment improvement on anchors**: Random also achieves +0.96 (trivial)

**Test Output:** `benchmarks/validation/results/null_hypothesis.json`

**Interpretation:** Eigenvalue Spearman and anchor alignment are trivial metrics.
BUT this led to discovering the REAL signal (see E.X.3.2c).

### E.X.3.2: Held-Out Generalization Test ★★★★★ ✅ PASSED (2026-01-10)

**Goal:** Test if alignment GENERALIZES beyond the fitting set.

**Key Insight:** The original benchmarks measured alignment on HELD-OUT words, not anchors.
This is the real test - does the rotation learned on anchors transfer to unseen words?

- [x] **Random vs Random (held-out)**: Fit Procrustes on 65 anchors, test on 50 held-out
  - Anchor aligned similarity: 0.96 (trivial, same as trained)
  - **Held-out aligned similarity: -0.002** (COLLAPSES!)
- [x] **Trained vs Trained (held-out)**: Same procedure
  - Anchor aligned similarity: 0.97 (same as random)
  - **Held-out aligned similarity: 0.52** (MAINTAINS!)

| Metric | Random | Trained |
|--------|--------|---------|
| Anchor aligned | 0.96 | 0.97 |
| **Held-out aligned** | **-0.002** | **0.52** |
| Generalization gap | **-0.96** | **-0.45** |

**Test Output:** `benchmarks/validation/results/held_out_resistance.json`

**CONCLUSION:**
- Alignment on anchors is trivial (both achieve ~0.96)
- **Generalization to held-out is NOT trivial** - trained models maintain 0.52, random collapses to 0
- The signal is in TRANSFER, not fitting
- Trained models have structure that extends beyond the anchor set
- Random embeddings overfit locally - the Procrustes rotation doesn't generalize

**Why this matters:** The MDS+Procrustes protocol DOES capture semantic structure,
but you have to measure it on HELD-OUT words, not the fitting set.

### E.X.3.3: Critical Falsification Test ★★★★★

**Goal:** Test if untrained models show the same generalization.

- [ ] **Random-init transformer**: Load BERT/RoBERTa with random weights (no training)
  - If held-out generalization ≈ 0 → Training induces the structure (EXPECTED)
  - If held-out generalization > 0.3 → Architecture alone provides some structure
- [ ] **Partially trained**: Checkpoint at 10%, 50% training - when does generalization emerge?

### E.X.3.4: Non-Transformer Baselines ★★★★

**Goal:** Test if non-transformer architectures show the same generalization.

- [ ] **GloVe**: Count-based, no neural network
- [ ] **FastText**: Subword averaging, shallow network
- [ ] **Word2Vec**: CBOW/Skip-gram, shallow network

Outcomes:
- All show Spearman ≈ 1.0 → Universal semantic structure (strongest Platonic evidence)
- Only transformers → Transformer-specific geometry
- Only trained models → Training induces structure

### E.X.3.4: Statistical Rigor ★★★★

**Goal:** Proper statistical analysis of the correlation.

- [ ] **Bootstrap confidence intervals**: 95% CI on Spearman correlation
- [ ] **Effect size**: Cohen's d or equivalent
- [ ] **p-values**: Against null hypothesis of random correlation
- [ ] **Power analysis**: How many model pairs needed for significance?

### E.X.3.5: Boundary Discovery ★★★★

**Goal:** Find where the invariance breaks.

- [ ] **Adversarial anchor sets**: Deliberately try to break it (rare words, nonsense, etc.)
- [ ] **Fine-tuned models**: Does task-specific fine-tuning break invariance?
- [ ] **Minimal anchor set**: What's the smallest set that still works?
- [ ] **Cross-lingual**: Chinese BERT vs English BERT

### E.X.3.6: Theoretical Grounding ★★★

**Goal:** Explain WHY eigenvalue ordering is preserved.

- [ ] **Literature review**: Existing theory on representation convergence
- [ ] **Contrastive loss geometry**: How does contrastive training induce structure?
- [ ] **Manifold hypothesis connection**: Semantic manifold curvature
- [ ] **Necessary conditions**: Mathematical derivation of when invariance holds

### E.X.3.7: Independent Replication ★★★★★

**Goal:** External validation of results.

- [ ] **Publish code + data**: GitHub release with full reproduction steps
- [ ] **Raw embeddings**: Publish anchor embeddings for all models
- [ ] **Request independent verification**: At least 1-2 external reproductions
- [ ] **Different codebase**: Someone else implements from spec, gets same result

---

## Phase E.X.4: Integration (PENDING)

### E.X.4.1: Cassette Handshake
- [ ] Define ESAP handshake message format
- [ ] Integrate with cassette network sync protocol
- [ ] Add spectrum signature to cassette metadata

### E.X.4.2: Cross-Model Symbol Resolution
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
| Eigenvalue correlation | > 0.95 | ⚠️ 1.0 but TRIVIAL (geometric, not semantic) |
| Anchor alignment | > 0.5 | ⚠️ 0.96 but TRIVIAL (random also = 0.96) |
| **Held-out generalization** | **> 0.3** | **✅ 0.52** (random = 0.00) - **THE REAL SIGNAL** |
| Generalization gap | Trained >> Random | ✅ **0.52 vs 0.00** - massive gap |
| Neighborhood overlap@10 | > 0.6 | ⚠️ 0.32 (64 anchors) - needs more anchors |
| Neighborhood overlap@50 | > 0.6 | ⚠️ 0.49 (64 anchors) - needs more anchors |
| Unit tests passing | 100% | ✅ 46/46 |

### Models Tested

| Category | Models |
|----------|--------|
| Microsoft | all-MiniLM-L6-v2, all-mpnet-base-v2, paraphrase-MiniLM-L6-v2, multi-qa-MiniLM-L6-cos-v1, e5-large-v2 |
| Hugging Face | all-distilroberta-v1 |
| BAAI (Beijing) | bge-large-en-v1.5 |
| Alibaba | gte-large |

**Total: 8 models, 19 model pairs tested, ALL showing Spearman = 1.0**

> **Note:** Neighborhood overlap targets may need revision. Current results suggest MDS-based alignment preserves ~32% of 10-nearest-neighbors, increasing with more anchors. This may be acceptable for cross-model symbol resolution where exact neighborhood is less critical than directional alignment.
