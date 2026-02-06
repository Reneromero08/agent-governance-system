# Q47: Minimal State Representation for R Exists

## Hypothesis
Semantic space exhibits a minimal state representation analogous to the Bloch sphere in quantum mechanics. Specifically: high-dimensional embedding vectors can be faithfully compressed to a low-dimensional state representation that preserves the information needed for R computation and gate decisions. This minimal representation captures the essential geometric structure of the embedding space, and its dimensionality relates to the effective dimensionality (Df) observed in trained models.

## v1 Evidence Summary
- Status: OPEN. No dedicated research was conducted.
- The document is 23 lines long and contains only a placeholder noting relation to Q44 (Born rule) and Q51 (complex plane).
- No hypothesis was formulated, no experiments designed, no tests run, no results generated.

## v1 Methodology Problems
1. **No research conducted**: There is nothing to critique methodologically because no methodology was applied.
2. **Original framing assumes quantum mechanics**: The "Bloch sphere holography" framing requires complex amplitudes and quantum state structure. Embedding vectors are real-valued and high-dimensional, with no natural Bloch sphere mapping.
3. **Quantum interpretation falsified**: Q42 explicitly confirmed R is fundamentally local/classical. The premise that classical embeddings have Bloch sphere structure contradicts the framework's own findings.
4. **Overlap with other questions**: The "holographic" distribution claim is partially addressed by Q40 (noise resistance via PCA redundancy). The dimensional reduction question overlaps with Q43 (effective dimensionality Df = 22).
5. **R-score inflated for no-work question**: R=1350 was assigned despite zero research output.

## v2 Test Plan

### Test 1: Minimal Dimensionality for Gate Preservation
- For embedding models (all-MiniLM-L6-v2 D=384, all-mpnet-base-v2 D=768), compute R from full embeddings on a labeled dataset (SNLI or STS Benchmark).
- Apply dimensionality reduction (PCA, UMAP, random projection) to d = {2, 3, 5, 10, 15, 22, 50, 100, 200} dimensions.
- Recompute R from reduced embeddings.
- Report: gate decision agreement with full-dimensional R at each d.
- Find d_min: the smallest d at which gate agreement exceeds 95%.

### Test 2: Information-Theoretic Lower Bound
- Compute the mutual information I(R_full; R_reduced) as a function of reduction dimensionality d.
- Apply the data processing inequality to establish a lower bound on d_min.
- Compare the empirical d_min from Test 1 against the information-theoretic lower bound.
- Report how close the best reduction method comes to the theoretical limit.

### Test 3: Participation Ratio as State Dimensionality
- Compute the participation ratio (Df) for embeddings from multiple models:
  (a) all-MiniLM-L6-v2 (D=384)
  (b) all-mpnet-base-v2 (D=768)
  (c) BERT-base (D=768)
  (d) GloVe 300d
  (e) Random vectors (negative control)
- Test whether d_min (from Test 1) approximately equals Df.
- Report the ratio d_min / Df for each model.
- If d_min approximately equals Df, this validates Df as the natural state dimensionality.

### Test 4: Geometric Faithfulness of Reduced Representation
- For each reduction method and d, measure preservation of:
  (a) Pairwise cosine similarity rank order (Spearman r)
  (b) k-NN neighborhood overlap (k=10, 50)
  (c) Cluster structure (adjusted Rand index of k-means clusters before and after reduction)
  (d) R values (Pearson and Spearman correlation between full and reduced R)
- Report which geometric properties are preserved at d_min and which require higher dimensionality.

### Test 5: Reduction Method Comparison
- Compare at least 5 dimensionality reduction methods:
  (a) PCA (linear, optimal for variance preservation)
  (b) Random projection (Johnson-Lindenstrauss guarantee)
  (c) UMAP (nonlinear, preserves local structure)
  (d) Autoencoder (learned nonlinear reduction)
  (e) Feature selection (keep top-d PCA components directly)
- For each method, report d_min for 95% gate agreement and the quality metrics from Test 4.
- Determine whether nonlinear methods achieve lower d_min than linear methods.

### Test 6: Stability of Minimal Representation
- Compute d_min on a calibration corpus (e.g., first half of SNLI).
- Apply the same reduction to a held-out corpus (second half of SNLI, STS Benchmark, 20 Newsgroups).
- Report whether d_min transfers across corpora or is corpus-specific.
- Test whether d_min is stable under embedding perturbation (add Gaussian noise at 1%, 5%, 10% levels).

### Test 7: Comparison with Bloch Sphere Geometry (Spirit of Original Question)
- For d_min = 2 or 3 (if achievable), visualize the reduced state space.
- Measure whether the reduced space has spherical geometry (uniform distribution on a sphere) or concentrated geometry (clustered on a manifold).
- If the representation is approximately spherical, report the radius distribution and angular distribution.
- Compare the geometry of the minimal state representation with the Bloch sphere (uniform on S^2 for pure states, inside the ball for mixed states).
- Note: this is exploratory. If d_min >> 3, the Bloch sphere analogy does not apply, and this should be reported honestly.

## Required Data
- **SNLI** (~570K sentence pairs, HuggingFace)
- **STS Benchmark** (~8K pairs, HuggingFace)
- **20 Newsgroups** (~18K documents, sklearn.datasets)
- **GloVe** (6B tokens, 300d, Stanford NLP)
- **Wikipedia random articles** (for corpus transfer test)

## Pre-Registered Criteria
- **Success (confirm):** A minimal state representation of dimensionality d_min << D exists such that R computed from the reduced representation agrees with full-dimensional R on gate decisions >= 95% of the time, AND d_min is consistent (within 2x) across at least 3 embedding models, AND d_min approximately equals Df (ratio d_min/Df between 0.5 and 2.0).
- **Failure (falsify):** No reduction to d < D/2 preserves >= 90% gate agreement, OR d_min varies by more than 5x across models (no universal minimal dimensionality), OR d_min bears no relationship to Df.
- **Inconclusive:** Gate agreement degrades gradually with no clear d_min, or d_min exists but is not related to Df, or results depend heavily on the reduction method.

## Baseline Comparisons
- Random projection at d = Df (Johnson-Lindenstrauss baseline)
- PCA at d = Df (variance-preserving baseline)
- Full-dimensional R (upper bound on accuracy)
- R computed from first d components of raw embedding (no reduction, just truncation)
- Scalar state: just use E or sigma alone (is a 1D state sufficient?)

## Salvageable from v1
- Nothing directly salvageable (no v1 research was conducted).
- Related findings from other questions:
  - Q43: Participation ratio Df = 22.2 for trained BERT, 99.2 for random -- establishes the expected scale of d_min.
  - Q40: "Dark Forest" test showed 94% corruption tolerance with only 3/48 MDS dimensions -- suggests d_min could be very small.
  - Q31: Effective dimensionality measurements showing trained models concentrate in approximately 22 dimensions.
  - These cross-question observations provide concrete hypotheses to test.
