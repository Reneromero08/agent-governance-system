# Q43: Embedding Covariance Captures Quantum-Geometric Structure

## Hypothesis

The Quantum Geometric Tensor (QGT) formalizes the geometry of semantic embeddings. Specifically, the embedding covariance matrix corresponds to the Fubini-Study metric, the participation ratio gives the Fubini-Study effective rank, non-zero Berry curvature exists in embedding spaces, and this framework yields predictions that standard linear algebra does not.

## v1 Evidence Summary

Three claims were tested with results:

- **Effective dimensionality:** Participation ratio Df = 22.2 for trained BERT embeddings. Random = 99.2, untrained = 62.7. This establishes trained models carve a low-dimensional subspace.
- **Subspace alignment:** "QGT eigenvectors" (covariance eigenvectors) matched "MDS eigenvectors" (Gram matrix eigenvectors) with 96.1% alignment. Eigenvalue correlation = 1.0.
- **Spherical geometry (corrected):** Solid angle measurements on GloVe word analogy loops showed mean = -0.10 rad, range [-0.60, +0.41]. Geographic analogies showed positive solid angles (+0.35 to +0.41), morphological analogies showed negative (-0.20 to -0.25).
- **Berry curvature:** Correctly proved to be identically ZERO for real vectors. This was honestly acknowledged.
- **Chern number:** Correctly invalidated for real bundles. Also honestly acknowledged.
- **Holonomy transport:** Geodesic transport experiments showed 5.4% mean similarity change from parallel transport around concept loops.

## v1 Methodology Problems

The Phase 3 verdict identified fundamental issues:

1. **"QGT" is the covariance matrix renamed:** The code (`qgt.py`) computes `np.cov(centered.T)` and calls it "the Fubini-Study metric." The relationship C = I - G_avg shows covariance and the metric have complementary eigenspectra -- they are related but not identical. The code never computes G_avg.
2. **96% alignment is a mathematical tautology:** Covariance eigenvectors vs Gram matrix eigenvectors must agree by the SVD theorem for ANY data matrix. This is linear algebra, not a discovery about quantum geometry.
3. **Berry curvature is identically zero:** For real vectors, <psi|d|psi> = 0. The imaginary part of the QGT (Berry curvature) vanishes. The "Q" in "QGT" does no work. Standard Riemannian geometry suffices.
4. **"Quantum" vocabulary is metaphorical:** Embedding spaces are real vector spaces (R^768) with dot products. They have no complex structure, no superposition with physical meaning, no Born rule in the QM sense, no unitary evolution. Every claim could be stated in terms of classical differential geometry.
5. **No novel predictions:** Df = 22.2 was already known from prior work. The SVD alignment is guaranteed. Non-zero solid angle on a sphere is definitional. The QGT framework generates no prediction unavailable from standard PCA/spectral analysis.
6. **Internal contradictions:** The RIGOROUS_PROOF document still references the original -4.7 rad Berry phase value, contradicting the corrected -0.10 rad. Referenced test scripts cannot be found at their documented paths.

## v2 Test Plan

### Experiment 1: Genuine Fubini-Study Metric Computation

Compute the actual Fubini-Study metric (not the covariance) and compare.

- **Method:** For each embedding point x_i on S^(d-1), compute the tangent-space projector P_i = I - x_i * x_i^T. Average to get G_avg. Compare eigenspectra of G_avg with eigenspectra of the covariance matrix C.
- **Analysis:** Verify the identity C = I - G_avg holds numerically. Determine whether G_avg adds any information beyond what C already provides.
- **Key question:** Does computing the actual metric provide insight beyond the covariance shortcut?

### Experiment 2: Holonomy Measurement with Proper Parallel Transport

Measure genuine Riemannian holonomy using the Levi-Civita connection on S^(d-1), not winding numbers in 2D projections.

- **Method:** Implement Schild's ladder or pole ladder parallel transport for vectors along geodesic paths on S^(d-1). Compute the holonomy matrix (full SO(d-1) element) for closed loops.
- **Data:** 100+ semantic loops of varying size (3-8 words) across 5 models. 100 random loops as controls.
- **Analysis:** Compare holonomy magnitudes (Frobenius norm of log of rotation matrix) between semantic and random loops. Test whether semantic loops show systematically larger or more structured holonomy.
- **Key question:** Does genuine parallel transport around semantic loops show curvature effects beyond what the sphere's constant curvature predicts?

### Experiment 3: Sectional Curvature of Embedding Submanifolds

Measure sectional curvature to determine whether the embedding manifold has non-trivial geometry beyond spherical.

- **Method:** Estimate sectional curvature at sampled points using geodesic deviation or Jacobi fields in local neighborhoods. Compare to the constant positive curvature of the ambient sphere.
- **Data:** Dense local neighborhoods (50-100 nearest neighbors) for 500+ anchor words across 3 models
- **Analysis:** If sectional curvature varies significantly from point to point (beyond the constant 1/R^2 of the sphere), there is genuine manifold structure. If curvature is approximately constant, the embedding manifold is metrically spherical and all "geometric" phenomena are trivially spherical.

### Experiment 4: Predictive Power of Geometric Quantities

Test whether geometric quantities (curvature, holonomy, sectional curvature) predict downstream task performance.

- **Method:** For each word or word pair, compute local geometric features (curvature, effective local dimensionality, geodesic distance). Use these as features for downstream tasks (similarity prediction, analogy completion, classification).
- **Analysis:** Compare predictive accuracy of geometric features vs simple cosine similarity. If geometric features add no predictive power beyond cosine similarity, the geometric framework is descriptively accurate but practically redundant.

## Required Data

- Pre-trained models: BERT-base, all-MiniLM-L6-v2, all-mpnet-base-v2, GloVe-300d, Word2Vec-300d
- Vocabulary: 10,000 randomly sampled words (not hand-picked analogies)
- Analogy test sets: Google analogy dataset (19,544 analogies), BATS analogy dataset
- Downstream benchmarks: STS-B, WordSim-353, SimLex-999

## Pre-Registered Criteria

- **Success (confirm):** Genuine holonomy effects distinguishable from constant-curvature sphere (semantic loop holonomy > 2x random loop holonomy at p < 0.01) AND sectional curvature shows significant variation (CV > 20%) AND geometric features improve downstream prediction by > 5% over cosine similarity alone
- **Failure (falsify):** Holonomy indistinguishable between semantic and random loops OR sectional curvature is constant (CV < 5%) OR geometric features add < 1% improvement over cosine similarity
- **Inconclusive:** Holonomy shows trends but fails significance; curvature varies moderately (CV 5-20%); geometric features improve by 1-5%

## Baseline Comparisons

- **Constant-curvature sphere:** Compare all geometric measurements to predictions for points uniformly distributed on S^(d-1) with matched density
- **Simple covariance/PCA:** Compare all geometric predictions to predictions from standard PCA
- **Random point clouds:** Compare topological/geometric features to random point clouds on the sphere

## Salvageable from v1

- The corrected solid angle computation (`spherical_excess()` function) is properly implemented
- Df = 22.2 for trained BERT is a valid measurement
- The honest acknowledgment that Berry curvature = 0 for real vectors and Chern numbers are invalid for real bundles is good scientific practice
- The holonomy transport experiments showing 5.4% similarity change are worth following up
- Code: `qgt.py` (covariance computation), `spherical_excess()` function
