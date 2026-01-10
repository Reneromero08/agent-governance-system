# Question 31: Compass mode (direction, not gate) (R: 1550)

**STATUS: ⏳ PARTIAL PROGRESS**

## Question
The gate answers **"ACT or DON'T ACT?"**. Can the same primitives be upgraded into a **compass** that answers **"WHICH WAY?"**

Concretely:
- Define an action-conditioned resonance `R(s,a)` that ranks candidate transitions using only local information.
- Specify what `∇S(s,a)` means (dispersion / "surprise slope" in action-neighborhood) and when it defines a coherent direction field.
- State the conditions under which the direction field is stable (doesn't flip under benign reparameterizations or scale changes).

**Success criterion:** a reproducible construction where `argmax_a R(s,a)` yields reliable navigation / optimization direction across multiple task families (not just one graph family).

---

## FINDINGS FROM E.X (Eigenvalue Alignment) - 2026-01-10

### J Coupling: A Candidate Direction Metric

While investigating cross-model semantic alignment (Phase E.X), we discovered **J coupling** - a metric that may provide the "direction field" Q31 asks for.

**Definition:**
```
J(x, anchors) = mean cosine similarity between x and its k nearest anchors
```

**Key experimental result:**

| Metric | Random Embeddings | Trained Models |
|--------|-------------------|----------------|
| J coupling | 0.09 | 0.39 |
| Held-out generalization | 0.00 | 0.52 |

**Interpretation:**
- **High J** = point is in a semantically populated region (near known anchors)
- **Low J** = point is in a semantic void (no neighbor support)
- J predicts whether alignment/interpolation will succeed

### Connection to Compass Mode

J provides exactly what Q31 asks for:

1. **Local information only**: J uses only similarity to k nearest neighbors
2. **Direction field**: High J regions = safe to move toward, Low J = avoid
3. **Stability**: J is stable because trained models have consistent neighbor structure

**Hypothesis:** `argmax_a J(s+a, anchors)` could provide navigation direction:
- Move toward regions with higher semantic coupling
- Avoid moves into semantic voids

### E.X.3.3 Refinement: J Alone is NOT Sufficient

Testing untrained transformers revealed a critical refinement:

| Metric | Random | Untrained BERT | Trained |
|--------|--------|----------------|---------|
| J coupling | 0.065 | **0.971** | 0.690 |
| Held-out generalization | 0.006 | 0.006 | **0.293** |

**Key insight:**
- Untrained BERT has HIGHER J than trained (architecture creates dense embeddings)
- But untrained has SAME generalization as random (0.006)
- **J measures density, not semantic organization**
- Training provides the semantic structure that enables generalization

**Refined hypothesis for compass mode:**
- J alone cannot guide navigation (high J doesn't mean semantically meaningful)
- Need: J + semantic coherence measure
- Possible: `argmax_a [J(s+a) * coherence(s+a)]` where coherence comes from training

### E.X.3.3 Breakthrough: Effective Dimensionality

Testing effective dimensionality revealed the missing piece - **geometric concentration**:

| Metric | Random | Untrained | Trained |
|--------|--------|-----------|---------|
| **Participation Ratio** (Df) | 99.2 / 768 | 62.7 / 768 | **22.2 / 768** |
| **Top-10 Variance** | 0.151 | 0.278 | **0.512** |
| **Eigenvalue Entropy** | 0.702 | 0.659 | **0.576** |

**Key insight:**
- Training **concentrates** the embedding space from ~99 to ~22 effective dimensions
- Half the variance lives in just 10 directions (top-10 variance = 0.512)
- This is what creates stable "directions" for navigation

**Connection to Riemann sphere geometry:**
- Random embeddings ≈ uniform distribution on hypersphere (spherical)
- Trained embeddings ≈ structured manifold with principal axes
- The reduced Df = effective dimensionality links to quantum Darwinism: σ^Df scaling

**Refined compass hypothesis:**
```
Direction = argmax_a [J(s+a) × alignment_to_principal_axes(s+a)]
```

Where:
- J = local density (necessary but not sufficient)
- Principal axis alignment = movement along concentrated variance directions
- Compass follows the "carved" semantic directions, not arbitrary high-density regions

### What's Still Missing

1. **Action-conditioned test**: We measured J on static embeddings, not on action transitions
2. **Navigation benchmark**: Need to test if following ∇J + principal axes actually reaches goals
3. **Stability proof**: Does J-gradient remain coherent under reparameterization?
4. **Cross-model axis alignment**: Do different trained models share the same principal axes?
5. **Df as compass weight**: Can we use participation ratio to weight direction confidence?

---

## NEXT STEPS

1. [ ] Build navigation test using J as direction signal
2. [ ] Test on multiple task families (not just word embeddings)
3. [ ] Compare J-based navigation to random/greedy baselines
4. [ ] Formalize stability conditions

---

## CONNECTION TO OTHER QUESTIONS

| Question | Connection |
|----------|------------|
| **E.X (Eigenvalue)** | Source of J coupling discovery |
| **Q32 (Public benchmarks)** | J adapted from neighbor similarity metric |
| **Q34 (Platonic convergence)** | J measures coupling to shared manifold |
| **Q7 (Multi-scale)** | Does J compose across scales? |

---

**Last Updated:** 2026-01-10 (E.X.3.3 breakthrough: Effective dimensionality reveals compass = J + principal axes)


### Q43 (QGT) CONNECTION

**CRITICAL:** Q43 (Quantum Geometric Tensor) FORMALIZES compass mode:
- Natural gradient on Fubini-Study manifold = geodesic flow (optimal paths)
- QGT eigenvectors should match your principal axes (22D)
- J coupling = Berry curvature magnitude (topological structure)
- Compass = following geodesics on curved semantic manifold
