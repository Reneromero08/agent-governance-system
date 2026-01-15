# Q51: Complex Plane & Phase Recovery

**Status:** OPEN
**Priority:** Critical (R = 1940)
**Dependencies:** Q48-Q50 (Semiotic Conservation Law)
**Date:** 2026-01-15

---

## The Question

**Are real embeddings shadows of a fundamentally complex-valued semiotic space?**

If so:
1. What information is lost in the projection (phase θ)?
2. Can we recover the lost phase from cross-correlations?
3. Are the 8 octants actually 8 phase sectors (2π/8 = π/4 each)?
4. Does training with complex weights preserve Df × α = 8e?

---

## Background: The Shadow Analogy

From Q48-Q50, we found:
- **α ≈ 1/2** (Riemann critical line, 1.1% deviation)
- **Growth rate 2π** (log(ζ_sem)/π = 2s + const)
- **8 octants** contribute additively (like thermodynamic ensembles)

The 2π is the signature of **complex structure**:
- e^(2πi) = 1 (fundamental periodicity)
- Riemann zeros are spaced by ~2π/log(t)
- The residue theorem involves 2πi

**Hypothesis:** Real embeddings are projections:

```
Complex Reality          Real Projection (Shadow)
─────────────────        ────────────────────────
z = r × e^(iθ)    →      x = r × cos(θ)
                         (θ lost)
```

Eigenvalue spectrum λ_k = |z_k| (magnitude only).
Phase θ_k was discarded when embeddings were trained as real vectors.

---

## Why This Matters

If semiotic space is fundamentally complex:

| What We See | What It Actually Is |
|-------------|---------------------|
| 8 octants (signs of PC1-3) | 8 phase sectors (θ = kπ/4) |
| Additive structure (Σ) | Phase superposition |
| α = 1/2 | Real part of complex exponent |
| 2π growth rate | Imaginary periodicity |
| e per octant | e^(iπ/4) contributions |

The conservation law 8e might be:
```
Σ |e^(ikπ/4)| for k = 0..7 = 8 × 1 = 8

But the PHASES:
Σ e^(ikπ/4) = 0  (phases cancel)
```

Real embeddings see magnitude (8e), complex embeddings see full structure.

---

## Questions to Answer

### Q51.1: Phase Signatures in Cross-Correlations

**Hypothesis:** Off-diagonal covariance encodes phase interference.

If z_i = r_i × e^(iθ_i) and z_j = r_j × e^(iθ_j):
```
⟨z_i, z_j⟩ = r_i × r_j × cos(θ_i - θ_j)
```

The cross-correlation depends on phase difference, not just magnitudes.

**Test:**
- Compute full covariance matrix (not just eigenvalues)
- Look for structure in off-diagonal elements
- Check if phase differences can be inferred

### Q51.2: 8 Octants as Phase Sectors

**Hypothesis:** Each octant corresponds to a phase sector of width π/4.

```
Octant 0: θ ∈ [0, π/4)
Octant 1: θ ∈ [π/4, π/2)
...
Octant 7: θ ∈ [7π/4, 2π)
```

**Test:**
- Map embeddings to complex plane using first 2 PCs as (Re, Im)
- Check if octant membership correlates with inferred phase
- See if e per octant becomes e^(iπ/4) per sector

### Q51.3: Complex-Valued Training

**Hypothesis:** Training with complex weights preserves 8e but reveals phase.

**Test:**
- Train embedding model with complex weights
- Compute Df × α for complex eigenvalues
- Compare to real baseline

### Q51.4: Berry Phase / Holonomy

**Hypothesis:** Semantic space has topological structure (winding number).

The 2π periodicity suggests closed loops in semantic space accumulate phase:
- Berry phase = 2π for closed loop
- Holonomy = accumulated rotation

**Test:**
- Construct closed paths in embedding space
- Measure accumulated "rotation" (change in principal axis alignment)
- Check if it equals 2π or multiples

---

## Connection to Prior Work

| Q50 Finding | Q51 Interpretation |
|-------------|-------------------|
| α ≈ 1/2 | Real part of complex critical exponent |
| 2π growth | Imaginary periodicity (hidden phase) |
| 8 octants | 8 phase sectors |
| Additive structure | Phase superposition |
| No Euler product | Phases cancel in product |

---

## Files to Create

| File | Purpose |
|------|---------|
| `test_q51_phase_signatures.py` | Cross-correlation analysis |
| `test_q51_octant_phases.py` | Map octants to phase sectors |
| `test_q51_complex_training.py` | Complex-valued embedding test |
| `test_q51_berry_phase.py` | Topological winding number |

---

## Success Criteria

| Test | Pass Condition |
|------|---------------|
| Phase recovery | Can infer θ from off-diagonal covariance |
| Octant = phase | Correlation between octant and inferred phase |
| Complex 8e | Df × α = 8e holds for complex eigenvalues |
| Berry phase | 2π (or multiple) for closed semantic loops |

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Phase is truly lost (not recoverable) | Document as negative result |
| Complex training is computationally expensive | Use small models first |
| Octant-phase mapping is arbitrary | Test multiple mappings |
| Berry phase = 0 (real vectors have no phase) | Use holonomy/solid angle instead |

---

*Created: 2026-01-15*
*Status: OPEN - Awaiting investigation*
