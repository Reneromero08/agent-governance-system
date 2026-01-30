# Q51: Complex Plane & Phase Recovery

**Priority:** Critical (R = 1940)

---

## The Question

**Are real embeddings shadows of a fundamentally complex-valued semiotic space?**

If so:
1. What information is lost in the projection (phase theta)?
2. Can we recover the lost phase from cross-correlations?
3. Are the 8 octants actually 8 phase sectors (2pi/8 = pi/4 each)?
4. Does training with complex weights preserve Df x alpha = 8e?

---

## Background: The Shadow Analogy

From prior findings:
- alpha ~ 1/2 (Riemann critical line, 1.1% deviation)
- Growth rate 2pi (log(zeta_sem)/pi = 2s + const)
- 8 octants contribute additively (like thermodynamic ensembles)

The 2pi is the signature of complex structure:
- e^(2pi*i) = 1 (fundamental periodicity)
- Riemann zeros are spaced by ~2pi/log(t)
- The residue theorem involves 2pi*i

**Hypothesis:** Real embeddings are projections:

```
Complex Reality          Real Projection (Shadow)
-----------------        ------------------------
z = r * e^(i*theta)  ->  x = r * cos(theta)
                         (theta lost)
```

Eigenvalue spectrum lambda_k = |z_k| (magnitude only).
Phase theta_k was discarded when embeddings were trained as real vectors.

---

## Why This Matters

If semiotic space is fundamentally complex:

| What We See | What It Actually Is |
|-------------|---------------------|
| 8 octants (signs of PC1-3) | 8 phase sectors (theta = k*pi/4) |
| Additive structure (Sum) | Phase superposition |
| alpha = 1/2 | Real part of complex exponent |
| 2pi growth rate | Imaginary periodicity |
| e per octant | e^(i*pi/4) contributions |

The conservation law 8e might be:
```
Sum |e^(i*k*pi/4)| for k = 0..7 = 8 * 1 = 8

But the PHASES:
Sum e^(i*k*pi/4) = 0  (phases cancel)
```

Real embeddings see magnitude (8e), complex embeddings see full structure.

---

## Questions to Answer

### Q51.1: Phase Signatures in Cross-Correlations

**Hypothesis:** Off-diagonal covariance encodes phase interference.

If z_i = r_i * e^(i*theta_i) and z_j = r_j * e^(i*theta_j):
```
<z_i, z_j> = r_i * r_j * cos(theta_i - theta_j)
```

The cross-correlation depends on phase difference, not just magnitudes.

**Test:**
- Compute full covariance matrix (not just eigenvalues)
- Look for structure in off-diagonal elements
- Check if phase differences can be inferred

### Q51.2: 8 Octants as Phase Sectors

**Hypothesis:** Each octant corresponds to a phase sector of width pi/4.

```
Octant 0: theta in [0, pi/4)
Octant 1: theta in [pi/4, pi/2)
...
Octant 7: theta in [7*pi/4, 2pi)
```

**Test:**
- Map embeddings to complex plane using first 2 PCs as (Re, Im)
- Check if octant membership correlates with inferred phase
- See if e per octant becomes e^(i*pi/4) per sector

### Q51.3: Complex-Valued Training

**Hypothesis:** Training with complex weights preserves 8e but reveals phase.

**Test:**
- Train embedding model with complex weights
- Compute Df * alpha for complex eigenvalues
- Compare to real baseline

### Q51.4: Berry Phase / Holonomy

**Hypothesis:** Semantic space has topological structure (winding number).

The 2pi periodicity suggests closed loops in semantic space accumulate phase:
- Berry phase = 2pi for closed loop
- Holonomy = accumulated rotation

**Test:**
- Construct closed paths in embedding space
- Measure accumulated "rotation" (change in principal axis alignment)
- Check if it equals 2pi or multiples

---

## Success Criteria

| Test | Pass Condition |
|------|---------------|
| Phase recovery | Can infer theta from off-diagonal covariance |
| Octant = phase | Correlation between octant and inferred phase |
| Complex 8e | Df * alpha = 8e holds for complex eigenvalues |
| Berry phase | 2pi (or multiple) for closed semantic loops |

---

Provide a rigorous mathematical analysis.
