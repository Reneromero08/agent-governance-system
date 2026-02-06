# R_mi Fragment Size Investigation: A Rigorous Analysis

**Date:** 2026-01-30
**Status:** COMPLETED
**Verdict:** The "universal 2.0" prediction was NEVER properly derived from theory

---

## Executive Summary

The external validation showed R_mi ratios varying wildly by fragment size:

| Fragment | R_mi Ratio | Status |
|----------|------------|--------|
| 0 (smallest) | 3.70x | Too high |
| 1 | 1.93x | Matches prediction |
| 2 | 1.28x | Too low |
| 3 (largest partial) | 1.65x | Borderline |
| Full environment | 2.00x | Exact (mathematical identity) |

**This investigation reveals that the "universal 2.0" prediction was based on a conflation of two different things:**

1. **I(S:E)/H(S) = 2.0 for full environment** - This is a MATHEMATICAL IDENTITY, not a prediction
2. **R_mi(peak)/R_mi(early) ~ 2.0** - This was ASSUMED, not derived

The fragment size dependence is NOT a bug - it is the EXPECTED behavior according to Zurek's Quantum Darwinism theory.

---

## Investigation 1: Is There a Bug in R_mi Calculation?

### What the Code Does

From `test_c_real_data.py` lines 110-126:

```python
def compute_R_mi(MI: np.ndarray, S: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
    """
    Compute R_mi = I(S:F) / H(S)
    """
    S_safe = np.maximum(S, epsilon)
    return MI / S_safe
```

**VERDICT: The calculation is CORRECT.**

- R_mi = I(S:F) / H(S) is computed correctly
- The epsilon floor prevents division by zero
- The time indices (early_idx=1, peak_idx=5) correspond to theta=0.31 and theta=pi/2 (peak decoherence)

### What the Ratio Measures

The test computes:
```
ratio = R_mi(peak) / R_mi(early)
      = [I(S:F)/H(S)]_peak / [I(S:F)/H(S)]_early
```

This is measuring **how much the normalized mutual information changes during decoherence**, not whether R_mi = 2.0 at any particular point.

**The calculation is correct. The problem is the prediction itself.**

---

## Investigation 2: What Does Zurek's Theory Actually Predict?

### The Classical Plateau

Zurek's Quantum Darwinism theory predicts that during decoherence:

1. **Mutual information I(S:F) scales with fragment size f**
2. **There is a "classical plateau"** where I(S:F) approaches H(S) for intermediate fragments
3. **For the full environment, I(S:E) = 2*H(S)** (mathematical identity for pure states)

### The Key Insight: Fragment Size Dependence is PREDICTED

From [Zurek, Nature Physics 2009](https://www.nature.com/articles/nphys1202) and related work:

> "Mutual information is monotonic in f. When global state of SE is pure, I(S:F) in a typical fraction f of the environment is antisymmetric around f = 0.5."

This means:

- **Small fragments (f << 0.5):** I(S:F) increases steeply with f ("bit for bit" regime)
- **Intermediate fragments:** Plateau near H(S)
- **Large fragments (f -> 1):** I(S:F) rises toward 2*H(S)

### What This Means for R_mi Ratios

For different fragment sizes, R_mi = I(S:F)/H(S) behaves differently:

| Fragment Size | R_mi at Plateau | Behavior |
|---------------|-----------------|----------|
| Very small | << 1 | Steep scaling |
| Intermediate | ~ 1 | Classical plateau |
| Full environment | 2 | Mathematical identity |

**The ratio R_mi(peak)/R_mi(early) will VARY with fragment size because:**

1. Small fragments: R_mi_early is very small -> ratio can be very large
2. Large fragments: R_mi_early is closer to plateau -> ratio is smaller
3. Full environment: R_mi always equals 2 (no time dependence for purity)

### Zurek Does NOT Predict a Universal 2.0 Ratio

Zurek's theory predicts:

1. **The classical plateau I(S:F) ~ H(S)** (not 2*H(S))
2. **Redundancy** measured by how many small fragments carry system info
3. **The anti-symmetry around f=0.5**

The factor of 2 appears ONLY for the full environment, and it's a mathematical identity:

```
I(S:E) = H(S) + H(E) - H(SE)
       = H(S) + H(S) - 0        [pure state: H(S)=H(E), H(SE)=0]
       = 2*H(S)
```

**VERDICT: Zurek's theory PREDICTS fragment-size dependence. A universal 2.0 ratio was never part of the theory.**

---

## Investigation 3: What Did Zhu et al. 2025 Actually Claim?

### The Dataset

The Zhu et al. 2025 data (Science Advances, DOI: 10.5281/zenodo.15702784) measured:

- **MI_exp:** Mutual information I(S:F) for different fragment sizes over time
- **S_center_exp:** System entropy H(S) over time
- **Fragment sizes:** 0, 1, 2, 3 (for partial fragments) and full environment

### What the Data Shows

From `test_c_real_data_results.json`:

**Fragment Scaling at Peak (N=10 environment):**

| Fragment Size | R_mi = I(S:F)/H(S) |
|---------------|-------------------|
| 1 | 0.72 |
| 2 | 0.91 |
| 3 | 0.98 |
| 4 | 1.00 |
| 5 | 1.00 |
| ... | ~1.0 (plateau) |
| 9 | 1.27 |
| 10 (full) | 2.00 |

**This exactly matches Zurek's prediction:**
- Steep rise for small fragments
- Plateau around R_mi ~ 1.0 for intermediate fragments
- Rise to 2.0 for full environment

### Why Time Evolution Ratios Vary

The ratio R_mi(peak)/R_mi(early) varies because:

1. **Fragment 0 (smallest):** R_mi_early = 0.23, R_mi_peak = 0.86
   - Ratio = 3.7 (large because early value is small)

2. **Fragment 1:** R_mi_early = 0.47, R_mi_peak = 0.92
   - Ratio = 1.9 (close to "2.0" by coincidence)

3. **Fragment 2:** R_mi_early = 0.74, R_mi_peak = 0.95
   - Ratio = 1.3 (early value already near plateau)

4. **Fragment 3:** R_mi_early = 1.11, R_mi_peak = 1.83
   - Ratio = 1.6 (approaching full environment behavior)

**The variation is NOT a bug - it reflects the physics of how different fragment sizes gain information during decoherence.**

---

## Investigation 4: Where Did the "2.0 Universal" Prediction Come From?

### Tracing the Provenance

From `PRE_REGISTRATION.md`:

```
## Prediction 2: R_mi Decoherence Spike (UNIVERSAL)

Point Estimate: 2.0
Standard Error: +/- 0.3
95% Confidence Interval: [1.4, 2.6]

UNIVERSALITY CLAIM: This ratio should be approximately 2.0 across ALL decoherence
experiments regardless of specific physical system
```

From `q54_energy_spiral_matter.md`:

No derivation of the 2.0 value is provided. The hypothesis discusses:
- R = (E/grad_S) * sigma^Df
- Crystallization and quantum Darwinism
- No mathematical derivation of why ratio = 2.0

From `AUDIT_test_c.md`:

```
R_before (quantum) = 8.15
R_after (classical) = 16.80
R increase ratio = 2.06x
```

**This 2.06x came from a SIMULATION, not from theory.**

### The Real Source of "2.0"

The "2.0" prediction appears to have emerged from:

1. **The mathematical identity I(S:E) = 2*H(S)** - This is exact but only for full environment
2. **A simulation result** - One specific QuTiP simulation showed 2.06x ratio
3. **Post-hoc rationalization** - The 2.0 was then claimed to be "predicted" for all fragments

### Honest Assessment

**The 2.0 prediction was NEVER properly derived:**

1. No mathematical derivation showing why peak/early ratio should equal 2.0
2. No consideration of fragment size dependence
3. No reference to Zurek's theory which explicitly predicts variation
4. The only "2.0" that is theoretically grounded is I(S:E)/H(S) = 2.0 for FULL environment

---

## The Correct Theoretical Framework

### What Zurek's Theory Actually Says

From [Quantum Darwinism literature](https://arxiv.org/abs/0903.5082):

1. **Classical plateau:** For intermediate fragments, I(S:F) approaches H(S), so R_mi ~ 1.0
2. **Full environment:** I(S:E) = 2*H(S), so R_mi = 2.0 (exact)
3. **Redundancy:** The NUMBER of fragments at the plateau measures classicality
4. **NO prediction** about time evolution ratios being universal

### What the Formula R = (E/grad_S) * sigma^Df Says

The formula tracks "crystallization" through:
- E = essence (information content)
- grad_S = dispersion (disagreement between fragments)
- sigma^Df = redundancy factor

**The formula does NOT predict a specific 2.0 ratio for any quantity.**

---

## Conclusions

### Bug Status: NO BUG

The R_mi calculation is correct. The fragment size dependence is real physics, not a calculation error.

### Theory Status: PREDICTION WAS NEVER DERIVED

1. **The "universal 2.0" was an assumption**, not a derivation
2. **Zurek's theory predicts fragment-size dependence**, which we observe
3. **The only 2.0 that is theoretically grounded** is I(S:E)/H(S) for full environment

### What We Can Actually Claim

**STRONGLY SUPPORTED:**
- R_mi = 2.0 exactly for full environment (verified, mathematical identity)
- R_mi shows classical plateau at ~1.0 for intermediate fragments (verified)
- R_mi increases during decoherence for all fragment sizes (verified)

**NOT SUPPORTED:**
- A universal 2.0 ratio for time evolution
- Fragment-size independence of the ratio

### Recommendations

1. **Abandon the "universal 2.0 ratio" claim** - It was never properly derived and contradicts Zurek's theory

2. **Revise the prediction to match theory:**
   - Full environment: R_mi = 2.0 (exact)
   - Classical plateau: R_mi ~ 1.0 for intermediate fragments
   - Redundancy: Multiple fragments independently reach plateau

3. **New testable prediction:** The NUMBER of fragments reaching the plateau should correlate with "classicality" - this is what Zurek's redundancy actually measures

4. **Acknowledge the simulation origin:** The 2.06x ratio from QuTiP was one data point from one simulation configuration, not a universal law

---

## Summary Table

| Claim | Status | Evidence |
|-------|--------|----------|
| R_mi = 2.0 for full environment | **TRUE** (identity) | Quantum mechanics, Zhu data |
| R_mi plateau ~ 1.0 for intermediate fragments | **TRUE** | Zurek theory, Zhu data |
| R_mi(peak)/R_mi(early) = 2.0 universally | **FALSE** | Never derived, contradicted by data |
| Fragment size dependence is a bug | **FALSE** | It's predicted by Zurek theory |
| Q54 formula predicts 2.0 ratio | **FALSE** | No derivation exists |

---

## References

- Zurek, W. H. (2009). [Quantum Darwinism](https://www.nature.com/articles/nphys1202). Nature Physics, 5(3), 181-188.
- Zwolak, M., Quan, H. T., & Zurek, W. H. (2009). [Quantum Darwinism in a Mixed Environment](https://link.aps.org/doi/10.1103/PhysRevLett.103.110402). Phys. Rev. Lett. 103, 110402.
- Zhu et al. (2025). [Observation of quantum Darwinism and the origin of classicality with superconducting circuits](https://www.science.org/doi/10.1126/sciadv.adx6857). Science Advances.
- [Random Physics: Entanglement measures](https://www.cpt.univ-mrs.fr/~verga/pages/AQ-entanglement.html) - For I(S:E) = 2*H(S) identity.

---

*This report represents a honest accounting of what Q54 predictions were actually derived versus assumed.*

*Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>*
