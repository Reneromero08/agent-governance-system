# Test B Methodology: Non-Circular Phase Lock Test

## Problem Statement

The original Test B (in `test_b_nist_data.py`) used a **circular proxy**:

```
Binding Energy:    E_n ~ 1/n^2  (Rydberg formula)
Phase Lock Proxy:  PL  ~ 1/n^2  (or Z^2/n^2)
```

Correlating these gives r = 1.0 by mathematical construction. This is a **tautology**, not a test.

## Solution: Independent Measures

The new test (`test_b_noncircular.py`) uses measures that are **not derived from the energy formula**:

### 1. Radiative Lifetimes (tau)

**Scaling**: tau ~ n^3 (NOT n^2)

**Why Independent**:
- Lifetimes come from the Einstein A coefficient: tau = 1/A_ki
- A_ki depends on the **dipole matrix element**: |<psi_k|r|psi_i>|^2
- Computing this requires the **full wavefunction**, not just the energy
- The wavefunction shape determines transition probability

**Data Source**: NIST ASD, Wiese & Fuhr JPCRD 38(4) 2009

| State | n | l | Lifetime (ns) | Binding Energy (eV) |
|-------|---|---|---------------|---------------------|
| 2p | 2 | 1 | 1.596 | 3.40 |
| 3p | 3 | 1 | 5.27 | 1.51 |
| 4p | 4 | 1 | 12.4 | 0.85 |
| 5p | 5 | 1 | 24.0 | 0.54 |
| 6p | 6 | 1 | 41.0 | 0.38 |

### 2. Oscillator Strengths (f)

**Scaling**: f_1s->np ~ n^(-3) (NOT n^(-2))

**Why Independent**:
- f = (2m*omega/(3*hbar*e^2)) * |<i|r|k>|^2
- Depends on wavefunction overlap integrals
- Not derived from energy eigenvalues

**Data Source**: NIST ASD

| Transition | f-value | A_ki (s^-1) |
|------------|---------|-------------|
| 1s->2p | 0.4162 | 6.26e8 |
| 1s->3p | 0.0791 | 1.67e8 |
| 1s->4p | 0.0290 | 6.82e7 |
| 1s->5p | 0.0139 | 3.44e7 |
| 1s->6p | 0.0078 | 1.97e7 |

### 3. Selection Rule Test

**Critical Observation**: States with **SAME n** have **SAME binding energy** but **DIFFERENT lifetimes**.

Example (n=3):
- 3s: tau ~ 158 ns
- 3p: tau ~ 5.3 ns
- 3d: tau ~ 15.5 ns

This proves that **binding energy alone does not determine stability**. Angular momentum structure matters.

## Expected Correlations

### If Hypothesis is Correct:
- Binding energy should correlate with SOME stability measure
- Correlation may be **negative** (more bound = faster decay)
- This is physically sensible: tighter binding = stronger radiation coupling

### If Hypothesis is Wrong:
- No significant correlation (|r| < 0.3)
- Random scatter between binding energy and lifetime

## Test Interpretation

| Correlation | Interpretation |
|-------------|----------------|
| r > 0.7 (positive) | Supports: more binding = more stable |
| r < -0.7 (negative) | Supports: binding affects stability (inverse) |
| |r| < 0.3 | **FAILS**: binding does not predict stability |

## Why This Test Can Fail

1. **Lifetime scales as n^3**: Different power law than energy (n^-2)
2. **Oscillator strength ~ n^-3**: Also different scaling
3. **Selection rules**: Same-energy states have different lifetimes

If binding energy and stability were just "both functions of n", we'd expect consistent power-law relationships. The actual physics is more complex.

## Data Sources

1. **NIST Atomic Spectra Database**: https://physics.nist.gov/asd
2. **Wiese, W.L. and Fuhr, J.R.** "Accurate Atomic Transition Probabilities for Hydrogen, Helium, and Lithium" - J. Phys. Chem. Ref. Data, Vol. 38, No. 4, 2009
3. **Physics LibreTexts**: Hydrogen transition calculations

## Comparison to Old Test

| Aspect | Old Test (Circular) | New Test (Non-Circular) |
|--------|---------------------|-------------------------|
| Phase Lock Proxy | 1/n^2 | tau, f, A_ki |
| Independence | NO (E ~ 1/n^2) | YES (tau ~ n^3) |
| Expected r | 1.0 (tautology) | Unknown (real test) |
| Can Fail | NO | YES |
| Scientific Value | None | Real hypothesis test |

## Conclusion

The non-circular test provides a genuine scientific test of the "phase lock leads to stability" hypothesis. By using radiative lifetimes and oscillator strengths - which depend on wavefunction structure, not just energy eigenvalues - we can determine whether binding energy truly correlates with state stability in a non-trivial way.
