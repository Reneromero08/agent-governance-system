# Lissajous Hypothesis — Final Report

Date: 2026-05-14 | Status: **FALSIFIED at three operationalization depths**

---

## Summary

The hypothesis that sigma is a function of frequency rationality between
stabilizer oscillators was tested at three levels of analysis. All three
produced null or tautological results.

## Phase 1: Aggregate Syndrome Density Ratios

**Operationalization:** Frequency ratios between distance levels computed from
syndrome density (fraction of detectors firing).

**Result:** NULL (r=-0.21, R2=0.04). Syndrome density ratios are nearly constant
(~1.05-1.18) across all p values. They don't vary enough to distinguish
sigma=6.9 from sigma=0.7.

## Phase 2: DEM Eigenvalue Spectra

**Operationalization:** Detector error model correlation matrices, eigenvalue
decomposition, dominant mode ratios between distance levels.

**Result:** NULL (r=+0.12, R2=0.01). The eigenvalue spectrum scales linearly
with p — doubling p doubles all eigenvalues proportionally. Frequency ratios
between distance levels are structural constants of the code geometry (matrix
dimensions 24, 120, 336), not functions of p.

**Key finding:** The DEM contains 15 multi-target error mechanisms for d=3
even with standard depolarizing noise (bug was in DemTarget method calls —
`is_separator()` not `is_separator`). But these correlations don't produce
p-dependent frequency ratios.

## Phase 3: Detection-Event Trajectories

**Operationalization:** 10,000-shot detection event samples, joint
distribution of the most strongly correlated detector pair, Lissajous
figure analysis.

**Result:** TAUTOLOGICAL. Three metrics correlate with sigma: diagonal
ratio (r=+0.75), detector correlation coefficient (r=-0.64), and joint
determinant (r=-0.65). But the "figure" is a 2x2 joint probability table
with 4 discrete states. Each shot is independent — no temporal trajectory,
no curve, no phase relationship. The diagonal ratio captures the same
information as `1 - syndrome_density`. The correlation is expected: higher
p means more coincident detector firings and lower sigma.

## Why the Hypothesis Failed

The Lissajous mechanism requires:
1. Coupled oscillators with identifiable frequency ratios
2. Phase relationships that produce closed vs. open trajectories
3. A rationality measure that varies with the error rate

The surface code's stabilizer network under standard Pauli noise provides
none of these at the detection-event level. Detectors fire independently
per shot (even when correlated through multi-target error mechanisms).
There are no temporal phase relationships to analyze. The "frequency ratios"
are structural constants of the code geometry, not dynamic variables.

## What Would Be Needed

A true test of the Lissajous hypothesis would require:
- Correlated noise with temporal structure (not per-shot independence)
- Phase-space trajectories from time-series detector data
- A mechanism where stabilizer frequencies lock into rational ratios below
  threshold and break above it

Standard circuit-level depolarizing noise — even with added crosstalk — does
not produce this behavior. The hypothesis is not viable with current QEC
simulation infrastructure.

## Files

- `lissajous/PHASE1_REPORT.md` — aggregate frequency ratios
- `lissajous/phase1_frequencies.py` — syndrome density analysis
- `lissajous/phase3_dem.py` — DEM eigenvalue analysis
- `lissajous/correlated/scout.py` — correlated noise scout
- `lissajous/phase4/phase4_trajectories.py` — detection-event trajectories
- `lissajous/phase4/phase4_results.json` — closure metrics
