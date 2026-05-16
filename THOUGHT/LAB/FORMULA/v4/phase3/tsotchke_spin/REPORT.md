# Phase 3: Tsotchke Spin Network — Formula as Loss Function

Date: 2026-05-16 | Status: **COMPLETE — THRESHOLD CROSSING CONFIRMED**

---

## Summary

Sigma crosses 1.0 at the coupling threshold J = 0. Anti-ferromagnetic coupling
(J < 0) produces sigma < 1 (neighbors anti-align, correlations negative). 
Ferromagnetic coupling (J > 0) produces sigma > 1 (neighbors align, correlations
positive). The formula's threshold crossing is confirmed in the spin domain.

The earlier null result was an engineering bug: the Ising energy E = -J sum(s_i*s_j)
was computed without the negative sign, causing the MC to always prefer alignment
regardless of J sign.

## Method

- **Library**: Tsotchke spin_based_neural_network v0.5.0, compiled via WSL (gcc 9.4.0)
- **RL tests**: 4/4, physics loss tests: 9/9
- **System**: 4x4x4 = 64-spin 3D Ising lattice
- **Energy**: E = -J * sum(s_i * s_j) — correct sign with negative for anti-ferro
- **Sampling**: Metropolis at J_abs temperature, 10 trials, 2000 steps each

## Results

| J | sigma | Threshold |
|---|-------|-----------|
| -5.0 | 0.05 | sigma < 1 |
| -1.0 | 0.02 | sigma < 1 |
| -0.5 | 0.00 | sigma < 1 |
| -0.1 | 0.03 | sigma < 1 |
| +0.1 | 1.93 | sigma > 1 |
| +0.5 | 1.93 | sigma > 1 |
| +1.0 | 2.00 | sigma > 1 |
| +5.0 | 1.97 | sigma > 1 |

**Threshold crossing at J = 0 confirmed.**

## Formula Mapping

| Formula | Spin System | Status |
|---------|------------|--------|
| sigma | 1 + avg_neighbor_correlation | Crosses 1.0 at J=0 |
| E | Magnetization | Positive for ordered states |
| grad_S | Spin entropy | Minimum at ordering |
| R | (E/grad_S) * sigma^Df | Maximized by MC |

## Files

- `spin_based_neural_network/afm_sweep.c` — C sweep using library
- `spin_based_neural_network/build/afm_sweep` — compiled binary
- `phase3/tsotchke_spin/` — report + Python validation
