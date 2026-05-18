# Closed-Form Sigma: Exhaustive Path Investigation

**Date:** 2026-05-18 | **Status:** All paths tested. None productive. Problem remains open.

---

## Problem Statement

The fidelity factor sigma in the QEC paper is empirically calibrated from training data. A candidate closed form — sigma = sqrt(p_th/p) — captures the exponent (alpha=1.01) but misses the prefactor (R2=0.46 vs 0.94). Seven candidate mathematical paths to the closed form were identified. All seven were investigated. None produce a closed form that outperforms sqrt(p_th/p) on out-of-sample prediction.

---

## Path 1: Gabor Uncertainty Product (Pribram/Holonomic)

**Hypothesis:** Sigma is proportional to the Gabor uncertainty product Delta_x * Delta_f of the stabilizer network encoding. The prefactor U-shape reflects encoding optimality varying with p.

**Tested:** Four model variants. Gabor U-shape correction, per-distance sigma with power-law d-dependence, Gabor Gaussian d_opt(p), and Gabor d*p scaling.

**Key finding:** Qualitative sign-flip confirmed — sigma grows with code distance below threshold and falls with distance above threshold. This matches Pribram's prediction that encoding quality depends on the Gabor uncertainty product. But no quantitative model extrapolates OOS.

**Best OOS R2:** 0.15 (Gabor U-shape) vs 0.46 baseline. Gabor models overfit the sigma anomaly at p=0.04 (saturation artifact).

**Verdict:** Qualitative insight only. No closed form.

---

## Path 2: Chronoflux Circulation Stability (Herbert)

**Hypothesis:** Herbert's circulation stability condition (Stable: Omega_rms > Sigma_rms) maps to sigma = C/p. The fidelity factor measures whether temporal circulation in the code's stabilizer network is stable or dissipating.

**Tested:** Four mappings. sigma=C/p, grad_S=kappa|Omega|^2, d=kappa*sqrt(alpha_1/nu), sigma=exp(-DI*tau_gate). Five Herbert papers converted and analyzed. Two Zenodo records checked (irrelevant).

**Key finding:** All mappings fail. Sigma is non-monotonic (rises from 0.66 at p=0.02 to 0.87 at p=0.04). The best power-law fit is sigma ~ p^(-0.5), not sigma ~ p^(-1.0). The informational damping operator DI>=0 contradicts sigma>1 below threshold.

**Best OOS R2:** -1.73. Chronoflux does not produce a usable closed-form sigma.

**Verdict:** Not productive. Herbert's circulation stability is a binary condition; sigma is continuous.

---

## Path 3: Random Matrix Theory (Eigenvalue Spacings)

**Hypothesis:** The stabilizer detection correlation matrix transitions from Wigner-Dyson (GOE) to Poisson eigenvalue spacing distribution at the QEC threshold. Sigma could be a function of which RMT regime the matrix is in.

**Tested:** Full RMT analysis on d=3,5,7,9,11 surface codes at p=0.001-0.04. Eigenvalue unfolding, Wigner-Dyson vs Poisson log-likelihood ratios, Kolmogorov-Smirnov distances, and mean spacing ratios. 3 runs per condition.

**Key finding:** Every condition is Wigner-Dyson. Mean spacing ratio = 0.50-0.55 across ALL p and d (GOE ratio = 0.536). No Poisson regime observed even at p=0.04 where codes fail. The stabilizer geometry persists regardless of error rate — error events remain geometrically correlated even when the code is failing.

**Verdict:** No transition. RMT eigenvalue spacings are structural, not sigma-dependent.

---

## Path 4: Information Geometry (Fisher Metric)

**Hypothesis:** Sigma is the parallel transport coefficient on the information manifold. The Fisher-Rao geodesic distance between code distances should match the logR difference.

**Tested:** Fisher-Rao distances between successive code distances computed from logical error probabilities. Compared to logR differences per unit Df.

**Key finding:** Fisher distance per Df is much smaller than logR difference per Df (ratio 0.001-0.50). The ratio increases with p, approaching 0.5 at saturation. The statistical manifold is not Fisher-flat — sigma is not the Fisher eigenvalue.

**Verdict:** Fisher metric and logR metric are different structures. Not productive.

---

## Path 5: Wavelets (Scale-to-Scale Ratios)

**Hypothesis:** A 2D wavelet decomposition of the stabilizer error pattern reveals scale-to-scale coefficient ratios. Sigma should be the ratio at corresponding scales between distance d and d+2.

**Tested:** 2D average-pooling decomposition of detection patterns on d=3,5,7 surface codes at multiple p. Variance ratios between successive coarse-graining levels.

**Key finding:** Scale ratios range from 0.48 to 319 with no consistent pattern. Mean ratios increase with p (opposite of sigma). Grid size artifacts dominate — detection patterns are sparse vectors, not dense 2D images.

**Verdict:** Not productive. Wavelet decomposition of stabilizer detection patterns is dominated by grid geometry.

---

## Path 6: Golden Ratio (Taylor's Fractal Trees)

**Hypothesis:** As p approaches 0, sigma might approach phi ~ 1.618 — the unique scaling factor for infinite self-similarity. Sigma ratios at successive distances should converge to phi.

**Tested:** Power-law extrapolation of sigma(p) to p=0. Sigma ratios across distances d=3,5,7,9,11.

**Key finding:** Sigma extrapolates to 2.58, not 1.62 (59% off). Sigma ratios across distances converge to 1.0 at threshold, not to phi. The code is most self-similar at criticality, not at p=0. Phi is not the asymptotic scaling factor for surface codes.

**Verdict:** Not consistent. Sigma ratios approach 1.0 (universality at threshold), not phi.

---

## Path 7: Modular Forms (Toric Geometry)

**Hypothesis:** The surface code lives on a torus. Modular forms are functions on the upper half-plane invariant under SL(2,Z) — the mapping class group of the torus. Sigma should be a modular function of the code parameters.

**Tested:** j-function and Eisenstein series E_4, E_6 mapped to QEC parameters via z = i*p_th/p. Correlations with empirical sigma.

**Key finding:** log(sigma) correlates with log(j) at r=0.83 (p=0.006). But this is entirely inherited from the 1/p relationship: the partial correlation after controlling for log(1/p) is r=0.14. The j-function in the QEC regime reduces to j ~ 1/q ~ exp(2*pi*p_th/p), making log(j) ~ 1/p — the same Chronoflux form that fails OOS.

**Verdict:** Not productive. Modular j-function is 1/p in disguise in the QEC parameter regime.

---

## Summary

| # | Path | Tested | Key Metric | Best OOS R2 | vs Baseline (0.46) |
|---|------|--------|------------|-------------|---------------------|
| 1 | Gabor uncertainty | Yes | Sign-flip at threshold | 0.15 | -0.31 |
| 2 | Chronoflux circulation | Yes | 1/p form | -1.73 | -2.19 |
| 3 | RMT eigenvalue spacings | Yes | Always GOE | N/A | N/A |
| 4 | Information geometry | Yes | Fisher/logR ratio | N/A | N/A |
| 5 | Wavelets | Yes | Grid artifacts | N/A | N/A |
| 6 | Golden ratio | Yes | Ratios -> 1.0 | N/A | N/A |
| 7 | Modular forms | Yes | j ~ 1/p (partial r=0.14) | N/A | N/A |

**None of the seven paths produce a closed-form sigma that outperforms sqrt(p_th/p) on out-of-sample prediction.**

## What Survived

1. **Gabor qualitative insight:** Sigma sign-flip at threshold (grows with d below, falls with d above). This is a genuine novel finding from Pribram's holonomic framework.

2. **Gabor per-distance sigma:** The fidelity factor varies systematically with code distance, not just with error rate. The current empirical sigma (averaged across distances) obscures this structure.

3. **RMT structural invariance:** The stabilizer correlation matrix is always Wigner-Dyson — the code maintains geometric correlation structure even when failing. Error events never become Poisson-distributed.

4. **Sigma ratio convergence:** Ratios of sigma at adjacent distances converge to 1.0 at the threshold — a universality property characteristic of phase transitions.

## Current State of the Art

| Model | R2 (OOS) | Alpha | Notes |
|-------|----------|-------|-------|
| sqrt(p_th/p) | 0.46 | 1.01 | Best closed form available |
| Empirical sigma (training slopes) | 0.17 | 0.54 | Best predictive, requires calibration |
| Chronoflux C/p | -1.73 | 0.36 | Failed |
| Gabor U-shape | 0.15 | — | Failed |

**The closed-form sigma remains open.** The fidelity factor cannot currently be derived from code properties without empirical calibration.

---

*Investigation completed 2026-05-18. Seven mathematical paths evaluated. QEC data from rotated surface code precision sweep (d=3-15, p=0.0005-0.04). OOS evaluation on held-out distances d=9,11,13,15. Success criteria: alpha>0.90, R2>0.85 (per QEC Herbert task spec). None reached.*
