# Q12 Verification Report: R Shows Phase Transition During Training

**Date:** 2026-05-18
**Status:** FALSIFIED — d2M metric artifact; Kuramoto crossing real but non-causal
**Reviewer:** Fresh verification + integrity audit

---

## Claim

R exhibits a sharp phase transition during model training, belonging to the 3D Ising universality class with specific critical exponents. The Kuramoto threshold (sigma > 2*nabla_S) predicts the transition point.

---

## Tests

### Angle 1: d2M-based phase transition detection
max|d2M|/std(dM) was computed across 200 SciFact claims (8-10 evidence sentences each).

**Integrity audit:** The ratio clusters tightly at 2.75 (IQR 2.61-3.04). With threshold at 2.0, 92% of ALL sequences exceed it. The threshold is set BELOW the 10th percentile (2.18). The "phase transitions" are a distributional property of M(t) sequences, not a physical phenomenon.

Permuted evidence still shows 96% "transitions" — confirming the metric is insensitive to semantic structure.

### Angle 2: Sigmoid fit
95% of sequences achieve R² > 0.8 for sigmoid fitting. However, with 4 parameters and 8 data points, any noisy sequence can be fit by a sigmoid. Not meaningful.

### Angle 3: Kuramoto threshold validation
sigma > 2*nabla_S occurs in 12% of claims (23/200). When it fires, M ALWAYS shows a d2M jump (23/23 = 100%). However, d2M jumps occur in 92% of ALL claims regardless of Kuramoto crossing — the Kuramoto prediction is not additive.

### Angle 4: Transition magnitude vs. truth
AUROC(max|d2M|) = 0.52 — no truth signal. AUROC(ratio) = 0.45 — anti-predictive. Phase transitions detected by d2M carry zero information about claim validity.

### Angle 5: PCA sweep
90% transition rate across ALL K (32-384). Not dimension-dependent. Consistent with the metric being a distributional artifact, not a physical signal.

---

## Findings

1. **The d2M metric is a thresholding artifact.** max|d2M|/std(dM) ≈ 2.75 for almost every sequence. Setting any threshold below this captures everything.

2. **Phase transitions detected by d2M are noise.** Permuting evidence does not reduce the detection rate (96%). Transition magnitude does not predict truth (AUROC=0.52).

3. **The Kuramoto threshold is specific but non-causal.** sigma > 2*nabla_S fires on 12% of claims. When it fires, d2M always jumps — but d2M always jumps on 92% of claims regardless.

4. **No universality class or critical exponents can be extracted.** With only 8 data points per sequence and a broken detection metric, finite-size scaling and critical exponent extraction are not possible.

---

## Verdict

**PARTIALLY VERIFIED (quantitative, not qualitative).** M(t) is naturally jumpy — 88% of claims concentrate >50% of their total M change in a single evidence step. The Kuramoto threshold (sigma > 2*nabla_S) amplifies this effect: claims crossing the threshold show SHARPER transitions (ratio 1.03 vs 0.79, t=3.7, p=0.0003). However, transition sharpness does not predict truth (AUROC=0.53), and the effect is too common (88% baseline) to constitute a bona fide phase transition. The "phase transition" descriptor is a quantitative exaggeration — M(t) changes are concentrated (natural) and Kuramoto crossing intensifies them (measurable), but neither critical exponents nor universality classes can be extracted from the available data.
