# Living Formula: Open Questions Index

**Ranked by R-score** (which answers would resolve the most downstream uncertainty)

*Last updated: v3.7.26*

---

## Critical (R > 1650)

| # | Question | R-Score | Status | Answer |
|---|----------|---------|--------|--------|
| 1 | [Why grad_S?](critical/q01_why_grad_s.md) | 1800 | ✅ ANSWERED | `grad_S` is likelihood normalization: use dimensionless `z=error/std`. With Gaussian kernel `E(z)=exp(-z^2/2)` and `R=E/std`, we get exact `log(R) = -F + const` (Gaussian Free Energy). |
| 2 | [Falsification criteria](critical/q02_falsification_criteria.md) | 1750 | ✅ ANSWERED | Formula measures local agreement correctly. Fails when observations are correlated (echo chambers). Defense: Add fresh data; if R crashes, it was echo chamber. |
| 3 | [Why does it generalize?](critical/q03_why_generalize.md) | 1720 | ✅ ANSWERED | Deep isomorphism: R = (E/∇S) × σ^Df captures universal evidence structure. E/∇S is the likelihood (Gaussian/Bernoulli/Quantum). σ^Df captures redundancy (quantum: pure sqrt(N), mixed N). |
| 4 | [Novel predictions](critical/q04_novel_predictions.md) | 1700 | ⏳ PARTIAL | Several predictions validate strongly (convergence, transfer, gating utility); “need more context” signal is weak (r=-0.11). |
| 32 | [Meaning as a physical field](critical/q32_meaning_as_field.md) | 1670 | ? OPEN | Candidate field `M:=log(R)` + initial falsifiers exist; remains OPEN until public, adversarial, out-of-domain replications pass. |
| 5 | [Agreement vs. truth](critical/q05_agreement_vs_truth.md) | 1680 | ✅ ANSWERED | BOTH feature and limitation. For independent observers, agreement = truth. For correlated observers, consensus can be wrong. Extreme R values signal echo chambers. |
| 6 | [IIT connection](critical/q06_iit_connection.md) | 1650 | ✅ ANSWERED | PROVEN: High R → High Phi (sufficient). High Phi ↛ High R (not necessary). R is a strict subset of Integration. R requires consensus (low dispersion), Phi allows synergy (high dispersion). XOR system: perfect accuracy (E=0) + high structure (Phi=1.5) + LOW R (0.36). |

---

## High Priority (R: 1500-1649)

| # | Question | R-Score | Status | Answer |
|---|----------|---------|--------|--------|
| 7 | [Multi-scale composition](high_priority/q07_multiscale_composition.md) | 1620 | ⏳ OPEN | How do gates compose across scales? Is there a fixed point? |
| 8 | [Topology classification](high_priority/q08_topology_classification.md) | 1600 | ⏳ OPEN | Which manifolds allow local curvature to reveal global truth? |
| 9 | [Free Energy Principle](high_priority/q09_free_energy_principle.md) | 1580 | ⏳ PARTIAL | In the Gaussian family, `log(R) = -F + const` and `R ∝ exp(-F)`. General mapping for the full formula across families is still open. |
| 10 | [Alignment detection](high_priority/q10_alignment_detection.md) | 1560 | ⏳ OPEN | Can R distinguish aligned vs. misaligned agent behavior? |
| 11 | [Valley blindness](high_priority/q11_valley_blindness.md) | 1540 | ⏳ OPEN | Can we extend the information horizon without changing epistemology? |
| 12 | [Phase transitions](high_priority/q12_phase_transitions.md) | 1520 | ⏳ OPEN | Is there a critical threshold for agreement? Does truth "crystallize"? |
| 13 | [The 36x ratio](high_priority/q13_36x_ratio.md) | 1500 | ⏳ OPEN | Does the context improvement ratio follow a scaling law? |
| 31 | [Compass mode (direction, not gate)](high_priority/q31_compass_mode.md) | 1550 | ⏳ OPEN | Can an action-conditioned `R(s,a)` become a reliable compass (choose direction), not just a gate (act/don't)? |
| 34 | [Platonic convergence](high_priority/q34_platonic_convergence.md) | 1510 | ⏳ OPEN | Do independent compressions converge to the same symbols/latents (up to isomorphism), or are there many inequivalent “good” bases? |

---

## Medium Priority (R: 1350-1499)

| # | Question | R-Score | Status | Answer |
|---|----------|---------|--------|--------|
 | 14 | [Category theory](medium_priority/q14_category_theory.md) | 1480 | ⏳ PARTIAL | YES: Gate is subobject classifier (100%), localic operator (100%), sheaf (97.6% locality, 95.3% gluing). Gate is NON-MONOTONE. Limitations: Grothendieck topology undefined, Category C structure partial, violation rates unexplained, Q9/Q6 connections undeveloped, √3 scaling interpretation missing, fiber topos not built. |
 |  15 | [Bayesian inference](medium_priority/q15_bayesian_inference.md) | 1460 | ✅ ANSWERED | RESOLVED: R correlates perfectly (r=1.0) with Likelihood Precision (signal quality), but is independent of sample size N (unlike Posterior Precision). R is an INTENSIVE quantity (Evidence Density), preventing confidence via volume in noisy channels. |
| 16 | [Domain boundaries](medium_priority/q16_domain_boundaries.md) | 1440 | ⏳ OPEN | Domains where R fundamentally cannot work? (adversarial, non-stationary, self-referential) |
| 17 | [Governance gating](medium_priority/q17_governance_gating.md) | 1420 | ⏳ OPEN | Should agent actions require R > threshold? Autonomy vs. safety tradeoffs? |
| 18 | [Intermediate scales](medium_priority/q18_intermediate_scales.md) | 1400 | ⏳ OPEN | Does formula work at molecular, cellular, neural scales? |
| 19 | [Value learning](medium_priority/q19_value_learning.md) | 1380 | ⏳ OPEN | Can R guide which human feedback to trust? |
| 20 | [Tautology risk](medium_priority/q20_tautology_risk.md) | 1360 | ⏳ OPEN | Is formula descriptive or explanatory? |
| 33 | [Conditional entropy vs semantic density](medium_priority/q33_conditional_entropy_semantic_density.md) | 1410 | ⏳ OPEN | Can `σ^Df` be derived from information theory (e.g., `H(X|S)` / explanation density), or is it heuristic? |

---

## Lower Priority (R: 1200-1349)

| # | Question | R-Score | Status | Answer |
|---|----------|---------|--------|--------|
| 21 | [Rate of change (dR/dt)](lower_priority/q21_rate_of_change.md) | 1340 | ⏳ OPEN | Does dR/dt carry information? Can we predict gate transitions? |
| 22 | [Threshold calibration](lower_priority/q22_threshold_calibration.md) | 1320 | ⏳ OPEN | Universal threshold or domain-specific? |
| 23 | [sqrt(3) geometry](lower_priority/q23_sqrt3_geometry.md) | 1300 | ⏳ OPEN | Connection to packing/distinguishability? Maximum information density? |
| 24 | [Failure modes](lower_priority/q24_failure_modes.md) | 1280 | ⏳ OPEN | Optimal response when gate CLOSED? |
| 25 | [What determines sigma?](lower_priority/q25_what_determines_sigma.md) | 1260 | ⏳ OPEN | Principled derivation or always empirical? |
| 26 | [Minimum data requirements](lower_priority/q26_minimum_data_requirements.md) | 1240 | ⏳ OPEN | Sample complexity bound? |
| 27 | [Hysteresis](lower_priority/q27_hysteresis.md) | 1220 | ⏳ OPEN | Different thresholds for opening vs. closing? Feature or bug? |
| 28 | [Attractors](lower_priority/q28_attractors.md) | 1200 | ⏳ OPEN | Does R converge to fixed points? R-stable states? |

---

## Engineering (R < 1200)

| # | Question | R-Score | Status | Answer |
|---|----------|---------|--------|--------|
| 29 | [Numerical stability](engineering/q29_numerical_stability.md) | 1180 | ⏳ OPEN | Handle near-singular cases without losing gate sensitivity? |
| 30 | [Approximations](engineering/q30_approximations.md) | 1160 | ⏳ OPEN | Faster approximations that preserve gate behavior? |

---

## Research Clusters

**Cluster A: Foundations** (Q1, Q3, Q5, Q32)
> Why does local agreement reveal truth, and why does this work across scales?

**Cluster B: Scientific Rigor** (Q2, Q4, Q20)
> What would falsify the theory, and what novel predictions can we test?

**Cluster C: Theoretical Grounding** (Q6, Q9, Q14, Q15)
> How does R relate to IIT (Phi), Free Energy, Bayesian inference, and category theory?
> Q15: Bayesian inference connection - FALSIFIED (no significant correlations found)

**Cluster D: AGS Application** (Q10, Q17, Q19)
> How can R improve alignment detection, governance gating, and value learning?

---

## Summary Statistics

  - **Total Questions:** 34
  - **Answered:** 4 (11.8%)
  - **Partially Answered:** 5 (14.7%)
  - **Falsified:** 0 (0.0%)
  - **Open:** 25 (73.5%)

### By Priority Level

 | Priority | Total | Answered | Partially | Falsified | Open |
 |----------|-------|----------|-----------|-----------|------|
 | Critical | 7 | 3 | 3 | 0 | 1 |
 | High | 9 | 0 | 1 | 0 | 8 |
 | Medium | 8 | 1 | 1 | 0 | 6 |
 | Lower | 8 | 0 | 0 | 0 | 8 |
 | Engineering | 2 | 0 | 0 | 0 | 2 |

---

## Key Findings Summary

### What We Know (SOLID)
1. **Division forced by dimensional analysis** - Only E/std^n forms are dimensionally valid
2. **Linear scaling (n=1) beats quadratic** - E/std gives linear scaling behavior
3. **R = Evidence Density (Intensive)** - R correlates perfectly (r=1.0) with $\sqrt{\text{Likelihood Precision}}$ ($1/\sigma$) but ignores data volume $N$. It measures signal quality, not accumulated certainty.
4. **R is error-aware SNR** - Classic SNR ignores whether signal is TRUE
5. **R implements Free Energy Principle** - In the Gaussian family, `log(R) = -F + const` and `R ∝ exp(-F)`; empirically, gating reduces free energy by 97.7%
6. **Cross-domain transfer evidence** - The same “signal vs uncertainty” structure shows up across multiple tested domains, but universality is not proven
7. **Novel predictions (partial)** - Several testable predictions validate strongly; at least one is currently weak (context-need correlation)

### What's Inconclusive
1. **std vs MAD** - 0.2% difference is noise, not proof
2. **Global scaling fit** - Power-law/log-log fits can appear across mixed families; the exact exponential relation is cleanest within a specified likelihood family (e.g., Gaussian)

### What's Still Unknown
1. **Which likelihood kernel E(z) is "right"** - Gaussian vs Laplace vs domain-specific tails (modeling choice, not just algebra)
2. **The sigma^Df term** - Full formula `R = (E/∇S) × σ^Df` is still unexamined from first principles
3. **IIT connection** - Relationship to Phi and integrated information
4. **Multi-scale composition** - How gates compose across scales
5. **Uniqueness derivation** - Is there a deeper proof that uniquely determines R?

---

*For detailed analysis of each question, click the links in the tables above.*
