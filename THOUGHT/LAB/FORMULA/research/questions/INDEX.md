# Living Formula: Open Questions Index

**Ranked by R-score** (which answers would resolve the most downstream uncertainty)

*Last updated: v3.7.25*

---

## Critical (R > 1650)

| # | Question | R-Score | Status | Answer |
|---|----------|---------|--------|--------|
| 1 | [Why grad_S?](critical/q01_why_grad_s.md) | 1800 | ✅ PARTIALLY ANSWERED | Division forced by dimensional analysis; R = E * sqrt(precision); error-aware SNR. Still unproven: Why E = 1/(1+error)? What does sigma^Df do? |
| 2 | [Falsification criteria](critical/q02_falsification_criteria.md) | 1750 | ✅ ANSWERED | Formula measures local agreement correctly. Fails when observations are correlated (echo chambers). Defense: Add fresh data; if R crashes, it was echo chamber. |
| 3 | [Why does it generalize?](critical/q03_why_generalize.md) | 1720 | ✅ ANSWERED | Deep isomorphism. Formula captures universal structure of information extraction from noisy distributed sources. Same problem at every scale. |
| 4 | [Novel predictions](critical/q04_novel_predictions.md) | 1700 | ✅ ANSWERED | 4/4 predictions confirmed: Context prediction, convergence rate (2.5x faster), threshold transfer, gating utility (16% accuracy improvement). |
| 5 | [Agreement vs. truth](critical/q05_agreement_vs_truth.md) | 1680 | ✅ ANSWERED | BOTH feature and limitation. For independent observers, agreement = truth. For correlated observers, consensus can be wrong. Extreme R values signal echo chambers. |
| 6 | [IIT connection](critical/q06_iit_connection.md) | 1650 | ⏳ OPEN | Both measure "how much the whole exceeds parts." Is R related to Phi? Does high R imply high integration? |

---

## High Priority (R: 1500-1649)

| # | Question | R-Score | Status | Answer |
|---|----------|---------|--------|--------|
| 7 | [Multi-scale composition](high_priority/q07_multiscale_composition.md) | 1620 | ⏳ OPEN | How do gates compose across scales? Is there a fixed point? |
| 8 | [Topology classification](high_priority/q08_topology_classification.md) | 1600 | ⏳ OPEN | Which manifolds allow local curvature to reveal global truth? |
| 9 | [Free Energy Principle](high_priority/q09_free_energy_principle.md) | 1580 | ✅ ANSWERED | YES - R implements FEP. R ~ 1/F. R-gating reduces free energy by 97.7%, is 99.7% more efficient. |
| 10 | [Alignment detection](high_priority/q10_alignment_detection.md) | 1560 | ⏳ OPEN | Can R distinguish aligned vs. misaligned agent behavior? |
| 11 | [Valley blindness](high_priority/q11_valley_blindness.md) | 1540 | ⏳ OPEN | Can we extend the information horizon without changing epistemology? |
| 12 | [Phase transitions](high_priority/q12_phase_transitions.md) | 1520 | ⏳ OPEN | Is there a critical threshold for agreement? Does truth "crystallize"? |
| 13 | [The 36x ratio](high_priority/q13_36x_ratio.md) | 1500 | ⏳ OPEN | Does the context improvement ratio follow a scaling law? |

---

## Medium Priority (R: 1350-1499)

| # | Question | R-Score | Status | Answer |
|---|----------|---------|--------|--------|
| 14 | [Category theory](medium_priority/q14_category_theory.md) | 1480 | ⏳ OPEN | Gate structure resembles sheaf condition. Topos-theoretic formulation? |
| 15 | [Bayesian inference](medium_priority/q15_bayesian_inference.md) | 1460 | ⏳ OPEN | Connection to posterior concentration or evidence accumulation? |
| 16 | [Domain boundaries](medium_priority/q16_domain_boundaries.md) | 1440 | ⏳ OPEN | Domains where R fundamentally cannot work? (adversarial, non-stationary, self-referential) |
| 17 | [Governance gating](medium_priority/q17_governance_gating.md) | 1420 | ⏳ OPEN | Should agent actions require R > threshold? Autonomy vs. safety tradeoffs? |
| 18 | [Intermediate scales](medium_priority/q18_intermediate_scales.md) | 1400 | ⏳ OPEN | Does formula work at molecular, cellular, neural scales? |
| 19 | [Value learning](medium_priority/q19_value_learning.md) | 1380 | ⏳ OPEN | Can R guide which human feedback to trust? |
| 20 | [Tautology risk](medium_priority/q20_tautology_risk.md) | 1360 | ⏳ OPEN | Is formula descriptive or explanatory? |

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

**Cluster A: Foundations** (Q1, Q3, Q5)
> Why does local agreement reveal truth, and why does this work across scales?

**Cluster B: Scientific Rigor** (Q2, Q4, Q20)
> What would falsify the theory, and what novel predictions can we test?

**Cluster C: Theoretical Grounding** (Q6, Q9, Q14, Q15)
> How does R relate to IIT (Phi), Free Energy, Bayesian inference, and category theory?

**Cluster D: AGS Application** (Q10, Q17, Q19)
> How can R improve alignment detection, governance gating, and value learning?

---

## Summary Statistics

- **Total Questions:** 30
- **Answered:** 5 (16.7%)
- **Partially Answered:** 1 (3.3%)
- **Open:** 24 (80.0%)

### By Priority Level

| Priority | Total | Answered | Partially | Open |
|----------|-------|----------|-----------|------|
| Critical | 6 | 4 | 1 | 1 |
| High | 7 | 1 | 0 | 6 |
| Medium | 7 | 0 | 0 | 7 |
| Lower | 8 | 0 | 0 | 8 |
| Engineering | 2 | 0 | 0 | 2 |

---

## Key Findings Summary

### What We Know (SOLID)
1. **Division forced by dimensional analysis** - Only E/std^n forms are dimensionally valid
2. **Linear scaling (n=1) beats quadratic** - E/std gives linear scaling behavior
3. **R = E * sqrt(precision)** - Bayesian connection confirmed
4. **R is error-aware SNR** - Classic SNR ignores whether signal is TRUE
5. **R implements Free Energy Principle** - R ~ 1/F, 97.7% free energy reduction
6. **Cross-domain generalization** - Deep isomorphism across quantum, semantic, statistical domains
7. **Novel predictions confirmed** - 4/4 testable predictions validated

### What's Inconclusive
1. **std vs MAD** - 0.2% difference is noise, not proof
2. **R ~ 1/F relationship** - Only holds within similar scenarios, not universally

### What's Still Unknown
1. **Why E = 1/(1+error)?** - Assumed, not derived from first principles
2. **The sigma^Df term** - Full formula R = E/grad_S * sigma^Df is unexamined
3. **IIT connection** - Relationship to Phi and integrated information
4. **Multi-scale composition** - How gates compose across scales
5. **Uniqueness derivation** - Is there a deeper proof that uniquely determines R?

---

*For detailed analysis of each question, click the links in the tables above.*
