# Living Formula v3: New Research Questions

**Version:** 3.0
**Date:** 2026-02-05
**Status:** All questions PROPOSED
**Standard:** Inherits v2/METHODOLOGY.md and v2/GLOSSARY.md
**Prerequisite:** v2 Wave 1 (Q1, Q2, Q20) should be resolved before v3 work begins

---

## What v3 Is

v3 contains genuinely new research questions that emerged from the v1 verification process but don't map to any existing Q. These are questions the v1 project should have asked but didn't.

Each v3 question:
- Was identified by the verification as a gap or unexplored direction
- Is not a reformulation of an existing Q (those stay in v2)
- Has pre-registered hypotheses and methodology from the start
- Follows v2/METHODOLOGY.md rules

---

## Questions

### N1: Does E/grad_S Outperform Bare E?

**Motivation:** Q10 found raw E gives 4.33x discrimination vs R's 1.79x for alignment. If dividing by grad_S hurts performance, the entire formula's theoretical apparatus is moot. This is the single most important empirical question about the formula, and it was never directly tested in v1.

**Hypothesis:** R = E/grad_S outperforms bare E on standard NLP benchmarks.

**Test:** Head-to-head comparison on STS-B, SNLI, MNLI, SST-2. Compute both E and R for the same inputs. Compare discrimination power (AUC, correlation with human judgments, etc.).

**Why new:** v1 tested E and R separately. Nobody ran the direct head-to-head.

**Directory:** [n01_e_vs_r](questions/n01_e_vs_r/)

---

### N2: What Does grad_S Actually Encode?

**Motivation:** grad_S = std of pairwise cosine similarities. Is it noise? Information density? Uncertainty? Something else? If we know what grad_S measures, we know whether dividing by it is meaningful.

**Hypothesis:** grad_S correlates with a known, interpretable property of the observation set (e.g., topic diversity, annotation disagreement, embedding space density).

**Test:** Compute grad_S on datasets with known properties (varying topic diversity, varying annotator agreement, varying text quality). Probe what predicts grad_S.

**Why new:** v1 treated grad_S as "entropy gradient" philosophically but never empirically characterized it.

**Directory:** [n02_what_is_grad_s](questions/n02_what_is_grad_s/)

---

### N3: What Determines Sigma Per Domain?

**Motivation:** Q25 falsified sigma universality (varies 15x). But the VARIATION itself is data. If sigma is predictable from domain properties, that's a useful finding.

**Hypothesis:** Sigma is predictable from measurable domain properties (vocabulary size, text diversity, embedding dimensionality, etc.).

**Test:** Measure sigma across 20+ domains. Build regression model predicting sigma from domain features. If R^2 > 0.5, sigma is predictable.

**Why new:** v1 asked "is sigma universal?" (no). v3 asks "what determines sigma?" (nobody checked).

**Directory:** [n03_sigma_determinants](questions/n03_sigma_determinants/)

---

### N4: What Geometric Properties Are Architecture-Invariant?

**Motivation:** Q34 found convergence across models, dismissed as shared training data. But which specific geometric properties converge, and is it training data or something deeper?

**Hypothesis:** Some measurable geometric properties (eigenvalue distribution, intrinsic dimensionality, curvature statistics) are invariant across architectures trained on non-overlapping data.

**Test:** Train (or find) models on non-overlapping corpora. Measure suite of geometric properties. Compare. If properties converge despite different data, there's an architecture-independent invariant.

**Why new:** v1 observed convergence but didn't control for training data overlap.

**Directory:** [n04_architecture_invariants](questions/n04_architecture_invariants/)

---

### N5: What Do Domain-Specific Thresholds Encode?

**Motivation:** Q22 falsified universal threshold but found domain-specific ones exist. What determines the threshold per domain? This turns a falsification into a research program.

**Hypothesis:** Domain-specific R thresholds are predictable from domain properties (vocabulary entropy, mean similarity, embedding density).

**Test:** Measure optimal R threshold across 20+ domains. Correlate with domain features. Build predictor.

**Why new:** Q22 stopped at "universal doesn't exist." The follow-up question was never asked.

**Directory:** [n05_threshold_determinants](questions/n05_threshold_determinants/)

---

### N6: Is The Positive Lyapunov-R Correlation General?

**Motivation:** Q52's "falsification" found R positively correlates with Lyapunov exponent (opposite of predicted). If this holds generally, R measures effective dimensionality of attractors -- a novel finding.

**Hypothesis:** R (participation ratio) positively correlates with Lyapunov exponent across dynamical systems beyond logistic map and Henon attractor.

**Test:** Test on Lorenz attractor, Rossler attractor, Mackey-Glass, real-world chaotic time series (weather, heartbeat, stock prices). Measure R and Lyapunov. Check correlation sign and strength.

**Why new:** Q52 tested 2 systems. Generalization to N systems was never attempted.

**Directory:** [n06_lyapunov_generalization](questions/n06_lyapunov_generalization/)

---

### N7: Can R Detect Real-World Semantic Phenomena?

**Motivation:** Q32 showed good methodology on NLI tasks. Q18 showed deception detection failed. Where exactly is the boundary of R's utility for practical semantic tasks?

**Hypothesis:** R discriminates some real-world semantic phenomena (e.g., sarcasm, misinformation, stance) better than random but the boundary is characterizable.

**Test:** Test on established benchmarks: LIAR (misinformation), iSarcasm (sarcasm), SemEval stance detection, FEVER (fact verification). Compare R to baseline (bare E, random, majority class).

**Why new:** v1 tested scattered tasks. Nobody systematically mapped R's utility boundary.

**Directory:** [n07_utility_boundary](questions/n07_utility_boundary/)

---

## Priority Order

| Priority | Question | Rationale |
|----------|----------|-----------|
| 1 | N1 (E vs R) | If R doesn't beat E, the rest is academic |
| 2 | N2 (What is grad_S) | Understanding the denominator determines formula validity |
| 3 | N7 (Utility boundary) | Practical value assessment |
| 4 | N3 (Sigma determinants) | Explains the parameter instability |
| 5 | N5 (Threshold determinants) | Builds on Q22's falsification |
| 6 | N4 (Architecture invariants) | Deepens convergence finding |
| 7 | N6 (Lyapunov generalization) | Deepens Q52's finding |

---

*v3 initialized: 2026-02-05. All questions PROPOSED.*
