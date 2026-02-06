# Phase 4 Verification Report: Conservation Law & Complex Structure

**Date:** 2026-02-05
**Reviewer count:** 5 adversarial skeptic subagents
**Scope:** Q34 (Platonic Convergence), Q48 (Riemann Bridge), Q49 (Why 8e), Q50 (Completing 8e), Q51 (Complex Plane)

---

## Executive Summary

Phase 4 examined the "breakthrough" claims -- the conservation law Df*alpha=8e, the Riemann connection, the complex semiotic space, and phase recovery. **These are the most overclaimed results in the entire project.** Q51 (R=1940) and Q48 (R=1900) received the harshest downgrades. The 8e "conservation law" is numerological (the project's own HONEST_FINAL_STATUS.md agrees at 15% confidence). The complex plane structure is imposed by PCA projection choice, not discovered in the data. The Riemann connection's core hypothesis (GUE statistics) was cleanly falsified.

| Target | Claimed | Recommended | Soundness | Key Problem |
|--------|---------|-------------|-----------|-------------|
| Q34 (Platonic, R=1510) | ANSWERED | PARTIAL | GAPS | Convergence expected from shared training data. R never computed. |
| Q48 (Riemann, R=1900) | BREAKTHROUGH | EXPLORATORY (R~600) | INVALID | GUE hypothesis cleanly falsified. Semantic zeta has no zeta properties. |
| Q49 (Why 8e, R=1880) | BREAKTHROUGH | EXPLORATORY | INVALID | Numerology. Monte Carlo falsification failed (p=0.55). Own docs say 15%. |
| Q50 (Completing 8e, R=1920) | RESOLVED | EXPLORATORY (R~600-800) | CIRCULAR | CV=6.93% is not conservation. e-per-octant tautological. 3-5 independent models, not 24. |
| Q51 (Complex Plane, R=1940) | ANSWERED | REFUTED (R~200) | INVALID | Complex structure imposed by PCA choice. Fails 0/19 on random bases. |

---

## Cross-Cutting Findings

### Finding P4-01: The 8e "Conservation Law" Is Numerology

Three independent lines of evidence converge on this conclusion:
1. **The project's own assessment:** HONEST_FINAL_STATUS.md labels 8e "NUMEROLOGY" at 15% confidence
2. **Monte Carlo falsification:** p=0.5498 -- 55% of random constants match as well or better (Q49)
3. **Competitor constants:** 7pi (0.69% error) and 22 (0.73%) fit nearly as well as 8e (0.43%)
4. **Self-contradiction:** alpha=1/(2*Df) from CP^n geometry implies Df*alpha=0.5, not 21.75
5. **The derivation of 8:** 2^3 from 3 PCA dimensions with binary signs is true for ANY dataset, not specific to semiotics
6. **The derivation of e:** The entropy test for e as independent constant FAILS (H/e=0.70)

### Finding P4-02: The Complex Semiotic Space Does Not Exist

Q51 (R=1940, second highest in project) is the most comprehensively refuted claim:
- Complex structure is imposed by declaring PC1=real, PC2=imaginary -- works for any 2D data
- Try3 Experiment 1: phase arithmetic fails on ALL 50 random bases for ALL 19 models (0/19)
- Try2: zero signature (roots of unity) fails replication: 0/19 models
- Berry phase is zero for real vectors (Q43)
- The genuine observation (word analogy parallelogram) is Mikolov et al. 2013

### Finding P4-03: The Riemann Connection Is Falsified

Q48 (R=1900) contains a clean, honest falsification buried under replacement claims:
- All models match Poisson (random) spacing, NOT GUE (Riemann)
- The semantic zeta function has no functional equation (tested, failed), no Euler product (tested, failed)
- alpha=0.5 coincidence: individual models range 0.462 to 0.840 (68% spread)
- Test code explicitly tries 5 candidate constants and picks the best match

### Finding P4-04: Model "Independence" Is Overstated

Q50 claims 24 independent models, but ~19/24 are transformer encoders on overlapping web corpora. Effective independent architecture classes: 3-5. This inflates all cross-model statistics.

---

## Cumulative R-Score Correction Recommendations

The Phase 4 targets contain the project's highest R-scores and the largest recommended corrections:

| Q | Current R | Recommended R | Delta |
|---|-----------|---------------|-------|
| Q51 | 1940 | ~200 | -1740 |
| Q48 | 1900 | ~600 | -1300 |
| Q50 | 1920 | ~600-800 | -1120 to -1320 |
| Q49 | 1880 | ~600 | -1280 |
| Q34 | 1510 | ~900-1100 | -410 to -610 |

---

## Cumulative Issue Tracker (Phases 1-4)

| ID | Issue | Severity | Phase |
|----|-------|----------|-------|
| P1-01 | 5+ incompatible E definitions | CRITICAL | 1 |
| P1-02 | Axiom 5 embeds the formula | CRITICAL | 1 |
| P2-01 | Theoretical connections are notational relabelings | HIGH | 2 |
| P3-01 | Quantum interpretation falsified by own evidence | CRITICAL | 3 |
| P3-03 | Test fraud: suppress FALSIFIED, relabel ANSWERED | CRITICAL | 3 |
| P3-04 | R numerically unstable, often abandoned for bare E | HIGH | 3 |
| P4-01 | 8e conservation law is numerology (project agrees) | CRITICAL | 4 |
| P4-02 | Complex semiotic space does not exist (0/19 replication) | CRITICAL | 4 |
| P4-03 | Riemann connection cleanly falsified (GUE -> Poisson) | HIGH | 4 |
| P4-04 | Model independence overstated (3-5 classes, not 24) | HIGH | 4 |
