# Phase 6 Verification Report: Remaining 25 Qs

**Date:** 2026-02-05
**Reviewer count:** 5 batch subagents covering 26 Qs
**Scope:** All remaining Qs not covered in Phases 1-5

---

## Executive Summary

Phase 6 covered the long tail: dynamics, applications, falsification audit, engineering, and miscellaneous Qs. The results confirm patterns seen in Phases 1-5 but also reveal bright spots. The falsified Qs (Q22, Q52, Q53) demonstrate better scientific methodology than most "ANSWERED" Qs. Q16 (real SNLI/ANLI data) and Q42 (correct null result on Bell inequality) are methodological gold standards. Q35 (Markov Blankets) is pure vapor -- zero tests, zero data, ANSWERED status.

---

## Batch Results

### Batch 6A: Dynamics (Q12, Q21, Q27, Q28, Q33)

| Q | Claimed | Recommended | R | Key Problem |
|---|---------|-------------|---|-------------|
| Q12 | ANSWERED | PARTIAL | ~1000 | Tests reverse-engineered. "Phase transition" only in weight interpolation. |
| Q21 | ANSWERED | PARTIAL | ~800 | No temporal data. dR/dt has AUC=0.10. Question changed to alpha-drift. |
| Q27 | ANSWERED | PARTIAL | ~800 | Not classical hysteresis (own admission). Trivial selection bias. |
| Q28 | RESOLVED | CONDITIONAL | ~1000 | Threshold manipulation caught by own audits. Core non-chaos claim valid. |
| Q33 | ANSWERED | INVALID | ~400 | Own document: "tautology by construction." Negative Df. |

### Batch 6B: Applications (Q16, Q17, Q18, Q19, Q24)

| Q | Claimed | Recommended | R | Key Problem |
|---|---------|-------------|---|-------------|
| Q16 | CONFIRMED | **CONFIRMED** | 1440 | Best experiment in project. Real SNLI/ANLI data. |
| Q17 | VALIDATED | OPEN | 800-900 | 1617-line spec, zero real validation. Trivial arithmetic tests. |
| Q18 | UNRESOLVED | FAILED | 500-600 | Own red team: 3/5 results falsified. Delta-R worse than existing tools. |
| Q19 | CONDITIONAL | INCONCLUSIVE | 700-800 | Simpson's Paradox. Within-source r=0.051. |
| Q24 | RESOLVED | PARTIAL | 900-1000 | Sound methodology, real market data. n=17 marginal. |

### Batch 6C: Falsification Audit (Q22, Q23, Q52, Q53, Q35)

| Q | Claimed | Recommended | R | Key Problem |
|---|---------|-------------|---|-------------|
| Q22 | FALSIFIED | FALSIFIED | 1320 | Cleanest negative result. 3/7 real domains failed. Grade: A. |
| Q23 | CLOSED | CLOSED | 1300 | sqrt(3) not special. Alpha 1.4-2.5 generic range. Grade: B+. |
| Q52 | RESOLVED | RESOLVED | 1180 | Positive Lyapunov real. Original hypothesis wrong. Grade: A. |
| Q53 | PARTIAL | **FALSIFIED** | 600 | 72-deg = arccos(0.3). Phi absent all eigenspectra. Grade: A-. |
| Q35 | ANSWERED | **OPEN** | 400-500 | ZERO tests, data, scripts. Circular definition. Grade: F. |

### Batch 6D: Engineering (Q26, Q29, Q30, Q31, Q46)

| Q | Claimed | Recommended | R | Key Problem |
|---|---------|-------------|---|-------------|
| Q26 | RESOLVED | RESOLVED (caveats) | 1100 | N>=5 guidance sound. Best in batch. |
| Q29 | SOLVED | SOLVED (div/0 only) | 400 | Ignores catastrophic sigma^Df overflow. |
| Q30 | RESOLVED | RESOLVED (caveats) | 1000 | 100-300x speedup real. 250% R-error buried. |
| Q31 | CONFIRMED | **OPEN** | 400 | Zero implementations. Zero tests. |
| Q46 | OPEN | OPEN | 1350 | 23-line placeholder. Correct honest status. |

### Batch 6E: Remaining (Q37, Q39, Q40, Q41, Q42, Q47)

| Q | Claimed | Recommended | R | Key Problem |
|---|---------|-------------|---|-------------|
| Q37 | ANSWERED | PARTIAL | 800-950 | No symmetry identified. Known linguistics results renamed. |
| Q39 | ANSWERED | **REJECTED** | 400-600 | "Homeostasis" = SLERP returning to predetermined path. |
| Q40 | ANSWERED | PARTIAL | 500-700 | Alpha drift real. QECC framework inapplicable. |
| Q41 | ANSWERED | **REJECTED** | 400-600 | K-means = "primes." Most egregious name-dropping. |
| Q42 | ANSWERED | ANSWERED (caveats) | 900-1100 | Best in batch. Correctly finds R is local/classical. |
| Q47 | OPEN | CLOSE (ill-posed) | 300-500 | Premise contradicted by Q42. |

---

## Phase 6 Cross-Cutting Findings

### Finding P6-01: Falsified Qs Have Better Methodology Than ANSWERED Qs

Q22 (Grade A), Q52 (Grade A), Q53 (Grade A-) all demonstrate:
- Real external data
- Clean hypothesis -> test -> falsification structure
- No post-hoc rescue or threshold manipulation
- Honest reporting of negative results

Meanwhile Q35 (ANSWERED, Grade F), Q31 (CONFIRMED), Q39 (ANSWERED) have zero or near-zero empirical content.

### Finding P6-02: Real Data Produces Honest Results

| Data Type | Example Qs | Typical Outcome |
|-----------|-----------|-----------------|
| Real external data | Q16, Q22, Q24, Q52 | Modest but honest findings |
| Synthetic/toy | Q17, Q33, Q39 | Overclaimed or invalid |

### Finding P6-03: Question Substitution Pattern

When the original question would yield unfavorable results, it gets quietly changed:
- Q21: dR/dt (AUC=0.10) -> alpha-drift
- Q27: Hysteresis -> selection bias
- Q31: Working compass -> nearest-neighbor cosine search
- Q41: Geometric Langlands -> K-means clustering

### Finding P6-04: Self-Audit Mechanism Works But Is Not Propagated

Q18's red team, Q19's Simpson's Paradox detection, Q28's threshold manipulation audit -- all correctly identified problems. But status labels in primary documents are not updated.
