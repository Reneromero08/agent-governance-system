# Q15 v3 Verdict: Does R Correlate with Statistical Quality Measures?

**Version**: v3 (honest reframing)
**Date**: 2026-02-06
**Status**: FALSIFIED

---

## Reframing Note

v2 dressed frequentist statistics in Bayesian labels (see AUDIT.md, METH-1).
v3 drops all Bayesian claims and asks the honest question:

> Does R = (E / grad_S) * sigma^Df correlate with standard statistical
> measures of cluster quality better than E alone?

All measures tested are frequentist. No Bayesian inference is performed.

---

## Pre-Registered Criteria

**CONFIRMED** if R significantly outperforms E (Steiger p<0.05) in
correlation with >=2/3 statistical quality measures on >=2/3 architectures.

**FALSIFIED** if E significantly outperforms R on ALL measures on ALL
architectures.

**INCONCLUSIVE** otherwise.

---

## Results Summary

### Test 1: R vs Statistical Quality Measures

Three honestly-labeled frequentist quality measures were tested:

1. **Inverse trace of covariance** -- trivially related to E (rho = 1.000),
   included for completeness only.
2. **Bootstrap mean precision** -- 1/trace(Var(bootstrap centroids)),
   a frequentist resampling quantity measuring centroid estimation precision.
3. **Silhouette score** -- standard clustering quality metric, genuinely
   independent of E.

ESS was **removed** (it is n*(1-E), trivially identical to E in rank).

#### Spearman Correlations (non-trivial measures only)

| Architecture | R_full vs boot_prec | E vs boot_prec | R_full vs silhouette | E vs silhouette |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | 0.953 | **0.979** | 0.555 | **0.685** |
| all-mpnet-base-v2 | 0.942 | **0.981** | 0.498 | **0.648** |
| multi-qa-MiniLM-L6-cos-v1 | 0.950 | **0.978** | 0.425 | **0.638** |

E outperforms R_full on every measure on every architecture.

#### Steiger's Test (R_full vs E)

| Architecture | Measure | \|rho_R\| | \|rho_E\| | Z | p | Winner |
|---|---|---|---|---|---|---|
| MiniLM | bootstrap_mean_prec | 0.953 | 0.979 | -20.37 | <0.0001 | **E** |
| MiniLM | silhouette | 0.555 | 0.685 | -10.44 | <0.0001 | **E** |
| mpnet | bootstrap_mean_prec | 0.942 | 0.981 | -26.32 | <0.0001 | **E** |
| mpnet | silhouette | 0.498 | 0.648 | -10.17 | <0.0001 | **E** |
| multi-qa | bootstrap_mean_prec | 0.950 | 0.978 | -16.38 | <0.0001 | **E** |
| multi-qa | silhouette | 0.425 | 0.638 | -11.83 | <0.0001 | **E** |

E significantly outperforms R_full on ALL measures on ALL architectures.

#### Partial Correlations rho(R_full, measure | E)

| Architecture | Measure | Partial rho | p | Significant? |
|---|---|---|---|---|
| MiniLM | bootstrap_mean_prec | +0.158 | 0.139 | No |
| mpnet | bootstrap_mean_prec | +0.019 | 0.859 | No |
| multi-qa | bootstrap_mean_prec | +0.370 | 0.0004 | Yes |
| MiniLM | silhouette | -0.509 | <0.0001 | Yes |
| mpnet | silhouette | -0.509 | <0.0001 | Yes |
| multi-qa | silhouette | -0.545 | <0.0001 | Yes |

For bootstrap mean precision, R_full adds near-zero information beyond E
(2/3 architectures non-significant). For silhouette, the partial correlation
is **negative** -- R_full's additional components (grad_S, sigma, Df)
actively degrade the correlation with cluster quality after controlling for E.

### Test 2: Intensive Property

| Domain | Mean CV | Pass? |
|---|---|---|
| Text (384-dim) | 0.400 | NO (>>0.15) |
| Housing (8-dim) | 0.024 | YES |
| **Domain-weighted** | **0.212** | **NO** |

R_full is NOT intensive in the text embedding domain. The large drop at
N=500 (from ~1.0 to ~0.2) indicates strong sensitivity to sample size in
high-dimensional spaces. The housing domain passes trivially due to low
dimensionality (d=8, n>>d).

### Test 3: Continuous Purity Prediction

Redesigned from v2's trivially-easy binary gating (F1=1.0 for all methods).

| Architecture | \|rho(R_full, purity)\| | \|rho(E, purity)\| | Steiger Z | p | Winner |
|---|---|---|---|---|---|
| MiniLM | **0.892** | 0.833 | +11.43 | <0.0001 | **R_full** |
| mpnet | **0.886** | 0.813 | +11.38 | <0.0001 | **R_full** |
| multi-qa | 0.872 | 0.869 | +0.74 | 0.461 | Neither |

R_full significantly outperforms E in predicting continuous purity on 2/3
architectures. This is the one test where R_full adds value.

---

## Criterion Evaluation

| Criterion | Result | Detail |
|---|---|---|
| A: Steiger on quality measures | **FALSIFY** | E wins all measures, all archs |
| B: Intensive property | **FALSIFY component** | Domain-weighted CV = 0.212 > 0.15 |
| C: Purity prediction | **CONFIRM** | R_full wins 2/3 archs |

Decision rule: FALSIFIED if A_falsify OR (B_falsify AND C_falsify).
A_falsify = YES.

---

## VERDICT: FALSIFIED

R = (E / grad_S) * sigma^Df does NOT correlate with standard statistical
quality measures better than E alone. In fact, E alone significantly
outperforms R_full on every measure tested, across all three architectures.

### Key Findings

1. **E dominates R for statistical quality correlation.** The additional
   formula components (grad_S, sigma^Df) add noise, not signal, when
   measuring correlation with standard statistical quality measures.

2. **R_full adds near-zero information beyond E.** Partial correlations
   show that after controlling for E, R_full's residual correlation with
   bootstrap mean precision is non-significant (2/3 architectures). For
   silhouette score, the partial correlation is significantly **negative**.

3. **R_full is NOT intensive in high-dimensional spaces.** CV = 0.40 in
   the text domain (384-dim). The formula is sensitive to sample size when
   n is comparable to or smaller than d.

4. **R_full does predict purity better than E.** On 2/3 architectures,
   R_full significantly outperforms E in predicting cluster purity. The
   sigma^Df term captures structural information related to purity that E
   misses. This is a genuine positive finding, but it is not what Q15 asks.

### Honest Interpretation

The original Q15 ("Does R have a valid Bayesian interpretation?") cannot be
answered by this test because no Bayesian inference was performed (v2's
"Bayesian" quantities were frequentist). The reframed question ("Does R
correlate with statistical quality measures better than E?") is answered
clearly: **No, E alone is better.**

However, R_full does contain useful information. Its superior purity
prediction suggests the formula captures something about cluster structure
that E alone misses. This could be investigated as a separate question
(e.g., Q15b: "Does R capture structural cluster properties beyond E?").

---

## Audit Fixes Applied

| Audit Issue | Fix in v3 |
|---|---|
| BUG-1: ESS trivially = E | Removed entirely |
| BUG-2: Bootstrap precision mislabeled | Renamed to "bootstrap mean precision (frequentist)" |
| BUG-3: Biased CV aggregation | Domain-weighted averaging (equal weight text/housing) |
| BUG-4: Trivially easy gating | Replaced with continuous purity prediction |
| STAT-1: No significance test for E > R | Steiger's test on all comparisons |
| STAT-2: Only Spearman | Both Spearman and Pearson reported |
| STAT-3: No confidence intervals | Bootstrap 95% CIs for all key correlations |
| METH-1: Fake Bayesian labels | All labels honest, no Bayesian claims |
| METH-2: No partial correlations | rho(R_full, measure | E) computed throughout |

---

## Data and Reproducibility

- **Seed**: 42
- **Clusters**: 90 with continuous purity variation (20 pure, 20 mostly-pure, 20 mixed-2, 15 mixed-3, 15 random)
- **Architectures**: all-MiniLM-L6-v2, all-mpnet-base-v2, multi-qa-MiniLM-L6-cos-v1
- **Results file**: `results/test_v3_q15_results.json`
- **Code**: `code/test_v3_q15.py`
