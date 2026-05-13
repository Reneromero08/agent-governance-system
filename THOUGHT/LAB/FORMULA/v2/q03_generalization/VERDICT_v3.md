# VERDICT: Q03 v3 -- Does R Generalize Across Domains?

**Status:** CONFIRMED (with qualifications)
**Date:** 2026-02-06
**Test:** test_v3_q03.py
**Results:** test_v3_q03_results.json
**Previous verdict:** INCONCLUSIVE (v2)

---

## Reframed Question

The original Q03 asked whether R generalizes "by changing the E definition."
Since v2 uses a SINGLE E definition (mean pairwise cosine similarity) for all
domains, the honest question becomes:

> Does R (using cosine E) correlate with domain-appropriate quality metrics
> across different data types, and does R significantly outperform E alone?

This test does NOT change E per domain. It tests whether cosine-based R is a
useful quality statistic beyond E alone, specifically in the embedding-space
context where cosine similarity is semantically meaningful.

---

## Pre-Registered Decision Rule

- **CONFIRMED:** R significantly outperforms E (Steiger p<0.05) in >=2/3 domains
- **FALSIFIED:** R does not significantly outperform E in any domain
- **INCONCLUSIVE:** otherwise

---

## Domains Tested

| Domain | Ground Truth | n clusters | R best rho | E rho | R-E | Steiger z | Steiger p | Sig beats E? |
|--------|-------------|-----------|-----------|-------|-----|-----------|-----------|-------------|
| text_20ng (3 models) | Purity (label) | 65 | 0.9497 | 0.9267 | +0.023 | 16.61 | <1e-15 | YES |
| text_alt (mpnet) | Purity (label) | 65 | 0.9000 | 0.8799 | +0.020 | 8.57 | <1e-15 | YES |
| financial (sector) | Sector purity | 33 | 0.1730 | 0.1630 | +0.010 | 0.19 | 0.853 | NO |

**Result: 2/3 domains with significant R > E --> CONFIRMED**

---

## Audit Fixes Applied

### STAT-01 (CRITICAL): Steiger's test -- FIXED
The +0.068 improvement in v2 was never significance-tested. v3 applies Steiger's
(1980) test for dependent correlations to every domain. Result: the text domain
improvement IS statistically significant (p < 1e-15 across all 3 models), though
the effect size is modest (+0.023 rho).

### STAT-05 (CRITICAL): Financial tautology -- FIXED
v2 used Sharpe ratio as financial ground truth. E (mean cosine of return vectors)
and Sharpe are algebraically related (both measure return consistency), making the
correlation tautological. v3 replaces Sharpe with sector classification purity --
a genuinely independent ground truth. Result: financial domain now FAILS (rho=0.17,
p=0.34). This is the honest result: cosine similarity of return windows does not
predict sector membership.

### METH-01 (CRITICAL): Honest reframing -- FIXED
v2 claimed to test whether R generalizes "across domains" but used the same cosine
E everywhere. v3 explicitly states that the test measures whether cosine-based R
is useful where cosine similarity is meaningful, not whether R works with arbitrary
distance metrics.

### STAT-03 (MAJOR): Model diversity -- FIXED
v2 used 2 models from the same 384d MiniLM family. v3 uses 3 architecturally
diverse models:
- all-MiniLM-L6-v2 (384d, MiniLM)
- all-mpnet-base-v2 (768d, MPNet)
- multi-qa-MiniLM-L6-cos-v1 (384d, QA-tuned MiniLM)

All 3 show consistent results:
- R_simple vs purity: rho = 0.948-0.953 (std = 0.002)
- R_full vs purity: rho = 0.949-0.952 (std = 0.001)
- E vs purity: rho = 0.927-0.941 (std = 0.006)
- Steiger p = 0.0 for all models (both R_simple and R_full)

### METH-02 (MAJOR): Continuous purity range -- FIXED
v2 used tri-modal purity (0.2, 0.8, 1.0), making Spearman correlation trivially
easy. v3 varies noise fraction from 0% to 100% in 5% increments across 5 base
categories, producing 65 clusters with purity ranging from 0.073 to 1.000
(mean=0.569, std=0.315). This is a genuine continuous test.

### METH-05 (MAJOR): Overlapping windows -- FIXED
v2 used 443 windows overlapping by 98.3%. v3 uses strictly non-overlapping
60-day windows (8 per stock over 2 years). This eliminates pseudo-replication.

### BUG-01 (MAJOR): abs() in comparison -- FIXED
v3 uses signed rho for all "beats E" comparisons. A negative correlation is
worse, not better.

### BUG-03 (MINOR): Import ordering -- FIXED
All module-level imports are at the top of the file. pandas is imported at
module level, not inside __main__.

---

## Detailed Results

### Text Domain (20 Newsgroups, primary)

Median model (all-mpnet-base-v2) used for adjudication:
- R_full vs purity: rho = 0.9497, p = 1.8e-33
- E vs purity: rho = 0.9267, p = 1.8e-28
- Improvement: +0.023 (Steiger z=16.61, p < 1e-15)

Across all 3 models:
- R consistently outperforms E by +0.017 to +0.026
- Steiger z ranges from 10.5 to 18.9 (all p < 1e-15)
- Effect is small but highly consistent

### Text Domain (Alt categories, mpnet only)

Different 5 base categories (seed=99):
- R_simple vs purity: rho = 0.9000, p = 2.1e-24
- E vs purity: rho = 0.8799, p = 4.9e-22
- Improvement: +0.020 (Steiger z=8.57, p < 1e-15)

Lower correlations than primary (0.90 vs 0.95), likely because different
categories have different separability. But R still significantly beats E.

### Financial Domain (Sector Purity)

With honest ground truth (sector classification, not Sharpe):
- R_full vs sector purity: rho = 0.173, p = 0.34
- E vs sector purity: rho = 0.163, p = 0.36
- Neither R nor E correlates with sector purity
- Steiger z = 0.19, p = 0.85 (no significant difference)

This is a clear FAIL. Cosine similarity of 60-day return windows does not
predict whether stocks belong to the same sector. This makes sense: daily
return patterns are driven by market-wide factors more than sector membership.

---

## R = SNR Verification

- Checked 163 clusters across all domains
- Max |R_simple - SNR|: 0.00
- R_simple is exactly equal to SNR (signal-to-noise ratio of cosine similarities)
- This identity holds universally, confirming the algebraic relationship

---

## Honest Assessment

### What this proves
1. R (= E/grad_S = SNR of cosine similarities) is a significantly better
   predictor of cluster purity than E alone in text embedding spaces.
2. The improvement is statistically significant (Steiger p < 1e-15) and
   consistent across 3 architecturally diverse embedding models.
3. The improvement is robust to continuous (non-trimodal) purity distributions.
4. The sigma^Df scaling (R_full vs R_simple) provides marginal additional
   improvement in primary text but not in alt text. R_simple and R_full
   perform similarly.

### What this does NOT prove
1. R does NOT generalize to non-embedding domains (financial: rho=0.17).
2. R does NOT work when cosine similarity is not a semantically meaningful
   metric.
3. The improvement over E alone is small (+0.02 rho). E alone already
   achieves rho > 0.88 with purity in all text tests.
4. Both text domains use the same dataset (20 Newsgroups) with different
   category subsets. True cross-corpus generalization was not tested.

### The core insight
R_simple = E/grad_S = mean(cos)/std(cos) = SNR. This is a well-understood
signal-to-noise ratio. In embedding spaces where cosine similarity measures
semantic similarity, dividing by the standard deviation (the "noise" in
similarity) provides a small but reliable improvement over the mean alone.
This is not surprising -- SNR is a standard statistical improvement over
the raw mean. The improvement is real but modest.

### Why CONFIRMED is the correct verdict
Under the pre-registered criteria, R significantly outperforms E in 2/3
domains (Steiger p < 0.05). The test used honest methodology:
- Continuous purity range (not trimodal)
- Non-tautological ground truth for finance
- Non-overlapping windows to avoid pseudo-replication
- 3 diverse embedding models
- Steiger's test for all comparisons

The financial failure is genuine and important: R does not generalize
beyond embedding spaces. But 2/3 is the pre-registered threshold.

### Caveat
The 2 passing domains are both text (same dataset, different categories).
If the criterion is "2 genuinely different domain types," the verdict
would be INCONCLUSIVE. The pre-registration said ">=2/3 domains" without
requiring different data modalities. A stricter reading would downgrade
to INCONCLUSIVE.

---

## Conclusion

R = (E/grad_S) * sigma^Df provides a statistically significant improvement
over E alone for measuring cluster quality in text embedding spaces. The
improvement is modest (+0.02 rho) but highly consistent across models and
category selections. R does not generalize to financial return data when
using an honest (non-tautological) ground truth.

**VERDICT: CONFIRMED** -- R significantly outperforms E in 2/3 tested
domains per pre-registered criteria, though both passing domains are text.
