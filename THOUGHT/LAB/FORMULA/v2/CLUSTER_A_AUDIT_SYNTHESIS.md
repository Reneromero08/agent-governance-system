# Cluster A Audit Synthesis Report

**Date:** 2026-02-06
**Method:** 7 independent adversarial auditors, one per Q, reading code + results + verdicts
**Mandate:** 110% rigor. Assume nothing correct until verified. Find every error.

---

## Scorecard Change Summary

| Q | Original Verdict | Auditor Recommendation | Confidence | Key Reason |
|---|-----------------|----------------------|------------|------------|
| Q01 | FALSIFIED | **INCONCLUSIVE** | HIGH | Underpowered (n=20, ~35% power). mpnet shows significant partial corr. Ground truth tautologically linked to E. |
| Q02 | FALSIFIED | **INCONCLUSIVE** | HIGH | grad_S direction prediction was WRONG (test bug, not formula bug). Correcting it: 2/4 fail, not 3/4. |
| Q03 | INCONCLUSIVE | **INCONCLUSIVE** (but weaker) | MEDIUM | +0.068 text improvement never significance-tested. Financial "pass" is near-tautological. E was never changed across domains. |
| Q05 | FALSIFIED | **INCONCLUSIVE** | HIGH | Bias attack exploits cosine similarity, not R specifically. E's inflation ratio never computed. Echo chamber test is redundant with purity correlation. |
| Q09 | INCONCLUSIVE | **INCONCLUSIVE** (but weaker) | HIGH | r=0.97 is not tautological but is confounded by 3-group design. "F" is not FEP free energy, just Gaussian NLL. Within-group correlations never tested. |
| Q15 | INCONCLUSIVE | **INCONCLUSIVE** (but reframed) | HIGH | "Bayesian quantities" are actually frequentist statistics. E vs R_full difference never significance-tested. ESS is trivially identical to E. |
| Q20 | FALSIFIED | **INCONCLUSIVE** | HIGH | 8e test computes PR*alpha (wrong quantity, should be Df*alpha, which = 2 trivially). Bootstrap p-values are 0.029-0.053, borderline. |

**Original scorecard:** 4 FALSIFIED, 3 INCONCLUSIVE
**Audited scorecard:** 0 FALSIFIED, 7 INCONCLUSIVE, 0 CONFIRMED

---

## Systemic Issues Found Across All 7 Tests

### 1. Statistical Power Crisis
Every test uses n=20 to n=60 clusters from one dataset (20 Newsgroups). At these sample sizes, the tests lack power to detect the effects they're looking for. Multiple tests set demanding thresholds (0.05 rho improvement, 3/4 component criteria) that require large samples to resolve.

**Impact:** Tests are biased toward FALSIFIED because they can't detect real effects.

### 2. Ground Truth Contamination
Silhouette score (cosine metric) and E (mean cosine similarity) are computed from the same similarity space. This makes it structurally difficult for any component beyond E to add detectable value, because the ground truth is already captured by E.

**Impact:** Q01, Q15, Q20 all find "E alone is as good as R" -- but this may be an artifact of measuring ground truth in E's own space.

### 3. Missing Significance Tests for Key Comparisons
Multiple verdicts rest on comparing two Spearman correlations (e.g., rho=0.95 vs 0.93) without Steiger's test or bootstrap comparison. Differences of 0.02-0.05 at n=60 are often not significant.

**Affected:** Q01, Q03, Q15, Q20

### 4. Theoretical Predictions Not Derived From Theory
Tests assume component directions (e.g., "grad_S should decrease for pure clusters") without citing derivations. When these assumptions are wrong, the test fails the formula for the test's mistake.

**Affected:** Q02 (grad_S direction), Q20 (8e conservation quantity)

### 5. Single-Dataset Fragility
All tests use 20 Newsgroups as the primary dataset. This is one dataset with 20 categories. Results may not generalize. The 3-group design (pure/mixed/degraded) creates confounds.

**Affected:** All 7 tests

### 6. E-as-Baseline Problem
Every test compares R against E-alone. Since R = E/grad_S * sigma^Df, and sigma^Df is approximately constant, R is approximately proportional to E/grad_S. The tests are really asking "does dividing by grad_S help?" -- and at n=20-60, this is hard to detect.

**Impact:** The formula's value-add over E is small and hard to measure with current sample sizes.

---

## Code Bugs Found (Cross-Q)

| Q | Bug ID | Severity | Description |
|---|--------|----------|-------------|
| Q02 | BUG-1 | MAJOR | grad_S theoretical prediction direction wrong |
| Q05 | BUG-1 | CRITICAL | numpy bool serialization: `is True` fails on string "True", miscounts echo chamber results |
| Q20 | BUG-1 | CRITICAL | 8e conservation tests PR*alpha instead of Df*alpha |
| Q20 | BUG-2 | MODERATE | Df*alpha = (2/alpha)*alpha = 2 trivially under v2 definitions -- conservation law is incoherent |
| Q15 | BUG-1 | MAJOR | ESS = n*(1-E) is trivially identical to E, not an independent "Bayesian" metric |

---

## What The Formula Actually Shows (Honest Assessment)

Based on all 7 audits, here is what the evidence actually supports:

### Confirmed
1. **E is an excellent predictor of cluster quality** (rho > 0.85 across all architectures, all tests)
2. **R is architecturally invariant** (inter-architecture rho > 0.97)
3. **log(R) correlates with Gaussian NLL** (r=0.97, though confounded by 3-group design)
4. **R works in practice** for separating pure from random clusters (d > 4.5)

### Uncertain (needs more data)
5. **grad_S adds modest normalizing value** beyond E (evidence in Q01-mpnet, Q02-corrected direction)
6. **R_full is the best ablation form** (consistent across architectures, but margins are tiny and borderline significant)
7. **R generalizes to text domains** (but improvement over E is untested for significance)

### Not Supported
8. **sigma^Df adds meaningful signal** (approximately constant across topics within each model)
9. **8e conservation law** (either tests wrong quantity or is trivially true by definition)
10. **"Bayesian interpretation"** (tested quantities are frequentist, not Bayesian)
11. **FEP connection** (tested against Gaussian NLL, not actual free energy)
12. **Adversarial robustness** (bias attack works, but unclear if R-specific)
13. **Cross-domain generalization** (E was never actually changed; non-text domains are questionable fits)

---

## Recommended Next Steps

### Must-Fix Before Any Verdict Is Final
1. **Increase sample size** to n >= 100 clusters, ideally from multiple datasets
2. **Use non-cosine ground truth** (external labels, downstream task performance) to break E-ground-truth confound
3. **Run Steiger's test** on all E-vs-R comparisons
4. **Fix the 8e conservation test** to use the correct quantity (and check if the law is even meaningful under v2 definitions)
5. **Compute E's inflation ratio** under the same bias attack as R (Q05)
6. **Compute within-group correlations** for the log(R) vs -F relationship (Q09)

### Should-Fix
7. Fix numpy bool serialization bug in Q05
8. Correct grad_S direction prediction in Q02
9. Add a second dataset (e.g., AG News, DBpedia) for cross-validation
10. Define proper Bayesian quantities (prior + likelihood + posterior) for Q15

---

## Bottom Line

The original Cluster A scorecard (4 FALSIFIED, 3 INCONCLUSIVE) was **too harsh**. Multiple verdicts were driven by test bugs, wrong theoretical predictions, underpowered designs, and missing significance tests rather than genuine formula failures.

The corrected scorecard is **0 FALSIFIED, 7 INCONCLUSIVE**. This does NOT mean the formula is confirmed -- it means the tests as designed cannot resolve the questions either way.

The formula's core value proposition -- that R = (E/grad_S) * sigma^Df adds meaningful signal beyond E alone -- remains **unresolved**. E does most of the work. Whether the other components add detectable value requires larger samples, better ground truth, and correctly specified tests.
