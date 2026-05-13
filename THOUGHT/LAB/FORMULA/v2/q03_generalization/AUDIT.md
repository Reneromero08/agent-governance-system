# AUDIT REPORT: Q03 v2 -- Generalization Proof

**Auditor:** Adversarial Audit Agent
**Date:** 2026-02-06
**Scope:** Code correctness, statistical validity, methodological soundness, verdict accuracy
**Severity scale:** CRITICAL (invalidates results), MAJOR (materially affects interpretation), MINOR (should be noted), INFO (observations)

---

## 1. Code Bugs Found

### BUG-01: `beats_E_alone` uses absolute value comparison -- masks sign errors [MAJOR]

**File:** `test_v2_q03_fixed.py`, lines 732, 758, 782
```python
beats_e = abs(r_rho) > abs(e_rho)
```

The adjudication uses `abs()` for the "beats E" comparison. This means a metric with rho = -0.90 would be judged as "beating" E with rho = 0.85. For a quality metric, a strong *negative* correlation is worse, not better. In practice this did not trigger a wrong result here (all relevant rhos are positive), but it is a latent bug that could silently produce wrong verdicts in other runs.

**Impact:** None in this run. Latent risk.

### BUG-02: `all_fail` check uses inconsistent key access [MINOR]

**File:** `test_v2_q03_fixed.py`, lines 798-801
```python
all_fail = all(
    abs(d.get("best_R_rho", d.get("best_R_rho_vs_purity", 0))) < 0.3
    for d in domain_outcomes.values()
)
```

The text domain stores `best_R_rho_vs_purity` while tabular/financial store `best_R_rho`. The fallback chain `d.get("best_R_rho", d.get("best_R_rho_vs_purity", 0))` works but is fragile. If a new domain used neither key, it would silently default to 0, which is < 0.3, biasing toward `all_fail = True`.

**Impact:** None in this run. Poor code hygiene.

### BUG-03: Financial domain uses `pd` before import [POTENTIAL RUNTIME]

**File:** `test_v2_q03_fixed.py`, line 466
```python
if isinstance(raw.columns, pd.MultiIndex):
```

The `import pandas as pd` is at line 834 inside `if __name__ == "__main__"`. The `run_financial_domain()` function at line 466 references `pd.MultiIndex`, but `pd` is only imported in the main block. This works only because `import pandas as pd` executes before `run_financial_domain()` is called. If the function were ever called from another module, it would crash with `NameError: name 'pd' is not defined`.

**Impact:** None in this run (called from `__main__`). Structural defect.

### BUG-04: Tabular domain `rng_tabular` defined after function [MINOR]

**File:** `test_v2_q03_fixed.py`, line 435
```python
rng_tabular = np.random.RandomState(42)
```

The `run_tabular_domain()` function (line 332) references `rng_tabular` (line 378), but `rng_tabular` is defined at module scope on line 435, *after* the function definition. This works because Python resolves names at call time not definition time, but it is confusing and fragile.

**Impact:** None. Style issue.

### BUG-05: No NaN filtering on tabular cluster 15 (R^2 = -168.8) [MAJOR]

**File:** Results JSON, line 1412
```json
"r2_oos": -168.81460821972163
```

Cluster 15 (n=2612, the largest cluster) has an absurdly negative out-of-sample R^2 of -168.8. This indicates a pathological regression result (the model predictions are catastrophically wrong on the test set). This extreme outlier was included in the Spearman correlation computation and will dominate rank-order statistics. The verdict notes "even excluding it, no correlation emerges" -- but this was stated without evidence. The correlation should have been computed both with and without this outlier, and both results reported.

**Impact:** The tabular correlation of -0.032 is contaminated by an extreme outlier. While the verdict says tabular fails either way, this was not rigorously demonstrated.

### BUG-06: Tabular domain does not exclude lat/lon from feature vectors [MINOR]

**File:** `test_v2_q03_fixed.py`, lines 357-358
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

The code standardizes all 8 features including latitude and longitude, then clusters on lat/lon. The formula is then computed on all 8 standardized features. This means the "embedding" fed to the formula includes the very features used for clustering, creating a mild circularity: clusters that are geographically tight will have similar lat/lon features, which will increase cosine similarity, but this says nothing about predictive quality. A cleaner design would compute R on the 6 non-geographic features only.

**Impact:** Mild circularity, but the tabular domain failed anyway so this did not produce a false positive.

---

## 2. Statistical Errors

### STAT-01: The "+0.068 improvement" in text is NOT tested for significance [CRITICAL]

The verdict headline claims R_full beats E by +0.068 in correlation with purity. But:

1. The difference between two Spearman correlations (0.895 vs 0.827) is NOT tested for statistical significance. Two correlations from the same sample are not independent -- they share the same Y variable (purity).
2. The proper test is a **Steiger (1980) test for comparing dependent correlations**, or equivalently a Williams/Hotelling test. Neither was performed.
3. With n=60 clusters, a difference of 0.068 between rho=0.895 and rho=0.827 is *plausibly* significant but this must be formally tested. The confidence intervals for these two correlations likely overlap.
4. The verdict states "R works well for measuring cluster quality in text embedding space" and "sigma^Df provides meaningful improvement" without any statistical support for the claim that the improvement is real.

**Impact:** The only domain where R "beats E" might not have a statistically significant advantage. If so, R never significantly outperforms E, which would shift the verdict toward FALSIFY under the pre-registered criteria.

### STAT-02: Financial "rho=0.890 vs rho=0.895" difference is within noise [INFO]

The verdict correctly notes that E outperforms R in the financial domain. The difference (0.005) is negligible and not tested. This is handled honestly.

### STAT-03: Only 2 embedding models for text, both 384-dimensional MiniLM variants [MAJOR]

The text domain uses two models: `all-MiniLM-L6-v2` and `multi-qa-MiniLM-L6-cos-v1`. Both are:
- 384-dimensional
- Based on MiniLM architecture
- Trained with contrastive learning on semantic similarity

Using two similar architectures and averaging their rhos does not constitute a robust multi-model validation. The std_rho values (0.002-0.009) are tiny precisely because the models are near-identical. A proper multi-model test would include architecturally diverse models (e.g., BERT-base 768d, GTE-large 1024d, E5, etc.).

**Impact:** The "consistent across 2 models" claim is weaker than presented. The models are too similar to provide independent validation.

### STAT-04: Tabular domain has only 20 data points [MAJOR]

With n=20 geographic clusters, Spearman correlation has very low statistical power. To detect a moderate effect (rho=0.5) at alpha=0.05 with 80% power, you need n >= 29. The test is underpowered for detecting the effect size it claims to test.

Additionally, geographic k-means clusters are not independent: adjacent clusters share boundary conditions, and the number of clusters (k=20) was not justified.

**Impact:** The tabular "failure" might be a power failure, not a genuine null result. A more careful test would use more clusters (k=50 or k=100 via hierarchical methods) or a different clustering approach.

### STAT-05: Financial domain -- tautological correlation between E and Sharpe [CRITICAL]

This is the most serious statistical issue in the entire test.

Consider what E measures for the financial domain. Each "embedding" is a row of 60 daily returns for a stock. The pairwise cosine similarity between two return windows measures how similarly the stock performed in those two periods. For a stock with consistently positive returns, *all* 60-day windows will have similar directional bias, producing high mean cosine (high E). For a stock with mixed positive/negative returns, windows will point in different directions, producing lower E.

Now consider Sharpe ratio = mean(daily_return) / std(daily_return) * sqrt(252).

Both E and Sharpe are fundamentally measuring the same thing: the *consistency of directional return*. A stock with high mean return and low volatility has both:
- High Sharpe ratio (by definition)
- High E (return windows are consistently positive, hence similar)

This is not "R generalizing to finance." This is a near-tautological relationship: mean cosine of return vectors is algebraically related to the Sharpe ratio of the underlying returns. Specifically, for normalized return vectors, the mean cosine is approximately proportional to (mean/std)^2 for small mean/std ratios.

The rho=0.89 correlation between E and Sharpe is therefore not evidence of generalization -- it is evidence that cosine similarity of return windows is a proxy for return consistency, which is what Sharpe measures. This makes the "financial domain passes" result trivial rather than informative.

**Impact:** The financial domain "pass" is essentially a mathematical relationship, not evidence that R captures something meaningful about financial data quality. The verdict should have identified this.

### STAT-06: Silhouette scores are dubious ground truth for text [MINOR]

Silhouette scores are computed using cosine distance. R is also computed using cosine similarity. Correlating R (based on cosine similarity) with silhouette (based on cosine distance) is partially circular -- both measure the same geometric property. Purity is a better ground truth since it is defined by labels, not geometry.

**Impact:** The R_vs_silhouette correlations are inflated. The R_vs_purity correlations are more meaningful and should be the primary evidence.

---

## 3. Methodological Issues

### METH-01: The formula was designed for semantic embeddings -- testing it on raw features is a category error [CRITICAL]

The formula R = (E / grad_S) * sigma^Df is defined as:
- E = mean pairwise cosine similarity
- grad_S = std of pairwise cosine similarities
- sigma = participation ratio / ambient dimension
- Df = fractal dimension from eigenvalue decay

This was designed for *embedding spaces* where cosine similarity is a meaningful semantic distance metric. Applying it to:
- **Standardized tabular features** (8 housing features): Cosine similarity between feature vectors has no semantic meaning. Two houses with similar feature profiles are "close" in Euclidean space, but their cosine similarity depends on feature scaling and centering in ways that are not meaningful.
- **Rolling return windows** (60 daily returns): Cosine similarity between return windows measures directional alignment, which is a degenerate case -- it mostly reflects whether returns are positive or negative, not any deeper "quality" of the data.

The Q03 question asks if R generalizes "by changing the E definition." But E was NOT changed -- the same `compute_E` (mean pairwise cosine) was used for all three domains. The test actually measures whether mean pairwise cosine similarity is a useful statistic across different data types, which is a different and weaker question.

**Impact:** The test does not actually test what Q03 asks. The question presupposes changing E per domain (e.g., Euclidean distance for tabular, correlation for financial). Using the same cosine E everywhere and expecting it to work on non-embedding data is a setup for failure that was predictable from the start.

### METH-02: Cluster construction introduces confounds in text domain [MAJOR]

The text clusters are constructed as:
- 20 pure: 200 docs from 1 category (purity = 1.0)
- 20 mixed: 200 docs from 3-5 categories (purity = 0.20-0.34)
- 20 degraded: 160 from 1 category + 40 from others (purity = 0.80)

This creates a tri-modal purity distribution: clusters have purity of ~0.20-0.34, ~0.80, or 1.0. There are no clusters with intermediate purity (0.40-0.70). Spearman correlation with this distribution is essentially testing whether R can rank-order three groups, which is a much easier task than predicting purity on a continuous scale.

Furthermore, pure clusters have exactly 1 category, degraded have 1 dominant category + noise, and mixed have 3-5 categories. The number of categories is confounded with purity. R might be tracking the number of categories rather than purity per se.

**Impact:** The rho=0.89 correlation is inflated by the tri-modal purity distribution. A proper test would use a continuous range of purities (e.g., varying the noise fraction from 0% to 100% in 5% increments).

### METH-03: Cross-domain transfer test is inherently unfair [MINOR]

The transfer test calibrates an R_simple threshold on text and applies it to tabular/financial. But:
1. R values in different domains have completely different scales (text: 0.3-1.6, tabular: 0.5-7.4, financial: -0.02 to +0.08).
2. A threshold learned in one scale is meaningless in another.
3. The verdict acknowledges this but includes the test anyway, which adds noise to the report.

A fairer transfer test would normalize R values within each domain (z-scores) before applying a threshold, or use rank-based criteria.

**Impact:** The cross-domain transfer "failure" is trivially expected and does not add information.

### METH-04: Sharpe ratio uses the full return series, not per-window [MINOR]

**File:** `test_v2_q03_fixed.py`, lines 503-509

Sharpe ratio is computed over the entire return series for each stock:
```python
mean_daily = np.mean(ret_series)
std_daily = np.std(ret_series)
sharpe = (mean_daily / std_daily) * np.sqrt(252)
```

But R is computed on the matrix of 60-day *windows*. This means R summarizes the structure of overlapping windows, while Sharpe summarizes the full series. The two are measuring different time horizons: R reflects local 60-day patterns, Sharpe reflects the entire 2-year period. The correlation is high because both are dominated by the stock's overall return direction, but this is a methodological mismatch.

**Impact:** The financial correlation is between a window-level statistic (R) and a series-level statistic (Sharpe), which introduces a level-of-analysis mismatch.

### METH-05: Overlapping windows in financial domain create pseudo-replication [MAJOR]

The financial domain constructs ~443 overlapping 60-day windows per stock (windows shifted by 1 day). These windows overlap by 59/60 = 98.3%. The resulting "embedding matrix" (443 x 60) has extreme autocorrelation in its rows. This means:
- The pairwise cosine similarities are not independent observations
- The effective sample size is much smaller than 443
- The formula components (E, grad_S, sigma, Df) are computed on highly redundant data
- The sigma and Df estimates from eigenvalue analysis will be biased because the covariance matrix reflects temporal autocorrelation, not genuine distributional structure

**Impact:** The formula component values for financial data are unreliable due to pseudo-replication. The Df values (~4-7) are suspiciously high compared to text (~0.18) and tabular (~0.6), likely reflecting autocorrelation structure rather than fractal dimensionality.

---

## 4. Verdict Assessment

### The verdict of INCONCLUSIVE is incorrect. It should be INCONCLUSIVE or arguably FALSIFY.

Let me re-evaluate against the pre-registered criteria:

**CONFIRM requires:** R correlates in >= 2/3 domains AND R outperforms E in >= 2/3 domains.
- R correlates in 2/3 domains: YES (text and financial pass rho > 0.5, p < 0.05)
- R outperforms E in >= 2/3 domains: NO (only text, and even there not significance-tested)
- Result: CONFIRM criteria NOT met. **Correct.**

**FALSIFY requires:** R fails in ALL domains OR R never outperforms E.
- R fails in all domains: NO (text and financial pass)
- R never outperforms E: Depends on STAT-01. If the +0.068 text improvement is not significant, then R never *significantly* outperforms E, which is a reasonable reading of "never outperforms."
- Result: FALSIFY criteria *possibly* met if the text improvement is not significant.

**INCONCLUSIVE:** Otherwise.

The verdict of INCONCLUSIVE is defensible given the pre-registered criteria, but it is generous. Here is why a stronger case can be made for FALSIFY:

1. **Financial domain pass is tautological** (STAT-05): E correlates with Sharpe because both measure return consistency. This is not "generalization."
2. **Text improvement is not significance-tested** (STAT-01): The only domain where R beats E might be a noise artifact.
3. **Tabular is a complete failure** (correct as stated).
4. **Cross-domain transfer fails** (correct as stated).
5. **The question asks about generalization** -- and the honest answer is that R does not generalize. It works in one domain (text, where cosine similarity is designed to be meaningful) and arguably works trivially in another (financial, via tautological relationship).

**My assessment:** The verdict of INCONCLUSIVE is technically correct per the pre-registered criteria but undersells the negative result. A more honest framing would be: "R does not generalize. It is a useful statistic (SNR) in embedding spaces where cosine similarity is meaningful, and trivially correlated with Sharpe in financial data via algebraic relationship. It adds nothing over E alone in 2 out of 3 domains."

The verdict document's "Honest Assessment" section (lines 101-120) actually captures this nuance well and is the most valuable part of the write-up.

---

## 5. Issues Requiring Resolution

### P0 (Must fix before accepting the verdict):

1. **STAT-01:** Test whether rho(R_full, purity) significantly exceeds rho(E, purity) using Steiger's test or bootstrap confidence intervals. If not significant, the verdict should shift to FALSIFY (R never *significantly* outperforms E).

2. **STAT-05:** Acknowledge in the verdict that the financial domain "pass" is a tautological relationship between cosine similarity of return vectors and return consistency. Reclassify financial as "trivially correlated" rather than genuine evidence of generalization.

### P1 (Should fix for credibility):

3. **STAT-04:** Rerun tabular with more clusters (k >= 40) to ensure adequate statistical power, or explicitly note the underpowered design in the verdict.

4. **METH-02:** Rerun text domain with a continuous range of purities (not tri-modal) to get a more honest correlation estimate.

5. **BUG-05:** Report tabular correlations both with and without the outlier cluster (R^2 = -168.8).

### P2 (Should note):

6. **METH-01:** Explicitly state in the verdict that E was NOT changed per domain (contradicting Q03's premise). The test used the same cosine E everywhere.

7. **METH-05:** Note that financial domain Df values (~4-7) are likely artifacts of overlapping windows, not genuine fractal dimension.

---

## 6. What Would Change the Verdict

### To CONFIRM:
- Test the text improvement (+0.068) and show it is statistically significant (p < 0.05 by Steiger's test)
- Change E definition per domain (e.g., Euclidean for tabular, Pearson correlation for financial) and show R outperforms E in >= 2 domains
- Use non-overlapping windows for financial data and show correlation persists
- Use a continuous purity range in text and show R still beats E

### To FALSIFY:
- Show the text improvement (+0.068) is NOT statistically significant
- This would mean R never significantly outperforms E in any domain
- Under the pre-registered criterion "R never outperforms E alone", this triggers FALSIFY

### To strengthen INCONCLUSIVE:
- Properly acknowledge the tautological nature of the financial correlation
- Test more domains (images, audio, biological sequences)
- Use domain-appropriate E definitions as Q03 originally intended

---

## 7. Summary Scorecard

| Category | Finding | Severity |
|----------|---------|----------|
| BUG-01 | abs() in beats_E comparison | MAJOR (latent) |
| BUG-02 | Inconsistent key access for all_fail | MINOR |
| BUG-03 | pd used before import | MINOR |
| BUG-04 | rng_tabular ordering | MINOR |
| BUG-05 | Extreme outlier R^2 not filtered | MAJOR |
| BUG-06 | lat/lon included in features | MINOR |
| STAT-01 | +0.068 not significance-tested | CRITICAL |
| STAT-02 | Financial difference within noise | INFO |
| STAT-03 | Only 2 similar embedding models | MAJOR |
| STAT-04 | Tabular underpowered (n=20) | MAJOR |
| STAT-05 | Financial E-Sharpe tautology | CRITICAL |
| STAT-06 | Silhouette circularity | MINOR |
| METH-01 | Same E for all domains (category error) | CRITICAL |
| METH-02 | Tri-modal purity confound | MAJOR |
| METH-03 | Cross-domain transfer inherently unfair | MINOR |
| METH-04 | Sharpe on full series, R on windows | MINOR |
| METH-05 | Overlapping windows pseudo-replication | MAJOR |

**Counts:** 3 CRITICAL, 5 MAJOR, 6 MINOR, 1 INFO

---

## 8. Final Assessment

The test is competently executed and the verdict document is unusually honest in its self-assessment. The "Honest Assessment" section correctly identifies most of the issues I found. However, the formal verdict of INCONCLUSIVE is generous because:

1. The one domain where R beats E (text) has an untested significance claim
2. The one domain where R correlates strongly (financial) is tautological
3. The test does not actually test what Q03 asks (E was not changed per domain)

The R = SNR identity verification is correct and valuable. The core insight -- that R_simple is just signal-to-noise ratio of pairwise cosine similarities -- is the most important finding of this test. It correctly reduces the formula from a mysterious "evidence measure" to a well-understood statistical quantity whose utility depends entirely on whether cosine similarity is a meaningful metric for the domain in question.

**Bottom line:** The verdict should remain INCONCLUSIVE per the pre-registered criteria, but the narrative should be updated to reflect that (a) the financial pass is tautological, (b) the text improvement is not significance-tested, and (c) Q03's actual question about changing E was not tested. If STAT-01 is resolved and the improvement is not significant, the verdict should be upgraded to FALSIFY.
