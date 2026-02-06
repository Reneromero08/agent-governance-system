# VERDICT: Q03 v2 -- Does R Generalize Across Domains?

**Status:** INCONCLUSIVE
**Date:** 2026-02-06
**Runtime:** 565 seconds

## Summary

R = (E / grad_S) * sigma^Df was tested across three genuinely different domains using real data, non-tautological ground truth, and adequate sample sizes. Results are mixed: strong in text and financial, absent in tabular. Critically, R does not consistently outperform E alone.

## Domain Results

### Domain 1: Text (20 Newsgroups) -- PASS, R beats E

| Metric | vs Purity (mean rho, 2 models) | vs Silhouette (mean rho) |
|--------|-------------------------------|--------------------------|
| R_full | **0.8948** | **0.8884** |
| R_simple | 0.8862 | 0.8739 |
| E alone | 0.8272 | 0.8160 |
| SNR | 0.8862 | 0.8739 |
| 1/grad_S | -0.4040 | -0.3757 |

- n = 60 clusters (20 pure, 20 mixed, 20 degraded)
- 2 embedding architectures (all-MiniLM-L6-v2, multi-qa-MiniLM-L6-cos-v1), consistent results
- All p-values < 1e-20
- R_full (rho=0.895) beats E alone (rho=0.827) by +0.068 in correlation with purity
- The sigma^Df term adds value here: R_full > R_simple > E

**Assessment:** Genuine signal. R works well for measuring cluster quality in text embedding space. The fractal scaling term (sigma^Df) provides meaningful improvement over both E alone and R_simple.

### Domain 2: Tabular (California Housing) -- FAIL

| Metric | vs Out-of-Sample R^2 (rho) | p-value |
|--------|---------------------------|---------|
| R_full | 0.015 | 0.950 |
| R_simple | -0.032 | 0.895 |
| E alone | -0.062 | 0.796 |
| SNR | -0.032 | 0.895 |

- n = 20 geographic clusters
- Ground truth: out-of-sample Ridge regression R^2 per cluster
- No metric shows any correlation with predictive quality
- Note: Cluster 15 has R^2 = -168.8 (extreme outlier), but even excluding it, no correlation emerges from the other 19 clusters

**Assessment:** Complete failure. Neither R nor E nor any component correlates with how well a geographic cluster's features predict housing prices. This is expected -- the formula measures pairwise similarity structure, which has no inherent connection to regression predictability.

### Domain 3: Financial (30 Stocks, 60-day Return Windows) -- PASS, but R does NOT beat E

| Metric | vs Sharpe Ratio (rho) | p-value |
|--------|----------------------|---------|
| E alone | **0.8950** | 2.5e-11 |
| R_simple | 0.8901 | 4.6e-11 |
| R_full | 0.8821 | 1.2e-10 |
| SNR | 0.8901 | 4.6e-11 |

- n = 30 stocks across 3 sectors (tech, healthcare, energy/industrial)
- Ground truth: annualized Sharpe ratio
- All metrics show very strong correlation (rho > 0.88)
- E alone (rho=0.895) slightly outperforms R_simple (rho=0.890) and R_full (rho=0.882)
- Dividing by grad_S and multiplying by sigma^Df slightly degrades performance

**Assessment:** Strong correlation exists, but it is driven by E (mean pairwise cosine similarity of return windows), not by the R formula's additional terms. The grad_S normalization and sigma^Df scaling add nothing -- they marginally hurt. Stocks with higher mean daily returns naturally have higher E (mean cosine of return windows) and higher Sharpe ratios. The R formula's signal is entirely coming from its E component.

## R = SNR Verification

- 110 clusters checked across all three domains
- Maximum |R_simple - SNR| = 0.00 (exact machine precision)
- **Confirmed: R_simple is identically SNR** (mean pairwise cosine / std pairwise cosine)

This is a mathematical identity, not an empirical finding. Both compute mean/std of the same pairwise cosine similarities.

## Cross-Domain Transfer

| Transfer | Accuracy | Chance |
|----------|----------|--------|
| Text calibration | 95.0% | ~50% |
| Text -> Tabular | 60.0% | 50% |
| Text -> Financial | 50.0% | 50% |

- Threshold calibrated on text (pure vs mixed clusters): R_simple >= 0.695
- Transfer to tabular: barely above chance (60% with n=20, not significant)
- Transfer to financial: exactly at chance (50%)

**Assessment:** Cross-domain transfer fails. The R_simple threshold that separates pure from mixed text clusters has no predictive value in tabular or financial domains. This is expected given that R values in different domains occupy completely different ranges (text: 0.3-1.5, tabular: 0.5-7.4, financial: -0.02 to +0.08).

## Pre-Registered Criteria Evaluation

From the task specification:
- **CONFIRM requires:** R correlates (rho > 0.5, p < 0.05) in >= 2/3 domains AND R outperforms E alone in >= 2/3 domains
- **FALSIFY requires:** R fails (rho < 0.3) in ALL domains, OR R never outperforms E alone
- **INCONCLUSIVE:** otherwise

Actual results:
- R correlates in 2/3 domains (text and financial) -- meets the first CONFIRM criterion
- R outperforms E alone in only 1/3 domains (text only) -- fails the second CONFIRM criterion
- R does not fail in all domains -- not FALSIFY
- R does beat E in at least one domain -- not "never outperforms"

**Verdict: INCONCLUSIVE**

## Honest Assessment

### What R does well
1. In text embedding space, R (especially R_full) meaningfully captures cluster quality better than E alone. The sigma^Df term adds genuine value by accounting for dimensionality structure.
2. R_simple = SNR is a well-established signal quality measure. Its high correlation with purity/silhouette in text is not surprising -- it is a principled statistic.

### What R does not do
1. R does not "generalize across domains" in any meaningful sense. It works in text and financial domains for completely different reasons, and fails entirely in tabular.
2. In the financial domain, E alone outperforms R. The additional complexity of the formula degrades performance.
3. Cross-domain transfer is no better than random.
4. R cannot predict out-of-sample regression quality (tabular domain).

### The core issue
R = E/grad_S is SNR (signal-to-noise ratio) applied to pairwise cosine similarities. SNR is a useful statistic, but it is not a "universal evidence measure" that generalizes across domains. Its correlation with ground truth depends entirely on whether pairwise cosine similarity structure is relevant to the quality metric in question:
- **Text clusters:** Cosine similarity directly measures semantic coherence. SNR of cosines naturally tracks purity. Works.
- **Financial return windows:** Mean cosine of return windows reflects directional bias (positive returns = positive cosines). Tracks Sharpe ratio. But E alone does this better.
- **Tabular geographic clusters:** Cosine similarity of standardized features does not predict regression quality. Does not work.

### Comparison with v1 claims
The v1 test claimed "universal generalization" across Gaussian, Bernoulli, and Quantum domains. v1 was computing the same E = exp(-z^2/2) everywhere (not cosine similarity). The v2 test uses the formula module's actual E definition (mean pairwise cosine) on genuinely different data. The v2 results are more honest but less impressive: partial success in domains where cosine similarity is inherently meaningful, failure otherwise.

## Files

- Test code: `THOUGHT/LAB/FORMULA/v2/q03_generalization/code/test_v2_q03_fixed.py`
- Results JSON: `THOUGHT/LAB/FORMULA/v2/q03_generalization/results/test_v2_q03_fixed_results.json`
- Formula module: `THOUGHT/LAB/FORMULA/v2/shared/formula.py`
