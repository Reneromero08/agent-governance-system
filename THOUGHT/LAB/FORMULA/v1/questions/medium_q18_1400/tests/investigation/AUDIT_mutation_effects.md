# ADVERSARIAL AUDIT: Delta-R Mutation Effects Prediction

**Date:** 2026-01-26
**Auditor:** Skeptical Analysis
**Verdict:** MISLEADING - Correlation is real but trivial and not novel

---

## Executive Summary

The claimed result that delta-R predicts mutation fitness (rho=0.107-0.127, p<1e-6) is **technically correct but practically meaningless**. The correlation:

1. Is **WEAKER than simple amino acid properties** (volume change alone does better)
2. Explains only **1.5-3.5% of variance** (96-98% unexplained)
3. Has **inflated significance** due to non-independent observations
4. Is **3-6x worse** than any established mutation predictor
5. The direction is correct but this is **obvious biology**, not novel insight

**CONCLUSION:** This is not evidence of a novel predictive framework. It is a trivial restatement that disruptive amino acid changes tend to harm proteins.

---

## 1. Effect Size Analysis

### Claimed Results
| Protein | Spearman rho | p-value |
|---------|-------------|---------|
| BRCA1   | 0.127       | 2.8e-15 |
| UBE2I   | 0.123       | 1.3e-11 |
| TP53    | 0.107       | 2.6e-07 |

### Reality Check: Variance Explained

| Protein | R-squared | Variance Explained | Unexplained |
|---------|-----------|-------------------|-------------|
| UBE2I   | 0.0353    | 3.5%              | **96.5%**   |
| TP53    | 0.0279    | 2.8%              | **97.2%**   |

**Interpretation:**
- Using delta-R to predict fitness is barely better than guessing the mean
- RMSE/std ratio = 0.98 (1.0 = no predictive value)
- By Cohen's guidelines, these are **SMALL** effect sizes

---

## 2. Confound Analysis: Is Delta-R Better Than Simple Properties?

### Direct Comparison of Predictors

**UBE2I (n=3021):**
| Predictor | Spearman rho | vs delta-R |
|-----------|-------------|------------|
| Volume change alone | **0.160** | BETTER |
| Hydrophobicity change | 0.111 | Similar |
| Charge change | -0.004 | Useless |
| **Delta-R** | 0.123 | Baseline |

**TP53 (n=2314):**
| Predictor | Spearman rho | vs delta-R |
|-----------|-------------|------------|
| Volume change alone | **0.131** | BETTER |
| Hydrophobicity change | 0.124 | Similar |
| Charge change | -0.002 | Useless |
| **Delta-R** | 0.107 | Baseline |

### Critical Finding

**Delta-R is WORSE than simple volume change alone!**

- UBE2I: Volume (0.160) beats delta-R (0.123) by 30%
- TP53: Volume (0.131) beats delta-R (0.107) by 22%

**Implication:** The "sophisticated" R-framework adds NO VALUE over simply measuring how much the amino acid size changes. This is not a discovery - it is known biochemistry repackaged.

---

## 3. Independence Problem: Inflated Sample Size

### The Issue
Mutations at the **same protein position** are not independent observations. All 19 substitutions at position X share structural context.

### Actual Sample Structure

| Protein | Claimed N | Unique Positions | Avg Mutations/Position |
|---------|-----------|------------------|----------------------|
| UBE2I   | 3,021     | 159              | 19.0                 |
| TP53    | 2,314     | 392              | 5.9                  |

### Effective Sample Size Calculation

Using Intraclass Correlation (ICC) to estimate design effect:

| Protein | ICC | Design Effect | **Effective N** |
|---------|-----|---------------|-----------------|
| UBE2I   | 0.44| 9.0           | **335** (not 3021) |
| TP53    | 0.64| 4.1           | **558** (not 2314) |

### Corrected P-Values

| Protein | Original p | **Adjusted p** | Still Significant? |
|---------|-----------|----------------|-------------------|
| UBE2I   | 1.3e-11   | **0.024**      | Marginal |
| TP53    | 2.6e-07   | **0.012**      | Yes |

With Bonferroni correction for 3 tests (alpha = 0.0167):
- UBE2I: **FAILS** (0.024 > 0.0167)
- TP53: Passes (0.012 < 0.0167)

**The claimed "all p<1e-6" is misleading because:**
1. Observations are not independent
2. Sample size is inflated ~10x for UBE2I
3. After correction, one result fails Bonferroni

---

## 4. Direction of Correlation

### Does It Make Biological Sense?

The correlation IS in the expected direction:
- Lower delta-R (more disruptive mutation) correlates with lower fitness
- This is **correct but obvious**

Example mutations from UBE2I:
```
CONSERVATIVE (high delta-R, high fitness):
  L113I: delta-R=-0.046, fitness=0.82 (Leu->Ile, similar)

RADICAL (low delta-R, low fitness):
  A106R: delta-R=-0.88, fitness=-0.10 (Ala->Arg, very different)
```

**This is basic protein biochemistry:**
- Similar amino acids = conservative substitution = tolerable
- Different amino acids = radical substitution = often deleterious

This has been known since the 1960s (Dayhoff matrices). Delta-R adds nothing new.

---

## 5. Comparison to Established Methods

### Literature Baselines for DMS Prediction

| Method | Typical Spearman rho | Notes |
|--------|---------------------|-------|
| SIFT | 0.40-0.55 | Conservation-based |
| PolyPhen-2 | 0.40-0.55 | Sequence + structure |
| EVmutation | 0.40-0.60 | Deep evolutionary model |
| ESM-1b | 0.45-0.65 | Protein language model |
| **Delta-R** | **0.10-0.13** | This work |

**Delta-R is 3-6x WORSE than any established method.**

It would not be competitive for any real variant effect prediction task.

---

## 6. Data Quality Issues

### Measurement Noise

UBE2I provides standard errors:
- Mean SE: 0.087
- Effect size (rho * std): 0.056
- **Effect/noise ratio: 0.65** (effect smaller than noise!)

TP53 provides NO standard errors:
- Cannot assess measurement reliability
- Fitness scores appear to be single-point estimates

---

## 7. Multiple Testing Concerns

### Were Failed Proteins Hidden?

Only 3 proteins tested, all showed positive (weak) correlation.

However, given:
- Effect sizes barely above zero
- Adjusted p-values marginal after corrections
- Selection of proteins with high-quality DMS data

It is plausible that a 4th or 5th protein was tested and dropped if negative.

No pre-registration of hypotheses was evident.

---

## Final Verdict: MISLEADING

### What the Claims Say:
"Delta-R predicts mutation fitness for BRCA1 (rho=0.127), UBE2I (rho=0.123), TP53 (rho=0.107) with all p<1e-6. This demonstrates genuine biological predictive power."

### What the Data Actually Show:

1. **The correlation exists** but is tiny (R^2 < 4%)
2. **Simple volume change predicts better** than delta-R
3. **P-values are inflated** due to clustered data
4. **The result is obvious** - it just says disruptive changes are bad
5. **It performs 3-6x worse** than any real mutation predictor

### Is This "Biological Predictive Power"?

**NO.** This is:
- A trivial restatement of known biochemistry
- Worse than simple existing features
- Not useful for any practical prediction
- Statistically overstated

### Recommendations

1. Remove claims of "predictive power" - this is misleading
2. Acknowledge delta-R underperforms volume change alone
3. Correct sample size reporting for non-independence
4. Do not compare to established predictors without proper benchmarks
5. Consider this a negative result: R-framework does not improve mutation prediction

---

## Appendix: Reproducibility

All analyses performed on data in:
- `THOUGHT/LAB/FORMULA/questions/18/real_data/cache/`

Key data files:
- `dms_data.json` (BRCA1)
- `dms_data_ube2i.json`
- `dms_data_tp53.json`

Test code: `test_dms_delta_r.py`

Results file: `dms_test_results.json`
