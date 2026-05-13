# Q20 v3 Verdict: Is R = (E/grad_S) * sigma^Df Tautological?

**Verdict: NOT TAUTOLOGICAL**

**Date:** 2026-02-06
**Test:** test_v3_q20.py
**Runtime:** 1178s (~20 min)

---

## Pre-Registered Criteria

| Criterion | Required | Observed | Met? |
|-----------|----------|----------|------|
| Steiger p < 0.05 (R_full > E) | >= 2/3 architectures | 3/3 | YES |
| CV R^2(R_full) > CV R^2(E) | >= 2/3 architectures | 3/3 | YES |

Both pre-registered criteria are met. Verdict: **NOT TAUTOLOGICAL**.

---

## Audit Fixes Applied

| Audit Issue | Fix Applied |
|-------------|-------------|
| BUG-1: 8e test computed PR*alpha, not Df*alpha | Dropped 8e conservation entirely (Df*alpha = 2 trivially under v2 defs) |
| STAT-1: Arbitrary 0.05 rho threshold | Replaced with Steiger's Z-test for dependent correlations |
| STAT-3: 60 clusters underpowered | Increased to 90 clusters (30 pure + 30 mixed + 30 random) |
| METH-2: Correlation insufficient for tautology | Added cross-validated prediction comparison |
| METH-3: 8e and tautology conflated | Separated: this test is purely about tautology |
| METH-4: Binary purity (pure vs 50/50) | Mixed clusters now have continuous purity (0.30-0.80 dominant fraction) |

---

## Key Results

### Test 1: Steiger's Z-Test (R_full vs E)

| Architecture | rho(R_full,purity) | rho(E,purity) | Z-stat | p-value | Significant? |
|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 0.868 | 0.840 | 4.82 | 7.3e-07 | YES |
| all-mpnet-base-v2 | 0.872 | 0.834 | 5.89 | 1.9e-09 | YES |
| multi-qa-MiniLM-L6-cos-v1 | 0.877 | 0.865 | 2.36 | 0.009 | YES |

R_full significantly outperforms E alone on all 3 architectures. The weakest result (multi-qa) still has p = 0.009, well below 0.05.

### Test 1b: Steiger's Z-Test (R_simple vs E)

R_simple (= E/grad_S, no sigma^Df term) also significantly outperforms E on all 3 architectures (all p < 0.05). This confirms that the grad_S component adds real predictive value.

### Test 1c: Steiger's Z-Test (R_full vs R_simple)

| Architecture | Z-stat | p-value | Significant? |
|---|---|---|---|
| all-MiniLM-L6-v2 | 3.88 | 5.2e-05 | YES |
| all-mpnet-base-v2 | 4.49 | 3.5e-06 | YES |
| multi-qa-MiniLM-L6-cos-v1 | -0.18 | 0.572 | NO |

The sigma^Df term significantly improves R on 2/3 architectures but not the third. This is the weakest component.

### Test 2: Cross-Validated Prediction (5-fold)

| Architecture | CV R^2(E) | CV R^2(R_simple) | CV R^2(R_full) | R_full - E |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | 0.644 | 0.696 | 0.689 | +0.045 |
| all-mpnet-base-v2 | 0.579 | 0.653 | 0.650 | +0.071 |
| multi-qa-MiniLM-L6-cos-v1 | 0.652 | 0.706 | 0.704 | +0.052 |

R_full outperforms E on all 3 architectures in out-of-sample prediction, with CV R^2 improvements of 4.5-7.1 percentage points. None of these differences are negligible (all > 0.01).

Note: R_simple slightly outperforms R_full in CV R^2, consistent with sigma^Df being a weak contributor that sometimes overfits.

### Test 3: Nested Model Comparison (AIC)

Adding 1/grad_S to E improves AIC on 3/3 architectures (delta > 2).
Adding sigma^Df improves AIC on 2/3 architectures.

This confirms: the dispersion term (grad_S) adds genuine explanatory power. The fractal term (sigma^Df) adds marginal value.

### Test 4: Ablation

R_full is the best ablation form on 2/3 architectures. R_simple beats R_full on the third (multi-qa), consistent with the Steiger and CV results.

### Test 5: Novel Predictions

All 3 architectures pass >= 2/3 novel predictions.

---

## Honest Assessment

### What the data shows:

1. **R_full is NOT a pure tautology.** It significantly and consistently outperforms E alone across all 3 architectures, with proper statistical testing (Steiger Z-test) and out-of-sample validation (5-fold CV R^2). The v2 verdict of FALSIFIED was incorrect -- it was caused by a test bug (wrong conservation quantity) and an unjustified threshold.

2. **The grad_S component is the real value-add.** Dividing E by the standard deviation of pairwise similarities (grad_S) consistently improves prediction. This is not trivial: it shows that cluster quality depends not just on mean similarity but on the signal-to-noise ratio of similarities.

3. **The sigma^Df term is a weak contributor.** It helps on 2/3 architectures (significant by Steiger and AIC) but not the third. In cross-validated R^2, R_simple slightly outperforms R_full, suggesting sigma^Df sometimes overfits. This term is not decorative but also not essential.

4. **R is fundamentally a signal-to-noise ratio.** R_simple = E/grad_S = mean(cosine_sims)/std(cosine_sims). This is by definition a signal-to-noise ratio. The question of whether a well-constructed SNR is "tautological" or "explanatory" is philosophical, not empirical. The empirical fact is that E/grad_S outperforms E alone, which means the formula captures structure that raw similarity does not.

### What this means for the formula:

- The formula R = (E/grad_S) * sigma^Df works, primarily because E/grad_S is a good signal-to-noise ratio.
- The sigma^Df term adds marginal value and could potentially be dropped without major loss.
- The 8e conservation law remains untestable under v2 definitions (Df*alpha = 2 trivially).
- "Not tautological" does not mean "deeply explanatory" -- it means the formula adds measurable predictive value beyond its most informative single component (E).

---

## Raw Numbers

- **n_clusters:** 90 (30 pure, 30 mixed, 30 random)
- **Purity range:** [0.065, 1.000]
- **Dataset:** 20 Newsgroups (18,846 documents, 20 categories)
- **Seed:** 42
- **Data hash:** ed9911d480529fbe
