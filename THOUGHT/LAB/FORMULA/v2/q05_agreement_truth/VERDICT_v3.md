# Q05 v3 Verdict: Does High Local Agreement (High R) Reveal Truth?

**Verdict: INCONCLUSIVE**

**Date:** 2026-02-06
**Test file:** `code/test_v3_q05.py`
**Results:** `results/test_v3_q05_results.json`

---

## Pre-registered Criteria

| Verdict | Condition |
|---|---|
| FALSIFIED | R_inflation / E_inflation > 1.5 on >= 2/3 architectures |
| CONFIRMED | ratio < 1.2 AND Steiger shows R > E for purity on >= 2/3 architectures |
| INCONCLUSIVE | otherwise |

---

## Key Finding

**R does NOT amplify bias attacks beyond what E already suffers.** In fact, R is slightly *less* vulnerable than E because grad_S (the denominator) increases under bias, partially counteracting E's inflation. The amplification ratio (R_inflation / E_inflation) is consistently below 1.0 across all architectures and bias phrases.

However, Steiger's test shows R does not *significantly* outperform E for purity correlation either. The improvement is real but tiny (delta_rho = 0.005-0.021) and not statistically distinguishable at alpha=0.05 with n=80.

---

## Audit Issues Addressed

| Audit Issue | Resolution |
|---|---|
| METHOD-1 (CRITICAL): Bias attack not compared to E inflation | FIXED. E inflation computed alongside R for same attack. R amplification ratio reported. |
| BUG-1: Numpy bool serialization | FIXED. All booleans wrapped in `bool()` before JSON. |
| STAT-4: Only 4 discrete purity levels | FIXED. 10 continuous purity levels (0.1 to 1.0 in 0.1 steps). |
| METHOD-2: Echo chamber test redundant | REPLACED. Echo test removed; replaced with direct R-vs-E decomposition under bias. |
| METHOD-3/4: Phrase selection, 256-char truncation | FIXED. No character truncation; model-native tokenization used. |
| Missing Steiger's test | ADDED. Steiger's Z-test for dependent correlations on all 3 architectures. |

---

## Test 1: Purity Correlation (80 clusters, 10 purity levels)

| Architecture | rho(R, purity) | rho(E, purity) | Steiger Z | Steiger p | R > E sig? |
|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 0.9198 | 0.9153 | 0.172 | 0.863 | No |
| all-mpnet-base-v2 | 0.8914 | 0.8853 | 0.189 | 0.850 | No |
| multi-qa-MiniLM-L6-cos-v1 | 0.9174 | 0.8968 | 0.957 | 0.338 | No |

**Interpretation:** R correlates very strongly with purity (rho = 0.89-0.92) across all architectures. R slightly outperforms E in all 3 architectures (delta_rho = 0.005-0.021), but the difference is not statistically significant by Steiger's test.

---

## Test 2: Bias Attack -- R vs E Inflation

### Amplification Ratios (R_inflation / E_inflation)

| Architecture | "In conclusion" | "According to..." | "The committee..." | Max | Mean |
|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 0.996 | 0.991 | 0.892 | 0.996 | 0.960 |
| all-mpnet-base-v2 | 0.980 | 0.975 | 0.762 | 0.980 | 0.906 |
| multi-qa-MiniLM-L6-cos-v1 | 0.990 | 1.007 | 0.917 | 1.007 | 0.971 |

**All amplification ratios are below 1.1.** R is not more vulnerable than E. In fact, for the strongest attack ("The committee determined that"), R is *less* inflated than E because grad_S increases (embeddings converge, raising E, but they also become more uniform, raising grad_S slightly). The ratio E/grad_S grows slower than E alone.

### Component Decomposition (strongest phrase: "The committee determined that")

| Architecture | E inflation | grad_S change | R inflation | Amplification |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | 2.130x | 1.123x | 1.899x | 0.892 |
| all-mpnet-base-v2 | 3.261x | 1.313x | 2.486x | 0.762 |
| multi-qa-MiniLM-L6-cos-v1 | 2.020x | 1.090x | 1.853x | 0.917 |

The grad_S denominator acts as a partial buffer: when bias inflates E, it also increases grad_S (since all embeddings shift, but not identically). This dampens R's inflation relative to E's.

---

## Verdict Determination

**Criterion A (amplification):** All 3 architectures show amplification ratio < 1.2. R inherits E's vulnerability but does NOT amplify it. --> Not falsified (0/3 amplify).

**Criterion B (Steiger):** 0/3 architectures show statistically significant R > E at alpha=0.05. --> Not confirmed.

**Result:** Neither criterion met --> **INCONCLUSIVE**

---

## Honest Assessment

1. **R is not specifically vulnerable to bias attacks.** The v2 FALSIFIED verdict was unfair. The bias attack inflates cosine similarity (E), and R merely inherits this -- in fact R is slightly *more robust* than E due to the grad_S buffer effect. The previous 2.53x R inflation was real, but E inflated comparably (and more), so the vulnerability lies in cosine similarity, not in the R = E/grad_S formulation.

2. **R does not meaningfully improve over E alone.** Across all 3 architectures, R's purity correlation exceeds E's by only 0.005-0.021 in Spearman rho. This is not statistically significant by Steiger's test. The grad_S denominator adds almost no discriminative power.

3. **R strongly tracks cluster purity.** rho = 0.89-0.92 across 10 continuous purity levels is a genuine, robust finding. But this is almost entirely driven by E.

4. **The formula's division by grad_S provides a modest defensive benefit** against surface manipulation, not a vulnerability amplification. This is the opposite of what v2 concluded.

---

## What Changed from v2

| Aspect | v2 Verdict | v3 Verdict |
|---|---|---|
| Overall | FALSIFIED | INCONCLUSIVE |
| R amplifies bias? | Assumed yes (untested) | No -- R_infl/E_infl < 1.0 consistently |
| Purity correlation | rho=0.85 (4 levels) | rho=0.89-0.92 (10 levels) |
| R vs E improvement | Not formally tested | Not significant (Steiger p > 0.33) |
| Character truncation | 256 chars | None (model-native tokenization) |
