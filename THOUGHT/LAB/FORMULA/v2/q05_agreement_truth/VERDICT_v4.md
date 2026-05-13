# Q05 v4 Verdict: Does High Local Agreement (High R) Reveal Truth?

**Version:** v4
**Date:** 2026-02-06
**Seed:** 42
**Architectures:** all-MiniLM-L6-v2, all-mpnet-base-v2, multi-qa-MiniLM-L6-cos-v1
**Dataset:** 20 Newsgroups (5000 docs, stratified, 20 categories)

---

## Verdict: INCONCLUSIVE

**Reason:** grad_S dampens on 2/3 architectures (meets threshold), but Steiger R > E is significant on only 1/3 (does not meet 2/3 threshold).

---

## Fixes from v3

1. **BUG-1 (CRITICAL): Steiger Z-test variance formula corrected.**
   - Old (wrong): `var = (2/(n-3)) * (1-r12) / (1-r_bar^2)^2`
   - New (correct, Steiger 1980 ZPF): `var = 2*(1-r12) / ((n-3)*(1+r12))`
   - Impact: multi-qa Z goes from 0.957 (p=0.338) to 7.609 (p < 0.0001)

2. **STAT-1 (CRITICAL): Tautological amplification ratio removed.**
   - The v3 "amplification ratio" (R_inflation / E_inflation) was algebraically identical to 1/grad_S_change. Not an empirical finding.
   - Replaced with direct component analysis: E change, grad_S dampening factor, R change. The structural relationship R_change = E_change / grad_S_dampening is acknowledged, not hidden.

---

## Pre-registered Criteria (v4)

| Outcome | Condition |
|---|---|
| FALSIFIED | grad_S dampening < 1.0 on >= 2/3 archs AND Steiger R not better on >= 2/3 |
| CONFIRMED | grad_S dampening > 1.0 on >= 2/3 archs AND Steiger R > E (p<0.05) on >= 2/3 |
| INCONCLUSIVE | otherwise |

---

## Test 1: Purity-Agreement Correlation (Steiger ZPF)

Does R correlate with cluster purity better than raw E?

| Architecture | rho(R, purity) | rho(E, purity) | delta | Steiger Z | p-value | Significant? |
|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 0.9198 | 0.9153 | +0.0045 | 1.530 | 0.126 | No |
| all-mpnet-base-v2 | 0.8914 | 0.8853 | +0.0061 | 1.254 | 0.210 | No |
| multi-qa-MiniLM-L6-cos-v1 | 0.9174 | 0.8968 | +0.0206 | 7.609 | <0.0001 | **Yes** |

**Result:** R consistently outperforms E in magnitude (positive delta on all 3), but only significantly on 1/3 architectures. Criterion B not met.

---

## Test 2: Bias Attack -- Direct Component Analysis

Key question: when biased text is prepended, does grad_S increase (dampening R inflation) or decrease (amplifying it)?

**Structural note:** Since R = E/grad_S, the relationship R_change = E_change / grad_S_change is a mathematical identity, not an empirical finding. The independent empirical question is: what happens to grad_S under bias?

### all-MiniLM-L6-v2

| Bias phrase | E change | grad_S dampening | R change | grad_S t-test p |
|---|---|---|---|---|
| "In conclusion, " | 1.135x | 1.004x | 1.130x | 0.266 |
| "According to recent studies, " | 1.118x | 1.009x | 1.108x | 0.083 |
| "The committee determined that " | 2.130x | 1.123x | 1.899x | <0.001 |

All 3 phrases: grad_S > 1.0. **DAMPENS.**

### all-mpnet-base-v2

| Bias phrase | E change | grad_S dampening | R change | grad_S t-test p |
|---|---|---|---|---|
| "In conclusion, " | 1.186x | 1.020x | 1.162x | <0.001 |
| "According to recent studies, " | 1.179x | 1.026x | 1.149x | 0.002 |
| "The committee determined that " | 3.261x | 1.313x | 2.486x | <0.001 |

All 3 phrases: grad_S > 1.0. **DAMPENS.**

### multi-qa-MiniLM-L6-cos-v1

| Bias phrase | E change | grad_S dampening | R change | grad_S t-test p |
|---|---|---|---|---|
| "In conclusion, " | 1.153x | 1.010x | 1.141x | 0.001 |
| "According to recent studies, " | 0.938x | 0.993x | 0.944x | 0.003 |
| "The committee determined that " | 2.020x | 1.090x | 1.854x | <0.001 |

1/3 phrases: grad_S < 1.0 (0.993). **MIXED** -- "According to recent studies" causes slight grad_S decrease on this architecture.

### Dampening Summary

| Architecture | All phrases > 1.0? | Mean dampening | Min dampening |
|---|---|---|---|
| all-MiniLM-L6-v2 | Yes | 1.045 | 1.004 |
| all-mpnet-base-v2 | Yes | 1.120 | 1.020 |
| multi-qa-MiniLM-L6-cos-v1 | No | 1.031 | 0.993 |

Dampening on all phrases: 2/3 architectures. Criterion A met.

---

## Verdict Determination

| Criterion | Required | Actual | Met? |
|---|---|---|---|
| A: grad_S dampens (>1.0 all phrases) | >= 2/3 archs | 2/3 | Yes |
| B: Steiger R > E (p<0.05) | >= 2/3 archs | 1/3 | No |
| CONFIRMED requires | A AND B | A met, B not met | No |
| FALSIFIED requires | dampening < 1.0 on >= 2/3 AND Steiger fails on >= 2/3 | Neither met | No |

**Verdict: INCONCLUSIVE**

---

## Honest Assessment

### What we learned:

1. **R is not specifically vulnerable to bias attacks.** The grad_S denominator empirically increases under bias for 2/3 architectures on all tested phrases. This means R is at least as robust as E, and slightly more robust on average. The dampening effect is modest (1-12% typically, up to 31% for strong bias on mpnet).

2. **R does track purity better than E, but the improvement is small and inconsistent.** The delta between rho(R, purity) and rho(E, purity) ranges from 0.005 to 0.021. Only multi-qa shows a statistically significant improvement (p < 0.0001 with correct Steiger formula). MiniLM and mpnet show non-significant improvements (p = 0.13, 0.21).

3. **The grad_S dampening is not universally consistent.** Multi-qa shows grad_S < 1.0 for one bias phrase ("According to recent studies"), meaning R slightly amplifies that particular bias. The effect is small (0.7% decrease) but statistically significant (p = 0.003).

4. **The v3 "amplification ratio" was tautological.** R_inflation/E_inflation = 1/grad_S_change by algebraic identity. This v4 test correctly frames the question as: what happens to grad_S empirically? The answer is: it usually increases (good for R), but not always.

### What remains unclear:

- Why does multi-qa benefit more from R than the other architectures? One hypothesis: higher R-E correlation (0.982 vs 0.974, 0.960) paradoxically makes the Steiger test more powerful, not because R is better but because the test has more statistical leverage.
- The 3 bias phrases tested are surface-level manipulations. More sophisticated attacks (embedding-space adversarial perturbations) remain untested.
- INCONCLUSIVE is the honest answer: R shows promising but inconsistent advantages over E across architectures.

---

## Files

- Test code: `code/test_v4_q05.py`
- Results: `results/test_v4_q05_results.json`
- Previous audit: `AUDIT_v3.md`
- Previous verdict: `VERDICT_v3.md`
