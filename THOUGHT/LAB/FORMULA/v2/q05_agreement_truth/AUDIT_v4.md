# Q05 v4 Adversarial Audit Report

**Auditor:** Adversarial audit agent (Round 3)
**Date:** 2026-02-06
**Scope:** Code, statistics, methodology, and verdict for Q05 v4 test
**Files reviewed:** formula.py, test_v4_q05.py, test_v4_q05_results.json, VERDICT_v4.md, AUDIT_v3.md

---

## Executive Summary

The v4 test properly addresses both critical issues from the v3 audit: the Steiger Z-test variance formula is now correct (ZPF 1980), and the tautological amplification ratio has been replaced with direct grad_S dampening analysis that honestly acknowledges the structural identity R_change = E_change / grad_S_change. No new bugs were found. The implementation is clean, the statistics are sound, and the INCONCLUSIVE verdict is correct.

One observation deserves note but does not constitute a bug: the single grad_S < 1.0 case (multi-qa + "According to recent studies") occurs in a context where E itself DECREASED under the bias phrase, so it does not represent a bias amplification failure. The criterion design is strict ("all phrases > 1.0") but was pre-registered, so it stands.

| Aspect | Score | Notes |
|---|---|---|
| Code correctness | 9/10 | No bugs found; clean implementation |
| Statistical rigor | 8/10 | ZPF formula correct; one methodological nuance (see STAT-1) |
| Methodological validity | 9/10 | Tautology acknowledged; honest framing; good experimental design |
| Verdict fairness | 9/10 | INCONCLUSIVE is correct; honest assessment is thorough |
| v3 audit issues addressed | 10/10 | Both critical issues fully resolved |
| Overall | 9/10 | Solid work with honest reporting |

---

## 1. v3 Issues Resolution

### BUG-1 (CRITICAL): Steiger Z-test variance formula -- RESOLVED

**v3 issue:** Formula used `(2/(n-3)) * (1-r12) / (1-r_bar^2)^2` instead of `2*(1-r12) / ((n-3)*(1+r12))`.

**v4 fix:** `test_v4_q05.py` line 133 now implements the correct ZPF formula:
```python
var_diff = 2.0 * (1.0 - r12) / ((n - 3) * (1.0 + r12))
```

**Independent verification:** I hand-computed all three Steiger Z-statistics from the raw correlation values in the results JSON. All three match to machine precision:

| Architecture | Code Z | Verified Z | Code p | Verified p | Match |
|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 1.5297 | 1.5297 | 0.1261 | 0.1261 | YES |
| all-mpnet-base-v2 | 1.2541 | 1.2541 | 0.2098 | 0.2098 | YES |
| multi-qa-MiniLM-L6-cos-v1 | 7.6087 | 7.6087 | 2.78e-14 | 2.78e-14 | YES |

**Additional cross-check:** I also computed the Meng-Rosenthal-Rubin (1992) variant, which includes a correction factor h for unequal correlations. Under MRR, multi-qa is still significant (p=0.020) while the others remain non-significant (p=0.65, 0.67). The verdict is invariant to the choice of Steiger variant. **STATUS: FULLY RESOLVED.**

### STAT-1 (CRITICAL): Tautological amplification ratio -- RESOLVED

**v3 issue:** `R_inflation / E_inflation` was algebraically identical to `1/grad_S_change`, making the amplification ratio a mathematical tautology presented as an empirical finding.

**v4 fix:** The amplification ratio is completely removed. Test 2 now reports three direct components:
- E_change (biased/clean)
- grad_S_dampening (biased/clean)
- R_change (biased/clean)

The code comment at line 377-380 and print statement at lines 385-387 explicitly acknowledge:
> "R = E/grad_S, so R_change = E_change/grad_S_change by definition."

I verified the identity holds across all 180 per-cluster data points (9 model-phrase combos x 20 clusters): zero mismatches. **STATUS: FULLY RESOLVED.**

### STAT-2 (MODERATE): Z-statistics not comparable across models -- PARTIALLY ADDRESSED

The v4 VERDICT notes that multi-qa has a larger delta_rho (0.021 vs 0.005-0.006) but does not fully explain why its Steiger test achieves such extreme significance (Z=7.6).

My analysis shows that multi-qa's significance is driven by TWO factors:
1. **Effect size:** 4x larger delta_rho than MiniLM/mpnet
2. **Test power:** Higher R-E correlation (r12=0.982 vs 0.974/0.960) makes the Steiger denominator smaller

Even if all models had multi-qa's r12, MiniLM and mpnet would still not reach significance (p=0.063, p=0.059). Conversely, even with MiniLM's r12, multi-qa would still be highly significant (p<0.0001). **The significance is genuinely driven by multi-qa having a larger R-E gap, not just a statistical artifact of higher r12.** This is not noted in VERDICT_v4 but does not affect the verdict.

### METHOD-5 (MINOR): Fisher transform on Spearman rho -- NOT ADDRESSED

Still using arctanh on Spearman's rho in the Steiger test. Acceptable at n=80. Not a material issue.

---

## 2. New Issues Found

### STAT-NEW-1 (MINOR): Criterion A design is strict but defensible

The CONFIRMED criterion requires grad_S dampening > 1.0 on ALL phrases for at least 2/3 architectures. Multi-qa fails because ONE phrase ("According to recent studies") shows mean grad_S dampening of 0.993 (0.7% below 1.0).

**Nuance:** In this case, E itself DECREASED (E_change = 0.938). The bias phrase made documents LESS similar, not more. A grad_S decrease in this context is NOT a bias amplification failure -- both E and R correctly reflect reduced agreement. The concerning scenario (E increases from bias AND grad_S fails to compensate) does not occur anywhere in the data.

Under alternative criteria:
- Mean dampening > 1.0 per model: 3/3 pass (all have mean > 1.0)
- Majority of phrases > 1.0 per model: 3/3 pass (multi-qa has 2/3 above 1.0)

However, the criterion was pre-registered, so it stands. **This is a strict-but-defensible design choice, not a bug.**

### STAT-NEW-2 (INFORMATIONAL): The Steiger test is asymptotic

The ZPF formula assumes large-sample asymptotics of the Fisher z-transform. At n=80, this is well within the regime where the approximation is reliable. No concern.

### STAT-NEW-3 (INFORMATIONAL): Multi-qa Z=7.6 interpretation

The v4 VERDICT correctly reports multi-qa's highly significant Steiger result (p < 0.0001) but the "Honest Assessment" section raises the hypothesis that this might be a statistical artifact of higher r12. My analysis (see STAT-2 resolution above) shows this hypothesis is **incorrect** -- the significance persists even at MiniLM's lower r12. The effect is genuine: multi-qa's grad_S denominator provides meaningfully more purity-correlation lift than for the other architectures.

---

## 3. Code Review

### Steiger implementation (lines 105-139)

Clean and correct. Proper edge-case handling:
- Returns NaN for n < 4
- Clips correlations to [-0.9999, 0.9999] before arctanh
- Guards against zero denominator
- Guards against negative variance

### Bias attack (lines 362-547)

Properly separates clean and biased embeddings. Re-encodes biased text through the model (no shortcut approximation). The one-sample t-test against 1.0 is the correct test for whether grad_S dampening differs from neutral.

### Formula imports (lines 46-56)

Uses the shared `formula.py` via importlib. Only imports `compute_E`, `compute_grad_S`, `compute_R_simple` -- does not use `compute_R_full` (which includes sigma^Df scaling). This is consistent with Q05 testing the simple R = E/grad_S form.

### No new bugs found.

---

## 4. Verdict Assessment

### Pre-registered criteria evaluation

| Criterion | Required | Actual | Correct? |
|---|---|---|---|
| A: grad_S dampens (all phrases > 1.0) on >= 2/3 | >= 2/3 models | 2/3 (MiniLM, mpnet) | YES |
| B: Steiger R > E (p < 0.05) on >= 2/3 | >= 2/3 models | 1/3 (multi-qa only) | YES |
| CONFIRMED requires | A AND B | A met, B not met | Correctly not confirmed |
| FALSIFIED requires | dampening < 1.0 on >= 2/3 AND Steiger fails on >= 2/3 | Neither met | Correctly not falsified |

### Is INCONCLUSIVE the correct verdict?

**YES.** The data genuinely supports INCONCLUSIVE:

1. R is not specifically vulnerable to bias (grad_S dampens on 2/3 architectures, and the one exception is not a true amplification scenario).
2. R does not consistently outperform E for purity correlation (only 1/3 architectures reaches significance, though all three show positive deltas).
3. Neither CONFIRMED nor FALSIFIED conditions are met.

The verdict narrative in VERDICT_v4.md is honest, transparent, and accurately represents the data.

### Verdict: **UPHELD**

INCONCLUSIVE is the correct verdict. No change required.

---

## 5. Remaining Issues

| ID | Severity | Issue | Action |
|---|---|---|---|
| STAT-NEW-1 | MINOR | Criterion A strict (all-phrases) design excludes E-decreasing cases from dampening credit | Informational only; pre-registered criterion stands |
| STAT-2 | MINOR | Multi-qa's extreme significance not fully explained in VERDICT_v4 honest assessment | Could note that the effect size (4x delta_rho), not r12, drives significance |
| METHOD-5 | MINOR | Fisher transform on Spearman rho is approximate | Acceptable at n=80; no action needed |

No critical or moderate issues remain. All v3 critical bugs are fully resolved.

---

## 6. Summary

The v4 test is methodologically sound, statistically correct, and honestly reported. Both critical v3 issues (Steiger bug and tautological ratio) are properly fixed. The code is clean with no new bugs. The INCONCLUSIVE verdict is the right call: R shows promising but inconsistent advantages over E, with genuine improvement on one architecture (multi-qa) and negligible improvement on the other two.

The one nuance worth noting: the only grad_S < 1.0 case occurs when the bias phrase actually REDUCES similarity (E < 1.0), so it is not a true bias amplification failure. Under a mean-based or majority-based criterion, all three architectures would pass criterion A. But the pre-registered criterion was "all phrases > 1.0," which is strict but defensible.

**Overall: 9/10. This is solid empirical work with honest reporting.**
