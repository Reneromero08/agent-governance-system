# Q20 v3 Adversarial Audit: Is R = (E/grad_S) * sigma^Df Tautological?

**Auditor:** Adversarial Audit Agent (Opus 4.6), Round 2
**Date:** 2026-02-06
**Verdict under review:** NOT TAUTOLOGICAL / CONFIRMED
**Prior audit:** AUDIT.md (v2 audit) recommended INCONCLUSIVE; v3 fixed all flagged issues.

---

## Summary Judgment

**AUDIT CONCLUSION: VERDICT SUSTAINED WITH CAVEATS**

The NOT TAUTOLOGICAL verdict is methodologically sound and correctly follows
from the pre-registered criteria. All v2 audit issues were addressed. However,
the verdict carries important semantic limitations that must be understood:
"not tautological" here means "R adds measurable predictive value beyond E
alone" -- it does NOT mean R encodes deep or novel theoretical structure.
R is fundamentally a signal-to-noise ratio, and its superiority over E is
the unsurprising fact that SNR outperforms mean signal alone.

---

## Audit of v2 Fix Implementation

### BUG-1: 8e Conservation (PROPERLY FIXED)

The v2 audit found that the 8e conservation test computed PR * alpha instead
of Df * alpha, and that under v2 definitions Df * alpha = 2 trivially. The v3
test drops the 8e conservation test entirely, with explicit justification:
"Dropped 8e conservation test entirely. Under v2 definitions, Df = 2/alpha,
so Df*alpha = 2 trivially. The conservation law is UNTESTABLE without an
independent Df measurement (e.g. box-counting)."

**Assessment:** Correct fix. The only honest options were (a) drop it or
(b) implement box-counting for Df. Dropping it is the conservative choice.
No new bugs introduced.

### STAT-1: Steiger's Test Replaces Arbitrary Threshold (PROPERLY FIXED)

The v2 test used an unjustified 0.05 absolute rho threshold. The v3 test
replaces this with Steiger's Z-test for dependent correlations, which is the
standard statistical method for comparing two correlations sharing a common
variable. The implementation (lines 64-109 of test_v3_q20.py) follows the
Steiger (1980) / Hotelling (1940) procedure.

**Assessment:** The right test for the right question. See statistical
verification below.

### STAT-3: 90 Clusters (PROPERLY FIXED)

Increased from 60 to 90 clusters (30 pure, 30 mixed, 30 random). Mixed
clusters now have continuous purity variation (0.30-0.80 dominant fraction)
rather than binary 50/50 splits.

**Assessment:** Better design. The continuous purity in mixed clusters
eliminates the v2 gap between pure (1.0) and mixed (0.5) that inflated
correlations.

### METH-2: Cross-Validated R^2 (PROPERLY FIXED)

Added 5-fold cross-validation to test out-of-sample predictive ability.
This is the strongest guard against overfitting and tautology.

**Assessment:** Correct implementation. See detailed verification below.

### METH-3: Nested AIC Comparison (PROPERLY FIXED)

Added cumulative AIC comparison: E -> E + 1/grad_S -> E + 1/grad_S + sigma^Df.
This directly measures whether each component adds explanatory value.

**Assessment:** Appropriate for the question. See detailed verification below.

---

## Statistical Verification

### Steiger's Z-Test: Implementation Audit

**Code reviewed:** test_v3_q20.py lines 64-109

The implementation:
1. Fisher z-transforms both correlations: z = 0.5 * ln((1+r)/(1-r)) -- CORRECT
2. Applies Hotelling's correction for dependent correlations:
   f = (1 - rho_xy) / (2 * (1 - r_mean_sq)), bounded at f <= 1.0 -- CORRECT
3. Computes SE: sqrt(2(1-rho_xy) / ((n-3)(1+r_mean_sq*f))) -- CORRECT
4. Z = (z_xz - z_yz) / SE -- CORRECT
5. One-sided p-value: 1 - norm.cdf(Z) -- CORRECT for testing rho_xz > rho_yz

**Concern identified:** The test uses ABSOLUTE values of rho when calling
steiger_z_test (line 619: `abs(rho_rfull_pur), abs(rho_e_pur), abs(rho_rfull_e)`).
This is correct here because all three correlations are positive (R_full and E
both correlate positively with purity), so abs() has no effect on the values.
But it WOULD be incorrect if any correlation were negative. For this specific
data: R_full rho ranges 0.868-0.877 and E rho ranges 0.834-0.865, all positive.
No issue for this test.

**Results plausibility check:**

| Architecture | rho(R_full) | rho(E) | rho(R_full,E) | n | Z | p |
|---|---|---|---|---|---|---|
| MiniLM-L6 | 0.868 | 0.840 | 0.979 | 90 | 4.82 | 7.3e-07 |
| mpnet-base | 0.872 | 0.834 | 0.974 | 90 | 5.89 | 1.9e-09 |
| multi-qa | 0.877 | 0.865 | 0.980 | 90 | 2.36 | 0.009 |

The high Z-statistics with n=90 are plausible because rho(R_full, E) is very
high (0.974-0.980), meaning these are highly dependent correlations. Steiger's
test is specifically designed for this situation. The large Z despite small
absolute rho differences (0.012-0.038) is because the test accounts for the
extreme correlation between R_full and E -- small differences become more
significant when the two measures are near-identical.

**Key concern: Are the Z-statistics TOO high?**

Let me sanity-check the MiniLM result. rho difference = 0.028, n = 90,
rho_xy = 0.979. Fisher z difference = atanh(0.868) - atanh(0.840) = 1.333 -
1.221 = 0.112. The denominator involves (1 - 0.979) = 0.021, which is tiny.
SE = sqrt(2 * 0.021 / (87 * (1 + 0.713 * f))). With r_mean_sq = 0.5*(0.868^2
+ 0.840^2) = 0.5*(0.753 + 0.706) = 0.730, f = 0.021 / (2*(1-0.730)) =
0.021/0.540 = 0.039. So SE = sqrt(0.042 / (87 * 1.028)) = sqrt(0.042/89.4) =
sqrt(0.00047) = 0.0217. Z = 0.112 / 0.0217 = 5.16.

My manual calculation gets Z ~ 5.16 vs the code's 4.82. Close but not exact.
The difference is likely from rounding in my manual calculation and the exact
clipping/bounding in the code. The order of magnitude is correct. **No bug.**

### Cross-Validated R^2: Implementation Audit

**Code reviewed:** test_v3_q20.py lines 112-152

The implementation:
1. Filters NaN values -- CORRECT
2. Shuffles indices with fixed seed (42) -- CORRECT, ensures same folds for
   all metrics
3. K-fold split without replacement -- CORRECT
4. Fits LinearRegression on train, predicts on test -- CORRECT
5. Uses sklearn r2_score on test set -- CORRECT

**Data leakage check:** The fold assignment is computed once using the same
seed for all metrics. Train/test split is done per fold. The linear model is
fit on train and evaluated on test. No information from the test set leaks
into training. **No leakage detected.**

**Results plausibility check:**

| Architecture | CV R^2(E) | CV R^2(R_simple) | CV R^2(R_full) | R_full - E |
|---|---|---|---|---|
| MiniLM-L6 | 0.644 | 0.696 | 0.689 | +0.045 |
| mpnet-base | 0.579 | 0.653 | 0.650 | +0.071 |
| multi-qa | 0.652 | 0.706 | 0.704 | +0.052 |

These values are consistent: R_simple slightly beats R_full in CV R^2 on all
3 architectures, consistent with sigma^Df adding marginal overfitting. The
improvements over E (4.5-7.1 percentage points) are substantial for CV R^2
and well above the negligibility threshold of 0.01.

**IMPORTANT OBSERVATION:** R_simple beats R_full in CV R^2 on ALL 3
architectures. This is a stronger signal than the VERDICT_v3.md
acknowledges. The sigma^Df term not only "sometimes overfits" -- it
consistently reduces out-of-sample prediction when added. The Steiger test
says R_full > R_simple in Spearman rho (2/3 arches), but CV R^2 says
R_simple > R_full (3/3 arches). This contradiction suggests the Steiger
advantage of R_full is driven by in-sample rank ordering that does not
translate to out-of-sample linear prediction.

### AIC Nested Model Comparison: Implementation Audit

**Code reviewed:** test_v3_q20.py lines 155-222

The implementation:
1. AIC = n * ln(RSS/n) + 2k where k = n_predictors + 1 (intercept) -- CORRECT
2. Cumulative model building: E -> E + 1/grad_S -> E + 1/grad_S + sigma^Df -- CORRECT
3. Delta AIC relative to best model -- CORRECT

**Concern: In-sample AIC vs out-of-sample.**

The AIC comparison uses in-sample R^2 and AIC, not cross-validated. AIC
includes a penalty term (2k) that partially accounts for overfitting, but
does not guard against it as strongly as CV. The fact that CV R^2 shows
R_simple > R_full while AIC shows the full model winning on 2/3 arches is
a mild inconsistency explained by AIC's weaker overfitting penalty.

**Results detail:**

For multi-qa, sigma^Df improves AIC by only 1.68 (below the dAIC > 2
threshold), correctly flagged as sigma_df_improves_aic = false. This is
internally consistent.

### Bootstrap: Cross-Validation of Steiger

| Architecture | Mean |rho| diff | Bootstrap p | 95% CI |
|---|---|---|---|
| MiniLM-L6 | 0.029 | 0.015 | [0.003, 0.062] |
| mpnet-base | 0.038 | 0.006 | [0.007, 0.076] |
| multi-qa | 0.012 | 0.144 | [-0.011, 0.037] |

Bootstrap confirms Steiger on 2/3 arches (MiniLM and mpnet at p < 0.05).
multi-qa shows p = 0.144, meaning bootstrap does NOT confirm Steiger for
that architecture (Steiger p = 0.009 vs bootstrap p = 0.144). This
discrepancy is explained by Steiger being a parametric test (more powerful
when assumptions hold) vs bootstrap being nonparametric (more conservative).

**This weakens multi-qa's Steiger result somewhat**, but the pre-registered
criteria only require >= 2/3, and both MiniLM and mpnet are confirmed by
both methods.

---

## Critical Assessment of the Verdict

### Is NOT TAUTOLOGICAL the correct verdict?

**By the pre-registered criteria: YES.** The criteria are:
- Steiger p < 0.05 on >= 2/3 arches: MET (3/3)
- CV R^2(R_full) > CV R^2(E) on >= 2/3 arches: MET (3/3)

The criteria were stated before running the test, and the test meets them.
The verdict follows.

### Are the pre-registered criteria appropriate?

**Partially.** The criteria test whether R_full outperforms E, but "not
tautological" has a broader meaning than "statistically better predictor."
The philosophical issue the original audit raised remains:

R_simple = E / grad_S = mean(cosine_sims) / std(cosine_sims)

This IS a signal-to-noise ratio. The test shows that SNR predicts cluster
purity better than signal alone. This is expected and unsurprising. You
would be shocked if SNR did NOT outperform raw signal for measuring
cluster quality. The question is whether this constitutes
"not tautological" or merely "not trivial."

The VERDICT_v3.md honestly acknowledges this in point 4 of the Honest
Assessment section. This is commendable intellectual honesty.

### Does the conclusion align with Q01/Q03?

Per the VERDICT_v3.md note that the v2 verdict of FALSIFIED was incorrect,
and Q01/Q03 also found R > E, the v3 verdict is consistent with the broader
pattern across questions: R adds measurable value over E, primarily through
the grad_S (dispersion) term.

### Are there p-value inflation concerns?

**Yes, but they are addressed.** The main inflation concern is STAT-4 from
the v2 audit: the extreme purity range (0.065 to 1.0) inflates all
correlations. However:

1. The Steiger test compares RELATIVE performance (R_full vs E), so
   absolute correlation inflation cancels out.
2. The CV R^2 test measures out-of-sample prediction, which is not
   inflated by range (though it benefits from it for statistical power).
3. The bootstrap provides a nonparametric check.

The p-values are real in the sense that R_full genuinely outperforms E
on this experimental design. Whether this generalizes to other designs
with narrower purity ranges is unknown.

### New Issues Found

**ISSUE-1 (MODERATE): R_simple consistently beats R_full in CV R^2.**

On ALL 3 architectures, R_simple has higher out-of-sample R^2 than R_full:
- MiniLM: R_simple 0.696 > R_full 0.689
- mpnet: R_simple 0.653 > R_full 0.650
- multi-qa: R_simple 0.706 > R_full 0.704

This means the sigma^Df term consistently hurts out-of-sample linear
prediction while helping in-sample rank correlation (Steiger). The
VERDICT_v3.md notes this but understates it ("slightly outperforms"). The
pattern is 3/3, not occasional.

The practical implication: R_simple = E/grad_S is the better formula.
sigma^Df is dead weight for prediction purposes.

**ISSUE-2 (LOW): Df has near-zero correlation with purity.**

Across all 3 architectures, Df correlates with purity at |rho| < 0.01
(p > 0.93). This means the fractal dimension estimate contributes no
information whatsoever to the prediction task. Since sigma^Df = sigma^(near-zero)
is close to 1.0 for all clusters, the sigma^Df term is mathematically
close to a constant multiplier. Its occasional Steiger significance
(2/3 arches) likely arises from small nonlinear interactions, not from
Df encoding meaningful structure.

**ISSUE-3 (LOW): E_over_std equals R_simple exactly.**

In the results JSON, E_over_std and R_simple have identical rho values
(e.g., both 0.8605 for MiniLM). This is expected since both are E/grad_S
(lines 357 and 365 of the test code compute the same thing). The code
correctly identifies them as the same quantity, confirming R_simple is
literally mean/std of cosine similarities.

**ISSUE-4 (INFORMATIONAL): Fixed seed means results are deterministic
but could be seed-sensitive.**

All randomization uses seed=42. The CV folds, cluster construction, and
bootstrap all use this seed. A robustness check across multiple seeds
would strengthen confidence, but this is standard practice and not a bug.

---

## Verification of Specific Audit Questions

### Q: Was 8e conservation properly handled?

**YES.** Dropped entirely with correct justification. No residual 8e code
or logic in the v3 test.

### Q: Is Steiger's test implemented correctly?

**YES.** Manual spot-check produces consistent Z-statistics. The
Fisher-z + Hotelling correction is the standard formulation.

### Q: Is cross-validated R^2 implemented correctly (no data leakage)?

**YES.** Same fold assignment for all metrics, clean train/test separation,
sklearn r2_score on held-out predictions. No leakage pathway.

### Q: Is the AIC nested model comparison done correctly?

**YES.** Standard AIC formula, cumulative predictor addition, dAIC > 2
threshold for "meaningful improvement" is conventional. Minor concern:
in-sample AIC is weaker than CV, and contradicts CV for sigma^Df.

### Q: Is NOT TAUTOLOGICAL the right framing?

**PARTIALLY.** R adds value over E -- this is empirically demonstrated.
But calling this "not tautological" could mislead readers into thinking R
captures deep structure. What R really does is combine signal (E) with
noise estimation (grad_S) into a signal-to-noise ratio. This is useful
but not surprising.

The VERDICT_v3.md's Honest Assessment section addresses this correctly.
The nuance section in the results JSON also captures it. The framing is
acceptable as long as the caveats are read.

### Q: Are the p-values real or inflated by experimental design?

**REAL for the relative comparison (Steiger), somewhat inflated for
absolute performance.** Steiger compares R_full vs E on the SAME clusters,
so design-driven inflation cancels. The absolute correlation values (0.83-
0.88) are inflated by the extreme purity range, but this is irrelevant
to the tautology question.

### Q: Does the conclusion align with Q01/Q03?

**YES.** Q01 and Q03 found R > E. The v3 Q20 result is consistent: R
significantly outperforms E via Steiger and CV R^2.

### Q: Are there new bugs?

**NO critical or high-severity bugs.** The code is clean, the statistics
are correctly implemented, and the logic follows the pre-registered criteria.
The issues found (ISSUE-1 through ISSUE-4) are methodological observations,
not code bugs.

---

## Severity Summary

| ID | Severity | Category | Description |
|----|----------|----------|-------------|
| ISSUE-1 | MODERATE | Methodology | R_simple beats R_full in CV R^2 on 3/3 arches (sigma^Df hurts out-of-sample) |
| ISSUE-2 | LOW | Methodology | Df has zero correlation with purity (|rho| < 0.01, p > 0.93) |
| ISSUE-3 | LOW | Informational | E_over_std = R_simple confirms R is literally mean/std |
| ISSUE-4 | INFO | Methodology | Single seed (42); robustness across seeds not tested |

**No bugs found in v3 code.**

---

## Final Assessment

### What v3 did well:
1. All v2 audit issues were addressed honestly and correctly
2. Pre-registered criteria were clearly stated and followed
3. Multiple statistical methods (Steiger, CV R^2, AIC, bootstrap, ablation)
   provide convergent evidence
4. The Honest Assessment section is intellectually rigorous and does not
   overstate the findings
5. The nuance about R being an SNR is explicitly stated

### What v3 could improve:
1. The sigma^Df term should be more clearly flagged as consistently harmful
   in CV R^2 (3/3, not "slightly" or "sometimes")
2. A multi-seed robustness check would strengthen the findings
3. The relationship between Steiger (rank-based, says R_full > R_simple on
   2/3) and CV R^2 (linear, says R_simple > R_full on 3/3) deserves more
   discussion -- these tests measure different things

### Bottom line:

The verdict NOT TAUTOLOGICAL is **correctly derived** from the pre-registered
criteria and **supported by the evidence**. The test is well-designed, properly
implemented, and addresses all issues from the v2 audit.

The practical takeaway is: **R_simple = E/grad_S is the formula that works.**
The sigma^Df term is empirically useless for out-of-sample prediction and
should be considered for removal. R works because signal-to-noise ratio
(mean/std of cosine similarities) is a better measure of cluster quality
than raw mean similarity. This is useful but not theoretically profound.

**Audit verdict: SUSTAINED WITH CAVEATS (see ISSUE-1 through ISSUE-4)**
