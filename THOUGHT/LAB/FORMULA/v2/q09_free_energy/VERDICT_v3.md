# Q09 v3 Verdict: Does log(R) correlate with Gaussian NLL?

**Verdict: INCONCLUSIVE**

**Date:** 2026-02-06
**Test version:** v3 (post-audit)
**Runtime:** 2757s (~46 min)

---

## Pre-registered Criteria

| Criterion | Threshold | Result | Pass? |
|---|---|---|---|
| Within-group Pearson \|r\| > 0.7 on >= 2/3 architectures | >= 2/3 arch pass | 3/3 arch pass | PASS |
| Identity residual std < 10% of range(NLL) | < 10% | 23.6-24.6% | **FAIL** |
| Overall \|r\| < 0.5 (falsify) | All arch below | None below (0.953-0.962) | N/A |
| Within-group \|r\| < 0.3 on all arch (falsify) | All arch below | None below | N/A |

**Both CONFIRM conditions required. Within-group passes, but identity fails. Verdict: INCONCLUSIVE.**

---

## What v3 Fixed (from v2 Audit)

1. **Continuous purity** (0.10 to 1.00 in 0.05 steps, 57 clusters) instead of 3 discrete groups -- eliminates the between-group confound that inflated v2's r=0.97
2. **Within-group correlations** computed in 3 bands (low/mid/high purity) -- the make-or-break test from the audit
3. **Honest labeling**: "Gaussian NLL" not "Free Energy" -- this is NOT FEP variational free energy
4. **Fixed ddof inconsistency**: scatter and covariance both use ddof=0
5. **Harder gating task**: pairwise concordance and narrow-band discrimination instead of trivial purity > 0.8 vs < 0.4
6. **Null hypothesis comparison**: tested whether other cluster statistics also correlate with NLL

---

## Key Results

### Overall Correlation (log(R_simple) vs -NLL)

| Architecture | Pearson r | Spearman rho |
|---|---|---|
| all-MiniLM-L6-v2 | 0.9530 | 0.9572 |
| all-mpnet-base-v2 | 0.9620 | 0.9688 |
| multi-qa-MiniLM-L6-cos-v1 | 0.9601 | 0.9675 |

High overall correlations survive the continuous-purity design.

### Within-Group Correlations (THE Critical Test)

| Architecture | Low (0.10-0.35) | Mid (0.40-0.65) | High (0.70-1.00) | Bands > 0.7 |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | 0.689 | 0.823 | 0.849 | 2/3 |
| all-mpnet-base-v2 | 0.826 | 0.814 | 0.893 | 3/3 |
| multi-qa-MiniLM-L6-cos-v1 | 0.805 | 0.854 | 0.877 | 3/3 |

**The within-group correlations are genuinely strong (0.69-0.89).** This is the key finding: the log(R) vs -NLL relationship is NOT purely a group-structure artifact. Even within narrow purity bands, the correlation holds at r = 0.69-0.89.

The one marginal result is MiniLM low-purity band at r=0.689 (just below 0.7), but this is still a strong effect by any standard statistical interpretation.

### Identity Check: log(R) + NLL = constant?

| Architecture | std(log(R)+NLL) | range(NLL) | Residual % |
|---|---|---|---|
| all-MiniLM-L6-v2 | 9.77 | 39.73 | 24.6% |
| all-mpnet-base-v2 | 12.33 | 50.38 | 24.5% |
| multi-qa-MiniLM-L6-cos-v1 | 8.48 | 35.97 | 23.6% |

**The identity log(R) = -NLL + const does NOT hold.** The residual is ~24% of the range across all architectures. log(R) and -NLL are strongly correlated but NOT approximately equal. This is consistent with v2's finding (30% residual) and confirms the audit's assessment.

### Null Hypothesis: Is R Special?

| Architecture | log(R) vs -NLL | trace(cov) vs -NLL | mean_var vs -NLL | mean_euclid vs -NLL |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | 0.953 | -0.966 | -0.966 | -0.956 |
| all-mpnet-base-v2 | 0.962 | -0.971 | -0.971 | -0.962 |
| multi-qa-MiniLM-L6-cos-v1 | 0.960 | -0.968 | -0.968 | -0.951 |

**R is NOT special in its correlation with NLL.** trace(cov), mean variance, and mean Euclidean distance all correlate with -NLL at comparable or slightly higher levels. 3 out of 5 alternative metrics match R's correlation within 0.05. The NLL-R relationship reflects that both measure aspects of cluster geometric tightness, which is expected and not specific to R.

### Harder Gating Task

| Architecture | Concordance (gap >= 0.10) | Hard concordance (gap 0.05-0.15) | Cohen's d (0.65-0.75 vs 0.85-0.95) |
|---|---|---|---|
| all-MiniLM-L6-v2 | 0.970 | 0.738 | 1.774 |
| all-mpnet-base-v2 | 0.952 | 0.713 | 1.351 |
| multi-qa-MiniLM-L6-cos-v1 | 0.964 | 0.735 | 1.720 |

R has strong concordance with purity even at narrow gaps, and large effect sizes for the discrimination task. However, since the null hypothesis test shows other simple metrics perform comparably, this is not evidence for R's unique value.

---

## Honest Assessment

### What the test shows:
1. **log(R) and -Gaussian_NLL are strongly correlated** (r = 0.95-0.96) across continuously varying cluster purity. This is not a 3-group artifact.
2. **The correlation holds WITHIN purity bands** (r = 0.69-0.89). This goes beyond mere "both respond to cluster quality." There is genuine within-group tracking.
3. **However, log(R) != -NLL + const.** The identity fails at ~24% residual. They co-vary but are not linearly equivalent.

### What the test does NOT show:
1. **No connection to the Free Energy Principle.** The NLL here is a Gaussian fit score. FEP free energy F = E_q[log q(z) - log p(x,z)] requires a recognition density and generative model. This test does not compute that.
2. **R is not special.** trace(cov), mean variance, and mean Euclidean distance all correlate with NLL at comparable levels. Any reasonable measure of cluster tightness will show this relationship.
3. **No practical advantage of R over simpler metrics** for quality gating. The null hypothesis comparison undermines claims of R's unique predictive power for NLL.

### What would change the verdict:
- **To CONFIRMED:** Identity residual < 10%, AND R must outperform the null hypothesis (simple stats), AND use FEP-proper free energy
- **To FALSIFIED:** Within-group correlations would need to collapse (they did not)

---

## Conclusion

The log(R) vs -Gaussian_NLL correlation is real and goes beyond group-structure artifacts (within-group r = 0.69-0.89). However, the exact identity log(R) = -NLL + const fails (24% residual), the "free energy" tested is not FEP-standard, and R does not outperform simpler cluster statistics in predicting NLL. The finding should be stated as: **"R, like other cluster tightness metrics, covaries strongly with Gaussian NLL"** -- not as evidence for a deep connection between R and the Free Energy Principle.
