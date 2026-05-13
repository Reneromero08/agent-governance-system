# Q15 v3 Adversarial Audit Report

**Auditor**: Adversarial audit agent (Round 2)
**Date**: 2026-02-06
**Scope**: Code, statistics, methodology, and verdict for Q15 v3 (statistical quality correlation)
**Verdict under review**: FALSIFIED
**Prior audit**: AUDIT.md (v2, which found 13 issues incl. 3 critical)

---

## Audit Fix Verification

### Was ESS (trivially = E) removed? (BUG-1 from v2 audit)

**YES -- FULLY FIXED.** ESS does not appear anywhere in test_v3_q15.py. The quality_keys are now `["inv_trace_cov", "bootstrap_mean_prec", "silhouette"]` (line 499), and the nontrivial_keys used for verdict criteria are `["bootstrap_mean_prec", "silhouette"]` (line 501). inv_trace_cov is included for completeness only and explicitly flagged as trivially related to E (line 501 comment, line 627-628 trivial check).

Confirmed in results: E vs inv_trace_cov Spearman rho = 1.000 on all three architectures, exactly as expected. This metric is correctly excluded from the verdict criteria.

**Status**: RESOLVED.

### Were "Bayesian" labels honestly replaced with "frequentist"? (METH-1 from v2 audit)

**YES -- FULLY FIXED.** A thorough search of the code and verdict document shows:

- The docstring (lines 1-37) explicitly states "honest reframing" and "no Bayesian claims."
- `compute_bootstrap_mean_precision` (line 257) has a docstring explicitly stating: "This is a FREQUENTIST resampling quantity... It is NOT a Bayesian posterior -- there is no prior, no likelihood model, no Bayes' rule." (lines 261-266).
- VERDICT_v3.md states at the top: "v3 drops all Bayesian claims" and throughout uses "frequentist" correctly.
- The question has been reframed from "Does R have a valid Bayesian interpretation?" to "Does R correlate with standard statistical measures of cluster quality better than E alone?" This is an honest reframing.

**Status**: RESOLVED.

### Was Steiger's test added and implemented correctly? (STAT-1 from v2 audit)

**YES -- ADDED. IMPLEMENTATION HAS A CONCERN.**

Steiger's test is implemented at lines 155-196. The implementation uses:

1. Fisher z-transform of the two correlations being compared (correct).
2. The denominator formula: `sqrt(2 * (1 - r_xy) / ((n - 3) * (1 + r_xy)))` (line 188).

This is a simplified version of Steiger (1980). The full Steiger formula accounts for the determinant of the 3x3 correlation matrix and uses a more complex variance estimate. The simplified version used here (sometimes called the "Hotelling-Williams" approximation) is:

```
Z = (z_xz - z_yz) * sqrt((n-3)(1+r_xy) / (2 * (1 - r_xy^2 - r_xz^2 - r_yz^2 + 2*r_xz*r_yz*r_xy) * ... ))
```

The code computes `h = (1 - r_xy) / (2 * (1 - r_mean_sq))` at line 185 but never uses it in the final Z computation. The variable `r_det` at line 181 is computed but also never used. These are vestigial computations from an attempt to use the full formula that was abandoned in favor of the simplified version.

**CONCERN (STAT-V3-1, LOW)**: The unused `r_det` and `h` variables suggest the implementer started with the full Steiger formula but fell back to a simpler approximation. The simplified formula used is valid and conservative (it tends to produce larger p-values than the full formula). Given the enormous Z-statistics observed (Z = -10 to -26 for Test 1, Z = +11 for Test 3), the choice of approximation is immaterial -- any reasonable implementation would yield p << 0.001.

However, there is a subtle issue: the code takes `abs()` of all three correlations before passing them to `steiger_test` (lines 598-599). Steiger's test is designed for signed correlations. Taking absolute values changes the test. If r_xz and r_yz have the same sign (which they do here -- all positive), then `abs()` is a no-op and the results are correct. But the code should document this assumption.

**Status**: RESOLVED (with minor code hygiene concern).

### Was the intensive property test fixed with domain-weighted CV? (BUG-3 from v2 audit)

**YES -- FIXED.** Lines 789-797 compute domain-weighted CV as the arithmetic mean of text_mean_cv and housing_mean_cv, giving equal weight to each domain. Result: (0.400 + 0.024) / 2 = 0.212. The raw unweighted mean (0.110) is also reported for comparison.

**Status**: RESOLVED.

### Was the trivially-easy gating test redesigned? (BUG-4 from v2 audit)

**YES -- FULLY REDESIGNED.** Test 3 now predicts continuous purity (Spearman correlation with purity) instead of binary gating (F1 score). This provides genuine discriminative power between metrics.

**Status**: RESOLVED.

### Were partial correlations added? (METH-2 from v2 audit)

**YES -- ADDED.** Partial Spearman correlations rho(R_full, measure | E) are computed at lines 199-235 and reported for all nontrivial measures across all architectures.

**Status**: RESOLVED.

### Were both Spearman and Pearson reported? (STAT-2 from v2 audit)

**YES.** Both correlation types are computed and reported throughout.

**Status**: RESOLVED.

### Were confidence intervals added? (STAT-3 from v2 audit)

**YES.** Bootstrap 95% CIs are computed via bootstrap_correlation_ci() for all key correlations.

**Status**: RESOLVED.

---

## Summary: All 9 v2 audit items addressed

| v2 Audit Item | Status |
|---|---|
| BUG-1: ESS trivially = E | RESOLVED (removed) |
| BUG-2: Bootstrap precision mislabeled | RESOLVED (honest labeling) |
| BUG-3: Biased CV aggregation | RESOLVED (domain-weighted) |
| BUG-4: Trivially easy gating | RESOLVED (continuous purity) |
| STAT-1: No significance test E > R | RESOLVED (Steiger's test) |
| STAT-2: Only Spearman | RESOLVED (both reported) |
| STAT-3: No confidence intervals | RESOLVED (bootstrap CIs) |
| METH-1: Fake Bayesian labels | RESOLVED (honest frequentist) |
| METH-2: No partial correlations | RESOLVED (computed throughout) |

---

## New Issues in v3

### STAT-V3-1 (LOW): Dead code in Steiger's test implementation

As noted above, `r_det` (line 181) and `h` (line 185) are computed but never used. This is harmless but suggests incomplete cleanup.

**Severity**: LOW.

### STAT-V3-2 (MODERATE): Steiger's test uses abs() of correlations, which is correct here but fragile

Lines 598-599 in Test 1 and lines 900-901 in Test 3 take `abs()` of all three correlations before passing to `steiger_test()`. Steiger's test compares signed correlations. Taking absolute values is valid when all three correlations have the same sign (which they do: all are positive, since R_full, E, and the quality measures are positively correlated). But the code does not check this assumption.

If in some future scenario the correlations had mixed signs, the `abs()` would silently produce incorrect results.

**Severity**: LOW (correct for current data, fragile for reuse).

### METH-V3-1 (MODERATE): bootstrap_mean_prec is still dominated by E

The v2 audit flagged this as BUG-2/MODERATE: bootstrap mean precision is approximately n/trace(Cov(X)), which for unit-norm embeddings is closely related to E. The v3 test honestly acknowledges the frequentist nature of this metric but does not address the near-collinearity with E.

Evidence from results:
- E vs bootstrap_mean_prec Spearman: 0.979, 0.981, 0.978 across architectures

These are extremely high, confirming that bootstrap_mean_prec is nearly a monotone function of E for this data. When the "target variable" is 97-98% determined by E, any test comparing R_full vs E for predicting that target is essentially testing whether R_full tracks E, not whether R_full captures independent statistical structure.

This is honestly acknowledged in the VERDICT_v3.md document but should still be flagged as a methodological limitation: one of the two "nontrivial" measures is in fact nearly trivially related to E.

**Severity**: MODERATE. The test honestly reports this limitation, and silhouette score provides a genuinely independent measure.

### METH-V3-2 (LOW): Silhouette computation only valid for 70/90 clusters

From the results, silhouette is computed for only 70 of 90 clusters (n_valid: 70). The 20 missing clusters are the "pure" type clusters -- because `compute_silhouette_approx` requires at least 2 unique labels (line 293-295), and pure clusters have only one label, returning NaN.

This means all silhouette-based comparisons are on a biased subset: the 70 clusters that are NOT pure. Since pure clusters have purity=1.0 and high E values, removing them eliminates one end of the purity spectrum from the silhouette analysis. This truncation could affect the silhouette-based Steiger tests.

However, this is inherent to the silhouette metric (it requires multi-label data) and is not a code bug. The test correctly handles NaN values.

**Severity**: LOW (methodological limitation, not a bug).

---

## Verdict Assessment: Should this be FALSIFIED or INCONCLUSIVE?

This is the central question of this audit.

### What the pre-registered criteria say

The criteria (lines 33-36 of the code docstring):

> CONFIRMED if R significantly outperforms E (Steiger p<0.05) in correlation
>   with >=2/3 statistical quality measures on >=2/3 architectures.
> FALSIFIED if E significantly outperforms R on all measures on all architectures.
> INCONCLUSIVE otherwise.

The code at line 1111 implements:
```python
elif criterion_a_falsify or (criterion_b_falsify and criterion_c_falsify):
    overall = "FALSIFIED"
```

Where `criterion_a_falsify` = "E wins on ALL nontrivial measures on ALL architectures."

### Does criterion_a_falsify correctly apply?

From the results:
- **bootstrap_mean_prec**: E significantly wins on 3/3 architectures (Steiger Z = -20.4, -26.3, -16.4, all p < 0.0001)
- **silhouette**: E significantly wins on 3/3 architectures (Steiger Z = -10.4, -10.2, -11.8, all p < 0.0001)

E wins on ALL 2 measures on ALL 3 architectures. `criterion_a_falsify = True`. This alone triggers FALSIFIED.

### But R_full beats E for purity prediction on 2/3 architectures

Test 3 shows R_full significantly outperforms E for continuous purity prediction on 2/3 architectures (Steiger Z = +11.4 and +11.4, p < 0.0001 on MiniLM and mpnet; Z = +0.74, p = 0.46 on multi-qa). `criterion_c_confirm = True`.

**This is a mixed result.** R_full is worse than E at correlating with frequentist quality measures, but better than E at predicting cluster purity.

### Is FALSIFIED the correct verdict, or should it be INCONCLUSIVE?

**ARGUMENT FOR FALSIFIED (as stated):**

The pre-registered criteria are clear and unambiguous. The FALSIFIED rule says: "if E significantly outperforms R on ALL measures on ALL architectures." This is satisfied. The purity prediction result is a separate criterion (C), and the decision rule says FALSIFIED can be triggered by criterion A alone (`criterion_a_falsify OR (B_falsify AND C_falsify)`). The pre-registration did not require ALL criteria to point in the same direction for FALSIFIED.

**ARGUMENT FOR INCONCLUSIVE:**

The pre-registered FALSIFIED criteria were arguably poorly designed. They allow FALSIFIED to trigger from criterion A alone, even when criterion C shows the opposite direction. A result where "E beats R on quality measures, but R beats E on purity prediction" is genuinely mixed. The natural reading of "FALSIFIED" is "the hypothesis is definitively wrong," but R_full demonstrably adds value for purity prediction, meaning the formula is not worthless.

Furthermore, the question being asked is "Does R correlate with standard statistical measures better than E?" The purity prediction in Test 3 IS a standard statistical measure (purity is a well-known clustering quality metric). If we included purity as a "quality measure" alongside bootstrap_mean_prec and silhouette, the result would be mixed, not FALSIFIED.

**MY ASSESSMENT: FALSIFIED is defensible but aggressive. INCONCLUSIVE would be more intellectually honest.**

The mechanical application of the pre-registered decision rule produces FALSIFIED. The code correctly implements its own criteria. But the criteria contain a structural asymmetry: FALSIFIED can trigger from criterion A alone, while CONFIRMED requires all three criteria (A AND B AND C). This asymmetry was baked into the pre-registration and biases toward FALSIFIED when results are mixed.

The VERDICT_v3.md document does a good job of acknowledging the nuance in its "Honest Interpretation" section (lines 155-165), noting that "R_full does contain useful information" and suggesting a follow-up question. This partially mitigates the overly blunt verdict.

**Recommendation**: The verdict FALSIFIED is acceptable if accompanied by the existing nuanced interpretation. A stronger approach would be to label this INCONCLUSIVE with an explanation that criterion A triggers falsification but criterion C shows a genuine positive finding.

---

## Is the Q15 result contradictory with Q01/Q03/Q20?

The team lead asked whether Q15 finding "E beats R" contradicts Q01/Q03/Q20 finding "R beats E with purity ground truth."

**This is NOT contradictory.** The key distinction is:

1. **Q15 Test 1 (quality measures)**: Asks whether R or E correlates better with bootstrap_mean_prec and silhouette. These are statistical properties of the embedding distribution. E naturally correlates more strongly with these because both bootstrap_mean_prec (~n/trace(Cov)) and silhouette (pairwise distance-based) are closely related to pairwise cosine similarity, which IS E. Adding noise via grad_S and sigma^Df degrades the correlation. **This measures something fundamentally different from purity.**

2. **Q15 Test 3 / Q01 / Q03 / Q20 (purity)**: Asks whether R or E predicts cluster purity. Purity is a LABEL-BASED quality measure (what fraction of documents belong to the dominant category). This depends on the semantic structure of the embedding space, not just the geometry. R_full's sigma^Df term captures structural information (compression ratio, fractal dimension) that is related to how well the cluster captures a coherent semantic group. **This is consistent with Q01/Q03/Q20.**

The resolution: **E is better at predicting geometric/distributional statistics of embeddings. R_full is better at predicting semantic/label-based quality.** These are different targets, and there is no contradiction.

The partial correlation evidence supports this interpretation:
- rho(R_full, silhouette | E) = **-0.51** (significantly negative on all architectures). The sigma^Df scaling HURTS geometric quality prediction.
- rho(R_full, purity) > rho(E, purity) on 2/3 architectures. The sigma^Df scaling HELPS semantic quality prediction.

---

## Are the "frequentist quality measures" appropriate targets for this question?

**PARTIALLY.** The v3 reframing honestly asks whether R correlates with statistical quality measures. But the specific measures chosen (inv_trace_cov, bootstrap_mean_prec, silhouette) are heavily weighted toward geometric/distributional properties:

- **inv_trace_cov**: Pure geometry. Trivially = E for unit-norm embeddings.
- **bootstrap_mean_prec**: Centroid estimation precision. Dominated by E (rho > 0.97 with E).
- **silhouette**: Clustering quality via pairwise distances. More independent but still geometric.

None of these measures capture semantic quality -- the degree to which clusters represent meaningful, coherent groups. If the test had included purity (which it does in Test 3) or adjusted Rand index or normalized mutual information as a "quality measure" in Test 1, the results would look different.

The choice of exclusively geometric quality measures in Test 1 structurally favors E (which is itself a pure geometric measure) over R_full (which incorporates structural complexity measures).

**Severity**: MODERATE. The measures are legitimate frequentist statistics, but the selection is not representative of all aspects of "statistical quality."

---

## Negative Partial Correlation: A Genuine Finding

The most scientifically interesting result in v3 is the consistently negative partial correlation of R_full with silhouette after controlling for E:

| Architecture | Partial rho(R_full, silhouette | E) | p |
|---|---|---|
| all-MiniLM-L6-v2 | -0.509 | <0.0001 |
| all-mpnet-base-v2 | -0.509 | <0.0001 |
| multi-qa-MiniLM-L6-cos-v1 | -0.545 | <0.0001 |

This means: **after controlling for E, the additional formula components (grad_S, sigma, Df) are ANTI-correlated with geometric clustering quality.** Clusters where the formula adds a large sigma^Df correction tend to have LOWER silhouette scores.

This is a real finding and supports the FALSIFIED verdict for the narrow question "Does R correlate better than E with geometric quality measures?" But it also reinforces the interpretation that R's additional components capture something DIFFERENT from geometric quality -- possibly semantic coherence, which is why R_full beats E for purity prediction.

---

## Summary

| Category | Count | Severity |
|---|---|---|
| v2 audit items resolved | 9/9 | -- |
| New code issues | 2 | 1 LOW, 1 LOW |
| New methodological issues | 2 | 1 MODERATE, 1 LOW |
| Verdict assessment issue | 1 | see below |

### New issues summary

| ID | Description | Severity |
|---|---|---|
| STAT-V3-1 | Dead code (r_det, h) in steiger_test | LOW |
| STAT-V3-2 | abs() of correlations correct but fragile | LOW |
| METH-V3-1 | bootstrap_mean_prec still ~97% determined by E | MODERATE |
| METH-V3-2 | Silhouette only valid for 70/90 clusters | LOW |

### Verdict Assessment

**FALSIFIED is mechanically correct** per the pre-registered decision rules. Criterion A (E significantly outperforms R on ALL nontrivial measures on ALL architectures) is unambiguously satisfied with enormous effect sizes (Steiger Z = -10 to -26).

**However, the result is genuinely mixed.** R_full significantly outperforms E for purity prediction on 2/3 architectures (criterion C confirms). The pre-registered decision rules have a structural asymmetry that allows criterion A alone to trigger FALSIFIED, even when criterion C shows the opposite. This asymmetry makes FALSIFIED technically correct but potentially misleading.

**The VERDICT_v3.md document handles this well** by including a nuanced "Honest Interpretation" section that acknowledges R_full's genuine purity prediction advantage and suggests a follow-up question.

### Recommendations

1. **Accept FALSIFIED** for the narrow question "Does R correlate with standard frequentist quality measures better than E?" This is clearly answered: No.

2. **Note for the record** that this does not mean R_full is worthless. The purity prediction result is genuine and scientifically interesting. The formula's additional components capture semantic structure that geometric measures miss.

3. **Minor code cleanup**: Remove dead variables `r_det` and `h` from `steiger_test()`.

4. **No re-run required.** The code is correct, the statistics are sound, and the verdict follows from the pre-registered criteria.

### Overall Quality Grade: GOOD

This is a substantial improvement over v2. All 9 audit items were addressed. The honest reframing, Steiger's tests, partial correlations, and continuous purity prediction are all well-implemented. The remaining issues are minor. The FALSIFIED verdict is defensible, and the interpretive text is honest about the nuances.
