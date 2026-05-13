# Q20 v2 Audit: Is R = (E/grad_S) * sigma^Df Tautological?

**Auditor:** Adversarial Audit Agent (Opus 4.6)
**Date:** 2026-02-06
**Verdict under review:** FALSIFIED
**Audit conclusion:** VERDICT SHOULD BE CHANGED TO INCONCLUSIVE -- the test contains a critical conservation-law bug and an arguably over-strict threshold, while also containing genuinely sound methodology in other areas.

---

## Code Bugs Found

### BUG-1 (CRITICAL): 8e Conservation Tests the WRONG Quantity

**The claimed conservation law is Df * alpha = 8e. The test computes PR * alpha instead.**

Evidence chain:

1. The README.md (line 5) states the hypothesis: "the conservation law Df * alpha = 8e"
2. The GLOSSARY.md (line 125-128) explicitly defines: "Df * alpha -- Spectral Product. v1 claim: Df * alpha = 8e (conservation law)."
3. The README.md Test 2 (line 63) says: "Measure Df * alpha for each."
4. But the test code's `compute_8e_metrics()` function (lines 205-233) computes **PR * alpha**, NOT Df * alpha.
5. In `formula.py`, `compute_Df()` (lines 108-148) computes Df = 2/alpha (via power-law fit of eigenvalue decay). This is NOT the same as PR.
6. The test's `compute_8e_metrics()` computes PR = (sum(eigenvalues))^2 / sum(eigenvalues^2) -- this is the **participation ratio**, identical to the `compute_sigma()` numerator (before dividing by d).

**The discrepancy:**
- `Df` in formula.py = 2/alpha (a fractal dimension estimate via box-counting relationship)
- `PR` in the 8e test = participation ratio of eigenvalues
- These are fundamentally different quantities. Df is typically O(1) (values like 0.18 in the data). PR is O(100) (values like 78-120 in the data).
- Df * alpha = (2/alpha) * alpha = 2. Always. By mathematical identity. So the conservation law "Df * alpha = 8e" as formulated with v2 definitions is trivially false -- it equals 2, not 21.7.

**This means:** The 8e conservation test is testing an internally inconsistent quantity. The v1 definition of Df was the participation ratio itself. The v2 definition is 2/alpha. The test uses v1's definition (PR) while claiming to test v2's conservation law. The 317.8% error reported for the conservation law is real for PR*alpha, but this is not the quantity the README's conservation law specifies.

**Impact on verdict:** The entire 8e conservation failure -- which accounts for one of the two falsification triggers -- may be a test bug rather than a formula failure. However, as noted above, with v2 definitions Df*alpha = 2 trivially, which means the conservation law itself is incoherent under v2 definitions.

### BUG-2 (MODERATE): Inconsistent alpha Computation Between Formula and 8e Test

The `compute_Df()` function in formula.py fits the power law on ALL positive eigenvalues (line 135-141). The `compute_8e_metrics()` function in the test fits alpha on only the TOP HALF of eigenvalues (line 219: `half = len(eigenvalues) // 2`). These will produce systematically different alpha values for the same data. This means alpha in the 8e test is not the same alpha used to compute Df in the formula.

### BUG-3 (MINOR): Glossary vs Implementation Mismatch for Df

The GLOSSARY.md (lines 76-87) defines:
```
Df = participation ratio of eigenvalues = (sum(lambda_i))^2 / sum(lambda_i^2)
```

But formula.py's `compute_Df()` (lines 108-148) implements:
```
Df = 2/alpha where alpha is the power-law exponent
```

These are two completely different definitions of Df sharing the same symbol. The glossary says Df is the participation ratio; the code says Df is 2/alpha. This naming collision is the root cause of BUG-1.

### BUG-4 (MINOR): Sigma Definition Mismatch

The GLOSSARY.md (lines 62-72) defines sigma as "V_eff / V_total" (vocabulary ratio). The code in formula.py (lines 72-105) computes sigma as "PR / d" (participation ratio normalized by ambient dimensionality). These are completely different quantities. The glossary definition is about token distributions; the code definition is about eigenvalue spectra.

---

## Statistical Errors

### STAT-1 (HIGH): The 0.05 Threshold Is Arguably Too Strict for High-Correlation Regime

When comparing rho values of ~0.90 vs ~0.92, a 0.05 absolute improvement threshold is extremely demanding. In the high-correlation regime (rho > 0.85), each additional 0.01 of rho is exponentially harder to achieve because most of the variance is already explained. The standard approach in psychometrics and NLP benchmarks is to use either:

- **Percentage improvement:** 0.02/0.90 = 2.2% improvement, which is modest but real
- **Fisher z-transform comparison:** The proper way to compare two correlation coefficients
- **Bootstrap confidence intervals:** Which the test actually does (p-values of 0.053, 0.053, 0.029)

The bootstrap results are borderline significant (2 of 3 at p=0.053, just barely missing 0.05). This is not strong falsification -- it is genuine ambiguity. In many scientific fields, a consistent 2% improvement in correlation across 3 independent datasets would be considered meaningful.

**The pre-registered threshold of 0.05 absolute rho improvement was not justified in the README.** The README says "by at least 5% in correlation" -- this is ambiguous between 5% absolute (0.05 rho units) and 5% relative. If relative, the test would pass: 0.02/0.90 = 2.2% > 5%? No, but the margin of 0.02 is on the wrong side of 0.05 either way.

### STAT-2 (MODERATE): Bootstrap p-values of 0.053 Are Not Decisive

Two of three architectures show p = 0.053 for the bootstrap comparison of R_full vs E. This is p = 0.053, not p = 0.5. It means:
- In ~5.3% of bootstrap resamples, E performs as well as or better than R_full
- In ~94.7% of bootstrap resamples, R_full outperforms E
- The 95% CI for the difference is [-0.004, 0.053] -- crossing zero by a hair

This is the definition of borderline. Calling it "non-significant" is technically correct at alpha=0.05, but calling it "falsified" is an overstatement. The proper characterization is "insufficient statistical power to distinguish."

### STAT-3 (LOW): 60 Clusters May Be Insufficient for Detecting Small Differences

With n=60 clusters, detecting a difference of 0.02 in Spearman rho at p<0.05 requires substantial statistical power. A power analysis would clarify whether 60 clusters can reliably detect effects of this size. The test may be underpowered for the question being asked.

### STAT-4 (INFORMATIONAL): Correlation Values Are Inflated by Design

The test design creates clusters with purity ranging from 0.05 (random) to 1.0 (pure). This extreme range inflates all correlations. Any reasonable monotonic function of embedding similarity will achieve rho > 0.8 when the ground truth spans from near-zero to 1.0. This explains why E alone achieves rho = 0.90 -- it is not because E is exceptionally good, but because the ground truth has extreme dynamic range. The proper comparison is not "how high is rho" but "which metric has higher rho" -- which is what the test does.

---

## Methodological Issues

### METH-1 (HIGH): The v2 Test Deviates Substantially from the Pre-Registered Plan

The README.md specified 5 tests:
1. Component comparison using STS-B/MTEB with human judgments
2. 8e conservation on in-distribution data across 5 modalities (audio, image, text, code, multilingual)
3. Pre-registered novel domain prediction (CLAP, BiomedBERT, ESM-2)
4. Ablation study on functional forms
5. Axiomatic-level tautology test

The actual v2 test implemented:
1. Component comparison using 20 Newsgroups with cluster purity (NOT human judgments)
2. 8e conservation on text only (NOT 5 modalities)
3. Novel predictions (weaker than pre-registered: no novel domain prediction)
4. Ablation study (implemented as specified)
5. Axiomatic test NOT implemented

The test is more limited than planned. However, the methodology that WAS implemented is sound: 20 Newsgroups with controlled purity is a legitimate and arguably more rigorous ground truth than human similarity judgments (which have their own noise).

### METH-2 (HIGH): Tautology Question Is Not Well-Served by Correlation Comparison

The core tautology concern is: "Is R just a dressed-up signal-to-noise ratio?" The test checks whether R outperforms its components in predicting purity. But this is only a necessary condition, not a sufficient one. Even if R outperforms E alone, R could still be "just" a signal-to-noise ratio that happens to be slightly more informative than E alone. The deeper tautology question -- whether the formula captures structure that a naive E/std would not -- requires a different kind of test (e.g., comparing R to mean/std directly, or testing on tasks where SNR should NOT predict the outcome).

### METH-3 (MODERATE): 8e Conservation Is a Separate Claim from Tautology

The 8e conservation law and the tautology question are logically independent. R could be non-tautological even if 8e conservation fails (the formula could add value without the conservation law holding). Conversely, 8e could hold even if R is tautological (the conservation law could be a mathematical identity). Mixing these two tests into a single verdict conflates two separate questions.

### METH-4 (LOW): The "R_full_outperforms_all_by_005" Criterion Requires Beating ALL Components

The falsification criterion `outperforms_count == 0` is triggered because R_full must beat ALL base components by 0.05 in ALL architectures. R_full actually beats 5 of 6 base components by more than 0.05 -- it only fails against E. The criterion is testing "does R add value over the BEST component" rather than "does R add value over its component parts collectively," which is a stricter and arguably less relevant question.

---

## Verdict Assessment

### Is FALSIFIED Correct?

**No. The correct verdict should be INCONCLUSIVE.** Here is why:

The code's falsification logic (lines 649-653) triggers FALSIFIED when:
```python
falsify = (
    (outperforms_count == 0) or
    (not eight_e_within_100) or
    (best_ablation_count == 0 and novel_pass_count == 0)
)
```

Both triggers that fired are problematic:

1. **`outperforms_count == 0`**: This is true because R_full does not beat E by 0.05 in any architecture. But R_full DOES beat E consistently (by 0.017-0.027) across all three architectures. The threshold is arbitrary and arguably too strict for the high-correlation regime. The bootstrap shows borderline significance (p=0.029-0.053). This should be INCONCLUSIVE, not FALSIFIED.

2. **`not eight_e_within_100`**: This is true because PR*alpha has 317.8% mean error. But as documented in BUG-1, the test computes the WRONG quantity. The conservation law specifies Df*alpha, not PR*alpha. With v2 definitions, Df*alpha = 2 trivially. This is a test bug, not a formula failure. The conservation law is either incoherent (under v2 definitions) or untested (under v1 definitions with v2 data).

Meanwhile, several indicators point AWAY from falsification:
- R_full is the best single metric in 2 of 3 architectures
- R_full is the best ablation form in 3 of 3 architectures
- R_full has positive correlation with purity in 3 of 3 architectures
- Novel predictions pass >= 2/3 in all architectures
- R_full consistently (if modestly) outperforms E alone

### What the Verdict Should Be

**INCONCLUSIVE with specific findings:**

1. **R_full is NOT a pure tautology.** It consistently outperforms E alone, though by modest margins (0.017-0.027 in rho). The improvement is borderline significant (p=0.029-0.053).

2. **The sigma^Df term adds negligible value.** R_simple (E/grad_S) performs nearly identically to R_full. The fractal scaling term is decorative or slightly harmful.

3. **R is effectively E/grad_S = mean/std of cosine similarities.** This IS a signal-to-noise ratio. Whether a well-constructed SNR is "tautological" or "explanatory" is a philosophical question, not an empirical one.

4. **The 8e conservation law is untestable under current definitions** due to the Df naming collision. Under v2 definitions, Df*alpha = 2 trivially. Under v1 definitions, PR*alpha ranges from 71-124 (not 21.7).

5. **The 0.05 threshold was pre-registered but not justified.** A more standard analysis (bootstrap CI, Fisher z-test) would characterize this as borderline evidence for modest improvement.

---

## Issues Requiring Resolution

### MUST FIX

1. **Df naming collision (BUG-1, BUG-3).** The glossary, formula.py, and the 8e test all use "Df" for different quantities. This must be resolved before any conservation law can be tested.

2. **Sigma definition mismatch (BUG-4).** The glossary says sigma = V_eff/V_total (vocabulary). The code says sigma = PR/d (eigenvalue). These are different.

3. **Alpha fitting inconsistency (BUG-2).** The formula and the 8e test fit alpha on different portions of the eigenvalue spectrum.

4. **Verdict should be INCONCLUSIVE.** The FALSIFIED verdict is not supported when one of the two falsification triggers is a test bug and the other is based on an unjustified threshold.

### SHOULD FIX

5. **Justify or revise the 0.05 threshold.** Either provide a power analysis showing 0.05 is the minimum detectable effect size, or use a more appropriate comparison method (Fisher z-test, bootstrap CI).

6. **Separate tautology from conservation law.** These are independent questions and should have independent verdicts.

7. **Implement the axiomatic tautology test.** This was in the pre-registered plan but not implemented. It is arguably the most direct test of tautology.

### NICE TO HAVE

8. **Test on human judgments (STS-B).** The pre-registered plan called for this. Cluster purity is a valid but different ground truth.

9. **Multi-modality 8e test.** The pre-registered plan called for 5 modalities. Only text was tested.

---

## What Would Change the Verdict

### To CONFIRMED:
- Fix the Df definition so all code uses one consistent definition
- Show R_full outperforms E alone with p < 0.01 (need more clusters or a larger dataset)
- Show the sigma^Df term contributes measurably (it currently does not)
- Show R outperforms a naive mean/std on a task where SNR should NOT predict the outcome
- Demonstrate 8e conservation under a consistent set of definitions (this may be impossible)

### To FALSIFIED (legitimately):
- Show that R_full does NOT outperform E alone even with adequate statistical power (e.g., 500+ clusters, p > 0.1)
- Show that the sigma^Df term consistently HURTS performance (currently mixed -- helps in 2/3 architectures, hurts in 1/3)
- Show that R = mean/std of cosine similarities, and this is the ONLY reason R works (i.e., any SNR metric works equally well)

### The honest current state:
R_full = (E / grad_S) * sigma^Df works well (rho ~0.92 with purity) primarily because E/grad_S is a good signal-to-noise ratio for cosine similarities. The sigma^Df term is decorative. Whether this constitutes "tautology" depends on whether you consider a well-constructed SNR to be tautological. The 8e conservation law cannot be assessed due to definitional inconsistencies.

---

## Severity Summary

| ID | Severity | Category | Description |
|----|----------|----------|-------------|
| BUG-1 | CRITICAL | Code | 8e test computes PR*alpha, not Df*alpha as specified |
| BUG-2 | MODERATE | Code | Alpha fit uses different eigenvalue range in formula vs 8e test |
| BUG-3 | MINOR | Code | Glossary defines Df as PR; formula.py defines Df as 2/alpha |
| BUG-4 | MINOR | Code | Glossary defines sigma as vocabulary ratio; code uses PR/d |
| STAT-1 | HIGH | Statistics | 0.05 absolute rho threshold unjustified in high-correlation regime |
| STAT-2 | MODERATE | Statistics | Bootstrap p=0.053 is borderline, not decisive |
| STAT-3 | LOW | Statistics | 60 clusters may be underpowered |
| STAT-4 | INFO | Statistics | Correlation inflation from extreme purity range |
| METH-1 | HIGH | Methodology | Test deviates from pre-registered plan (fewer modalities, different ground truth) |
| METH-2 | HIGH | Methodology | Correlation comparison is necessary but not sufficient for tautology |
| METH-3 | MODERATE | Methodology | 8e conservation and tautology are logically independent |
| METH-4 | LOW | Methodology | "Beat ALL components by 0.05" is stricter than necessary |

**Bottom line:** The FALSIFIED verdict is partially based on a test bug (BUG-1: wrong conservation quantity) and a borderline statistical call (STAT-1: p=0.053 at an unjustified 0.05 rho threshold). The correct verdict is INCONCLUSIVE. R_full shows modest but consistent improvement over E alone, the sigma^Df term is decorative, and the 8e conservation law is untestable under current definitions.
