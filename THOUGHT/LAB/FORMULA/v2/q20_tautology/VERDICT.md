# Q20 Verdict

## Result: INCONCLUSIVE (leans toward FALSIFIED)

The automated criteria give 2/3 PASS (component superiority, ablation) and 1/3 FAIL (novel predictions). By strict pre-registered criteria this is INCONCLUSIVE. However, see "Honest Assessment" below -- the passing criteria are themselves ambiguous, and the 8e conservation law failed catastrophically.

## Component Comparison (THE KEY RESULT)

| Metric | Spearman rho with human scores | p-value | |rho| |
|--------|-------------------------------|---------|-------|
| **R_full** | **-0.3779** | **0.025** | **0.378** |
| E alone | -0.0283 | 0.872 | 0.028 |
| 1/grad_S | -0.2549 | 0.139 | 0.255 |
| sigma^Df | +0.0420 | 0.811 | 0.042 |
| E/grad_S | -0.2431 | 0.159 | 0.243 |
| E * sigma^Df | -0.1095 | 0.531 | 0.110 |
| SNR | -0.1339 | 0.443 | 0.134 |

**Margins over components (criterion: >= 0.05 each):**

| vs Component | Margin in |rho| | Status |
|---|---|---|
| vs E | +0.350 | PASS |
| vs 1/grad_S | +0.123 | PASS |
| vs sigma^Df | +0.336 | PASS |
| vs E/grad_S | +0.135 | PASS |
| vs E*sigma^Df | +0.268 | PASS |
| vs SNR | +0.244 | PASS |

R_full outperforms ALL individual components and pairwise combos by >= 0.05. **This criterion PASSES.**

**CRITICAL CAVEAT**: The correlation is **negative** (rho = -0.378). Higher R_full values correspond to **lower** human similarity scores. If R is supposed to measure "truth" or "meaning quality," it measures the opposite on this benchmark. The pre-registered criteria asked only whether R outperforms its parts in absolute correlation, which it does -- but the direction is inverted.

## 8e Conservation

The 8e conservation law (Df * alpha = 8e = 21.746) was tested using v1 definitions (PR = raw participation ratio, alpha fit on top half of eigenvalues).

| Model | alpha (top half) | PR | PR * alpha | Error vs 8e |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | 0.748 | 104.70 | 78.31 | 260.1% |
| all-mpnet-base-v2 | 1.110 | 128.07 | 142.11 | 553.5% |
| multi-qa-MiniLM-L6-cos-v1 | 0.947 | 49.71 | 47.08 | 116.5% |
| Random matrix (mean of 5) | 0.203 | 322.04 | 65.48 | 201.1% |

**8e conservation FAILS catastrophically.** All three text embedding models produce PR*alpha values ranging from 47 to 142, errors of 117% to 554% against the target of 21.746. Random matrices produce errors of ~201%. The text models are actually *further* from 8e than random matrices in two of three cases.

**Important note on naming inconsistency:** The v1 tests that originally reported 8e conservation used `Df = PR` (raw participation ratio). The v2 shared formula uses `Df = 2/alpha` and `sigma = PR/d`. These are completely different quantities sharing the same symbol. The 8e claim uses the v1 definition.

**Why v1 reported different values:** The v1 tests used ~100 short code snippets, producing much smaller embedding matrices. With 2000 real text sentences (in-distribution data), the eigenvalue spectrum and participation ratio change dramatically. The 8e "conservation" appears to be an artifact of small sample sizes and specific data selection, not a genuine invariant.

## Ablation Study

| Form | Formula | |rho| | vs R_full |
|---|---|---|---|
| R_full | (E/grad_S) * sigma^Df | 0.378 | baseline |
| **R_sub** | **(E - grad_S) * sigma^Df** | **0.394** | **-0.016 (BEATS R_full)** |
| R_exp | E * exp(-grad_S) * sigma^Df | 0.208 | +0.170 (R_full wins) |
| R_log | log(max(E, 0.01)) / (grad_S + 1) | 0.013 | +0.365 (R_full wins) |
| R_add | E / (grad_S + sigma^Df) | 0.063 | +0.315 (R_full wins) |

R_full beats 3/4 alternatives. **This criterion PASSES.**

However, R_sub = (E - grad_S) * sigma^Df (subtraction instead of division) slightly outperforms R_full (|rho| = 0.394 vs 0.378). The specific division form E/grad_S is not uniquely privileged -- a simpler subtraction form works equally well or better.

## Novel Predictions

**Prediction 1:** "R_full will correlate with human similarity scores at |rho| > 0.4"
- Observed: |rho| = 0.378 (p = 0.025)
- **FAIL** (below 0.4 threshold, though statistically significant)

**Prediction 2:** "Clusters with R_full in the top quartile will have mean human score > 3.5"
- Top quartile mean human score: **1.96** (threshold was > 3.5)
- Bottom quartile mean human score: **3.24**
- **FAIL** -- R_full's top quartile has the *lowest* human scores, not the highest. The correlation is inverted.

## Key Finding

**R = (E / grad_S) * sigma^Df is not entirely tautological -- it captures structure beyond its individual components.** R_full outperforms E alone, 1/grad_S alone, sigma^Df alone, and pairwise combinations (E/grad_S, E*sigma^Df, SNR) by significant margins (0.12 to 0.35 in |rho|). The whole is genuinely different from its parts.

**However, the formula has serious problems:**

1. **Inverted direction.** R correlates *negatively* with human semantic similarity judgments. Higher R means lower quality by human standards. This undermines any claim that R measures "truth" or "meaning quality."

2. **8e conservation law fails.** On in-distribution text data with adequate sample size (n=2000), PR*alpha ranges from 47 to 142 across three models. The 8e = 21.746 "constant" is not conserved. The previous agreement appears to be an artifact of small, curated datasets.

3. **Functional form is not privileged.** R_sub = (E - grad_S) * sigma^Df slightly outperforms R_full. The division form is not uniquely correct.

4. **Novel predictions fail.** Both pre-registered predictions fail, one catastrophically (top-quartile R clusters have the *lowest* human scores, not the highest).

**One-sentence summary:** R is not a pure tautology -- it combines its components in a non-trivial way that captures genuine structure -- but it does not predict human quality judgments in the expected direction, the 8e conservation law does not hold on adequately-sampled data, and an alternative formula (subtraction) slightly outperforms the specific division form.

## Data

- **Dataset:** STS-B test split (mteb/stsbenchmark-sts), 1,379 sentence pairs, human scores 0.0-5.0
- **Primary model:** all-MiniLM-L6-v2 (384-dim embeddings)
- **8e models tested:** all-MiniLM-L6-v2, all-mpnet-base-v2, multi-qa-MiniLM-L6-cos-v1
- **Clustering:** 35 valid quantile-based clusters from 39 bins
- **Unique sentences for 8e test:** 2,000
- **Random seed:** 42

## Limitations

1. **Single benchmark.** Only STS-B was tested. Different benchmarks (MTEB clustering, NLI) might show different patterns. However, STS-B is among the most standard text similarity benchmarks.

2. **Single embedding model for Test 1.** Only all-MiniLM-L6-v2 was used for the component comparison. Results might differ with other models.

3. **Cluster size.** Only 35 valid clusters (n=35 for Spearman) limits statistical power. The p=0.025 for R_full is significant but not highly so.

4. **Negative correlation interpretation.** The negative correlation might reflect a genuine (if inverted) relationship -- R might be measuring something real, just not "quality" as assumed. This warrants further investigation.

5. **Naming inconsistency.** The v1 "Df" (raw participation ratio) and v2 "Df" (2/alpha) are different quantities. This inconsistency within the project makes cross-version comparisons unreliable and risks confusion.

6. **8e test uses different Df definition.** The 8e test necessarily used v1 definitions to test the v1 claim. But this means the 8e claim and the R formula use different variable definitions, raising the question of whether they are even part of the same coherent theory.
