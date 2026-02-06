# Q23: sqrt(3) Has Geometric Significance

## Hypothesis

The constant sqrt(3) appearing in the formula has geometric significance -- specifically, that it derives from hexagonal symmetry in information packing (sqrt(3) = 2*sin(pi/3)) rather than being an empirically fitted parameter.

## v1 Evidence Summary

Extensive testing across 5 embedding models revealed:

- **Hexagonal packing:** Peak angle at 57.5-62.5 degrees (near expected 60 degrees for hexagonal) but peak strength 1.87 was below 2.0 significance threshold. Nearest-neighbor ratios at 1.84 (expected 1.0 for hexagonal). Labeled NOT CONFIRMED.
- **Hexagonal winding angle:** Predicted 2*pi/3 = 2.094 rad. Measured: hexagons -1.57 rad (deviation 100%). 0/3 models supported. FALSIFIED.
- **Distinguishability threshold:** sqrt(3) achieves F1 = 0.900, Cohen's d = 2.07, but alpha = 2.0 achieves F1 = 1.000, Cohen's d = 2.19. sqrt(3) is in the optimal range but NOT uniquely optimal.
- **Multi-model grid search (5 models):** Optimal alpha varied: sqrt(2), sqrt(3), 2.0, 2.5. sqrt(3) optimal for only 2/5 models (40%). Mean optimal alpha = 1.876, std = 0.363.
- **Origin:** sqrt(3) was admittedly reverse-engineered from early experiments: 1D text alpha = 0.57 ~ 1/sqrt(3), 2D Fibonacci alpha = 3.0 = sqrt(3)^2.

## v1 Methodology Problems

The Phase 6C verdict confirmed the closure but noted:

1. **Synthetic test corpus:** Hand-curated word clusters (10 related + 10 unrelated), not real classification data. Results may not generalize.
2. **Small sample size:** 20 clusters total, no confidence intervals, no bootstrap resampling, no multiple comparison correction across 8 alpha values.
3. **Unstable optima:** Two consecutive runs produced different optimal alphas for 2/5 models, indicating a flat optimization surface near sqrt(3).
4. **No source for original fit:** The claim alpha(d) = sqrt(3)^(d-2) has no source documentation for the original experiments that produced it.
5. **Negative control failed:** Control 3 (sqrt(3) should be best among nearby values) FAILED -- 1.9 beats sqrt(3). The formula still uses sqrt(3) despite this.

## v2 Test Plan

### Experiment 1: Large-Scale Alpha Optimization on Real Benchmarks

Test optimal alpha across real NLP benchmarks, not synthetic word clusters.

- **Data:** STS-B, SICK, SST-2, MNLI, and 3+ non-NLP classification tasks
- **Method:** For each dataset, compute R = E^alpha / sigma for alpha in {0.5, 0.75, 1.0, sqrt(2), 1.5, sqrt(3), 1.8, 1.9, 2.0, 2.25, 2.5, e} across 5 embedding models
- **Analysis:** Report optimal alpha per dataset per model. Compute 95% bootstrap confidence intervals. Apply Bonferroni correction for multiple comparisons.
- **Key question:** Does sqrt(3) fall within the 95% CI of the optimal alpha for a majority of dataset-model combinations?

### Experiment 2: Continuous Alpha Surface with Error Bars

Map the alpha performance surface at fine resolution with proper uncertainty quantification.

- **Method:** Sweep alpha from 0.5 to 3.0 in steps of 0.05. For each alpha, run 100 bootstrap samples. Plot F1 (or AUC) vs alpha with 95% CI bands.
- **Analysis:** Determine whether the surface is flat (many alphas perform equivalently) or peaked (a specific alpha is clearly optimal). Compute the width of the "equivalence zone" where performance is within 1% of maximum.
- **Key question:** Is there a sharp optimum, or is the performance surface flat from ~1.4 to ~2.5?

### Experiment 3: Domain-Specific Alpha Optimization

Test whether optimal alpha varies systematically by domain or task type.

- **Data:** 10+ datasets spanning text classification, similarity, NLI, sentiment, entity typing, and non-NLP domains
- **Method:** Compute optimal alpha per domain. Test for clustering of optimal alpha values.
- **Analysis:** ANOVA or Kruskal-Wallis test for whether domain type predicts optimal alpha. If it does, report domain-specific recommendations.
- **Key question:** Should the formula use a domain-specific alpha rather than a fixed sqrt(3)?

## Required Data

- STS Benchmark (STS-B) from SentEval
- SICK relatedness dataset
- SST-2 sentiment classification
- MNLI natural language inference
- AG News topic classification
- 5+ embedding models: all-MiniLM-L6-v2, all-mpnet-base-v2, paraphrase-MiniLM-L6-v2, BGE-small, GTE-small

## Pre-Registered Criteria

- **Success (confirm sqrt(3) is special):** sqrt(3) falls within the 95% CI of optimal alpha for >= 70% of dataset-model combinations AND outperforms the next-best round number (1.5 or 2.0) by > 2% F1 on average
- **Failure (falsify sqrt(3) is special):** sqrt(3) falls outside the 95% CI for >= 50% of combinations OR a simpler value (e.g., 2.0) matches or beats sqrt(3) across the board
- **Inconclusive:** Performance surface is flat enough that sqrt(3), 1.5, 2.0, and sqrt(2) are all statistically indistinguishable

## Baseline Comparisons

- **alpha = 1.0:** Simple ratio E/sigma (no exponent)
- **alpha = 2.0:** Round number that often beat sqrt(3) in v1
- **alpha = 1.5:** Lower bound of v1's "good range"
- **alpha = sqrt(2):** Another irrational constant in the range
- **alpha = e:** Upper bound comparator

## Salvageable from v1

- The finding that the optimal alpha range is approximately 1.4-2.5 is well-established
- The observation that different models prefer different optimal alphas is genuine
- The multi-model grid search framework is reusable
- The falsification of hexagonal winding angle (2*pi/3) is a valid negative result
- Test script: `test_q23_sqrt3.py`
- Results: `q23_sqrt3_final_20260127.json`
