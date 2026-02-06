# Q03 Verdict

## Result: INCONCLUSIVE

R = E/grad_S does not clearly generalize across fundamentally different domains.
It shows a weak-to-moderate signal in one domain (tabular), zero signal in the
primary domain (text semantics), and no meaningful signal in financial time series.
Cross-domain threshold transfer "passes" on a technicality (low baseline accuracy),
but this is not compelling evidence of universality.

## Per-Domain Results

### Domain 1: Text Semantics (STS-B)
- **Dataset:** STS-B test set, 1,379 sentence pairs, encoded with all-MiniLM-L6-v2 (384-d)
- **Method:** Grouped pairs into 10 bins by human similarity score (0-5), computed R on pooled embeddings per bin
- **R_simple vs human score:** rho = 0.0303, p = 0.934 -- **FAIL (effectively zero correlation)**
- **R_full vs human score:** rho = -0.0303, p = 0.934 -- **FAIL**
- **Pair cosine vs human score:** rho = 1.000, p < 1e-63 -- **perfect correlation**
- **SNR vs human score:** rho = 0.030, p = 0.934 -- also fails
- **E (raw) vs human score:** rho = -0.042, p = 0.907 -- also fails

**Interpretation:** R has literally no correlation with human-judged text similarity.
The simple mean pairwise cosine similarity of paired sentences correlates perfectly
(rho = 1.0). R adds nothing; it is worse than every alternative tested. The division
by grad_S (std of pairwise similarities within the pooled cluster) destroys the
signal that E alone carries weakly. This is because pooling sentence1 and sentence2
embeddings into one cluster makes E measure intra-cluster cohesion (dominated by
embedding geometry), not the paired semantic similarity that humans judge.

Note: R_simple and SNR are numerically identical here (both equal E/grad_S computed
on the same pooled cluster), confirming that R_simple IS just a signal-to-noise ratio
of cosine similarities -- not a distinct metric.

### Domain 2: Numerical/Tabular (California Housing)
- **Dataset:** California Housing, 20,640 samples, 8 standardized features
- **Method:** Grouped by target value (median house price) into 15 bins, computed R on feature vectors per bin
- **R_simple vs target homogeneity:** rho = 0.521, p = 0.046 -- **MARGINAL PASS (barely above 0.5)**
- **R_full vs target homogeneity:** rho = 0.507, p = 0.054 -- borderline (not significant at 0.05)
- **E (raw) vs homogeneity:** rho = 0.525, p = 0.044 -- slightly better than R
- **SNR vs homogeneity:** rho = 0.521, p = 0.046 -- identical to R_simple
- **R_simple vs R-squared:** rho = -0.082, p = 0.771 -- no correlation with prediction quality

**Interpretation:** R_simple marginally correlates with target homogeneity (how
uniform house prices are within each bin). However, raw E correlates slightly
better (rho = 0.525 vs 0.521), meaning the division by grad_S does not improve
the signal. R_simple and SNR are again numerically identical. R tells us nothing
about prediction quality (R-squared) within each bin. The "pass" is marginal and
the signal comes entirely from E, not from the ratio.

### Domain 3: Financial Time Series (S&P 500 / SPY)
- **Dataset:** SPY, 753 trading days (2023-02-06 to 2026-02-05), 733 sliding 20-day windows
- **Method:** Grouped windows into 5 volatility regimes (quintiles), computed R on return-sequence vectors per regime
- **R_simple vs regime stability:** rho = -0.205, p = 0.741 -- **FAIL**
- **R_full vs regime stability:** rho = -0.205, p = 0.741 -- **FAIL**
- **R vs |autocorrelation|:** rho = -0.800, p = 0.104 -- suggestive but n=5, not significant

**Interpretation:** R has no meaningful correlation with regime stability or
predictability. With only 5 data points (volatility bins), statistical power
is extremely low, but even the direction is wrong: R is negative for high-volatility
regimes (which are actually the most stable, since volatility clusters in time).
This suggests R's sign structure (high E + low grad_S = high R) does not map to
financial regime quality in any useful way.

## Cross-Domain Transfer

- **Threshold calibrated on Domain 1 (Text):** R_simple >= 0.311 predicts "high quality"
- **Domain 1 accuracy:** 60% (only 10 bins, essentially coin-flip baseline)
- **Domain 2 accuracy with D1 threshold:** 66.7% (delta = 6.7%, within 20%) -- PASS
- **Domain 3 accuracy with D1 threshold:** 40% (delta = 20.0%, borderline) -- PASS (barely)
- **Domain 2 -> Domain 3 transfer:** D2 cal accuracy 73.3%, D3 accuracy 40% (delta = 33.3%) -- FAIL

**Honest assessment:** The D1-to-D2 and D1-to-D3 transfers "pass" the 20% criterion,
but this is misleading. The D1 calibration accuracy is only 60% -- barely above
chance for a binary classification. A threshold that achieves 60% on one domain
and 40-67% on others is not evidence of meaningful transfer; it is evidence that
the threshold has no discriminative power anywhere. A random threshold would
produce similar results. The D2-to-D3 transfer, with a stronger calibration
accuracy (73.3%), fails badly (40% on D3), which is more informative.

## Comparison vs Domain-Specific Metrics

### Domain 1 (Text)
| Metric | |rho| with human score | R beats? |
|--------|----------------------|----------|
| R_simple | 0.030 | -- |
| Pair cosine | 1.000 | NO |
| SNR | 0.030 | NO (identical) |
| E (raw) | 0.042 | NO |

R is catastrophically outperformed by the simplest domain-specific metric (cosine
similarity of paired sentences).

### Domain 2 (Tabular)
| Metric | |rho| with homogeneity | R beats? |
|--------|----------------------|----------|
| R_simple | 0.521 | -- |
| R-squared | 0.196 | YES |
| SNR | 0.521 | NO (identical) |
| E (raw) | 0.525 | NO |

R beats R-squared (which measures a different thing -- within-cluster predictability,
not cluster tightness). R does not beat E or SNR, and in fact R_simple IS SNR
for this computation.

### Domain 3 (Financial)
| Metric | |rho| with stability | R beats? |
|--------|---------------------|----------|
| R_simple | 0.205 | -- |
| |Sharpe| | 0.051 | YES |
| Volatility | 0.205 | NO (identical magnitude) |
| SNR | 0.205 | NO (identical) |
| E (raw) | 0.205 | NO (identical) |

All metrics are equally bad (none significantly correlate with stability).
R "beats" Sharpe ratio only because both are uninformative.

## Critical Finding: R_simple = SNR

A key discovery in this test: **R_simple is numerically identical to the simple
signal-to-noise ratio (mean/std) of pairwise cosine similarities.** This is not
a coincidence -- it is the definition:

    R_simple = E / grad_S = mean(cosines) / std(cosines) = SNR

This means R_simple is not a novel metric. It is the classical signal-to-noise
ratio applied to cosine similarity distributions. The "generalization" question
reduces to: does the SNR of cosine similarities predict quality across domains?
The answer is: weakly in one domain, not at all in the other two.

R_full adds sigma^Df scaling, but this consistently reduces correlation strength
compared to R_simple (Domain 2: 0.507 vs 0.521; Domain 1: -0.030 vs 0.030).
The fractal scaling term hurts rather than helps.

## Data

| Domain | Dataset | N samples | N groups | Source |
|--------|---------|-----------|----------|--------|
| Text | STS-B test | 1,379 pairs | 10 bins | HuggingFace mteb/stsbenchmark-sts |
| Tabular | California Housing | 20,640 | 15 bins | sklearn |
| Financial | SPY daily | 753 days / 733 windows | 5 regimes | yfinance |

Seed: 42. All code deterministic.

## Limitations

1. **Text domain: pooling destroys signal.** Pooling sentence1+sentence2 embeddings
   into one cluster and computing intra-cluster E is not the right way to measure
   pairwise similarity quality. The formula's E definition (mean pairwise cosine of
   ALL points in a set) does not align with the task of measuring how similar TWO
   specific sentences are. This is a structural mismatch between R's definition and
   the text similarity task.

2. **Financial domain: only 5 data points.** Five volatility bins give very low
   statistical power. However, even the direction of correlation is wrong, which
   is informative.

3. **Tabular domain: the signal comes from E, not R.** The division by grad_S
   does not add value over raw E. The "pass" is really an E pass, not an R pass.

4. **Cross-domain transfer baseline is too weak.** With D1 calibration at 60%,
   transfer "success" is meaningless. A proper transfer test requires a calibration
   accuracy of at least 70-80% to have discriminative power.

5. **R_simple = SNR means the novelty claim is undermined.** If R reduces to a
   classical metric (mean/std of cosine similarities), the universality claim
   must demonstrate that this particular SNR formulation has cross-domain validity
   beyond what is already known about SNR in signal processing.

6. **No protein structure, climate, or medical domains.** The v2 README called
   for 5 domains. Only 3 were tested here per the task specification.

## Verdict Rationale

Per pre-registered criteria:
- **Domain correlation (rho > 0.5):** 1/3 pass (need >=2 for confirm, <1 for falsify)
- **Cross-domain transfer:** 2/3 pass (but on a technicality; baseline too weak)
- **R beats alternatives:** 2/3 domains (but only against weak/irrelevant alternatives)

This falls in the INCONCLUSIVE range, but leans toward FALSIFIED. The one domain
where R "works" (tabular, rho = 0.521) shows that raw E works equally well
(rho = 0.525), meaning R's specific innovation (dividing by grad_S) adds no value.
The universality claim -- that R captures deep structure across fundamentally
different domains -- is not supported by this evidence.

**Bottom line:** R = E/grad_S is a repackaging of the signal-to-noise ratio of
cosine similarities. It works weakly where cosine similarity itself works
(feature-space clustering of similar targets), fails completely where the task
requires something different (text pair similarity, financial regime prediction),
and does not outperform simpler alternatives in any domain tested.
