# Q22: Universal R Threshold Exists

## Hypothesis

A universal R threshold exists that can serve as a decision boundary across application domains. Specifically: median(R) of the combined R distribution in any domain approximates the optimal classification threshold within 10% deviation, making it possible to deploy R-gating without domain-specific calibration.

## v1 Evidence Summary

The hypothesis was pre-registered and tested on 7 real-world domains using Youden's J statistic for optimal threshold selection. Embedding model: all-MiniLM-L6-v2.

| Domain | Measured Median(R) | Optimal Threshold | Deviation | Pass (< 10%)? |
|--------|-------------------|-------------------|-----------|---------------|
| STS-B | 2.16 | 2.49 | 12.95% | FAIL |
| SST-2 | 2.04 | 1.84 | 11.11% | FAIL |
| SNLI | 2.13 | 2.02 | 5.22% | PASS |
| Market Regimes | 0.20 | 0.35 | 43.14% | FAIL |
| AG-News | -- | -- | -- | PASS |
| Emotion | -- | -- | -- | FAIL |
| MNLI | 3.46 | 3.48 | 0.59% | PASS |

Result: 3 of 7 domains pass the 10% criterion. Even excluding the Market outlier, only 3 of 6 NLP domains pass. Even at 15% tolerance, 4 of 7 fail.

**The hypothesis was FALSIFIED.** R value ranges differ by 17x across domains (Market: 0.20 vs MNLI: 3.46). No single threshold works universally.

Three independent audits (DEEP_AUDIT, OPUS_AUDIT, VERIFY) all reproduced identical numerical results.

## v1 Methodology Problems

The Phase 6C verification found this to be one of the cleanest investigations in the framework, with only minor issues:

1. **Market domain is questionable.** Bull/bear regimes defined by date ranges, not ground-truth market states. Youden's J = 0.17 suggests classes may be nearly inseparable. However, excluding Market does not change the verdict.

2. **Single embedding model.** All NLP domains tested with all-MiniLM-L6-v2 only. Cross-model variation in R distributions (demonstrated in Q23) means these 7 domains are not fully independent. The universality failure could be worse with multiple models.

3. **No sample size guidance.** The falsification resolves the negative question but does not answer the constructive follow-up: how much calibration data is needed per domain to find a good threshold?

Verdict: FALSIFIED (CONFIRMED). R = 1320 unchanged. This is a well-executed negative result with sound methodology.

## v2 Test Plan

### Phase 1: Replicate the Falsification

1. Re-run the original 7-domain test with the GLOSSARY-defined R
2. Confirm the deviation pattern replicates
3. This establishes the baseline for all further analysis

### Phase 2: Multi-Model Universality Test

1. Repeat the 7-domain test across at least 3 embedding models:
   - all-MiniLM-L6-v2 (original)
   - all-mpnet-base-v2
   - e5-base-v2
2. For each model, compute median(R) and optimal threshold per domain
3. Report whether the universality failure is model-dependent or model-invariant
4. If any model shows universality (< 10% deviation on 5+ domains), that would challenge the falsification

### Phase 3: Calibration Efficiency Study

Since universal thresholds are falsified, the constructive question is: how efficiently can domain-specific thresholds be calibrated?

1. For each domain, vary the calibration sample size (n = 10, 25, 50, 100, 250, 500)
2. At each n, compute the calibrated threshold using Youden's J on the calibration set
3. Apply the calibrated threshold to a held-out test set
4. Measure: (a) deviation from full-data optimal, (b) F1 on held-out test set
5. Determine the minimum n required for < 5% deviation from optimal

### Phase 4: Relative Threshold Test

Test an alternative hypothesis: while absolute thresholds do not transfer, perhaps RELATIVE thresholds (percentile-based within each domain's R distribution) are more universal.

1. For each domain, compute the percentile of the optimal threshold within that domain's R distribution
2. Test whether a fixed percentile (e.g., p50, p60) consistently approximates optimal across domains
3. If a fixed percentile works (within 10% of optimal F1 across 5+ domains), that would be a useful practical result

## Required Data

- **STS-B** (Semantic Textual Similarity Benchmark)
- **SST-2** (Stanford Sentiment Treebank)
- **SNLI** (Stanford NLI)
- **MNLI** (Multi-Genre NLI)
- **AG-News** (news topic classification)
- **Emotion** (tweet emotion classification)
- **Market Regimes** (SPY via yfinance, 3+ years)
- Additional domains for replication: **QQP**, **MRPC**, **IMDB sentiment**

## Pre-Registered Criteria

For replication of falsification (Phase 1-2):
- **Success (confirm falsification):** Median(R) deviates > 10% from optimal on 4+ of 7 domains, replicated across 2+ models
- **Failure (overturn falsification):** Median(R) deviates < 10% from optimal on 5+ of 7 domains on any model tested
- **Inconclusive:** Results are model-dependent (some models show universality, others do not)

For calibration efficiency (Phase 3):
- **Success:** < 50 labeled examples per domain suffice for < 5% deviation from optimal threshold
- **Failure:** > 250 labeled examples required for < 5% deviation
- **Inconclusive:** 50-250 examples required (domain-dependent)

For relative threshold (Phase 4):
- **Success:** A fixed percentile achieves within 10% of optimal F1 on 5+ of 7 domains
- **Failure:** No fixed percentile achieves within 10% on 4+ domains
- **Inconclusive:** One percentile works for NLP domains but fails for non-NLP domains

## Baseline Comparisons

1. **Median(R)** as universal threshold (the falsified hypothesis, as control)
2. **Fixed R = sqrt(3)** (theoretical constant from Q23)
3. **Youden's J optimal on full data** (upper bound, not achievable in practice)
4. **Cross-validation optimal** (practical calibration baseline)
5. **Bare E with Youden's J** (does R-based thresholding outperform E-based thresholding?)

## Salvageable from v1

- **The entire Q22 investigation is salvageable.** The methodology is sound, the results are reproducible, and the falsification is genuine. This is one of the best experiments in the v1 project.
- **Test code and results** at `v1/questions/lower_q22_1320/` -- directly reusable for replication
- **The 7-domain result table** provides the baseline against which all v2 work compares
- **The Youden's J methodology** for optimal threshold selection is the right statistical approach
- **The finding that R ranges differ 17x across domains** is an important constraint on any R-based system
- **Audit reports** (DEEP_AUDIT_Q22, OPUS_AUDIT_Q22, VERIFY_Q22) at `v1/questions/lower_q22_1320/reports/` provide independent verification
