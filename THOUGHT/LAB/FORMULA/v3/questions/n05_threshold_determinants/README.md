# N5: What Determines Domain-Specific R Thresholds?

## Why This Question Matters

Q22 properly falsified the universal threshold hypothesis: only 3/7 real domains pass a single threshold. But domain-specific thresholds DO exist. What determines them? If we can predict the threshold from domain properties, Q22's falsification becomes the starting point of a research program rather than just a negative result.

## Hypothesis

**H0:** Optimal R thresholds per domain are predictable from measurable domain properties.

**Specific sub-hypotheses:**
- H0a: Threshold correlates with domain vocabulary entropy
- H0b: Threshold correlates with mean E (average cosine similarity in the domain)
- H0c: Threshold correlates with sigma (if N3 also finds sigma is predictable, this creates a chain)
- H0d: Threshold is a simple function of E distribution moments (mean, std, skew)

**H1:** Optimal thresholds are domain-specific but unpredictable -- each domain requires empirical calibration.

## Pre-Registered Test Design

### Datasets (minimum 15 domains, extending Q22's 7)

Q22's 7: STS-B, SST-2, SNLI, Market, AG-News, Emotion, MNLI

Add 8+: FEVER, LIAR, iSarcasm, MultiRC, BoolQ, COPA, RTE, WiC

### Procedure

1. For each domain:
   - Compute R (using v2 GLOSSARY E definition) for all items
   - Determine optimal threshold via ROC analysis (maximize Youden's J)
   - Compute domain properties: vocabulary entropy, mean E, std E, skewness of E distribution, sigma, intrinsic dimensionality
2. Build regression model: optimal_threshold ~ domain_properties
3. Cross-validate (leave-one-domain-out)
4. Test predictive accuracy: can the model predict the threshold for a held-out domain?

### Success Criteria

- **Predictable:** Leave-one-out prediction error < 15% on average
- **Partially predictable:** Error 15-30%, some structure but noisy
- **Unpredictable:** Error > 30%, empirical calibration required per domain

### Implications

- If predictable: R can be deployed on new domains without per-domain calibration
- If unpredictable: R requires calibration data for each new domain, limiting practical utility

## Dependencies

- Builds directly on Q22's falsification (v2/Q22)
- N3 results (sigma determinants) may provide features for the regression

## Related

- v2/Q22 (Threshold calibration -- the falsification this extends)
- N3 (Sigma determinants -- parallel question, shared methodology)
- v2/Q16 (Domain boundaries -- the confirmed Q that proved domain discrimination works)
