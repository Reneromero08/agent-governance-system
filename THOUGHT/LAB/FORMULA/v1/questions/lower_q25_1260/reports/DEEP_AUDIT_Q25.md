# Deep Audit: Q25 What Determines Sigma

**Auditor:** Claude Opus 4.5
**Date:** 2026-01-27
**Status:** UNRESOLVED - CONFLICTING EVIDENCE

---

## Summary

Q25 asks: "Is there a principled way to derive sigma, or is it always empirical?"

**Verdict: STATUS INCORRECT** - The question is marked "RESOLVED" but evidence is conflicting between synthetic and real data tests.

---

## Test Verification

### Code Review

**Test Files:**
- `experiments/open_questions/q25/test_q25_sigma.py` (1148 lines) - Synthetic data test
- `experiments/open_questions/q25/test_q25_real_data.py` - Real data validation

**Result Files:**
- `experiments/open_questions/q25/q25_results.json` (543 lines) - Synthetic results
- `experiments/open_questions/q25/q25_real_data_results.json` (280 lines) - Real data results

| Check | Status |
|-------|--------|
| Test files exist | YES |
| Results files exist | YES |
| Uses real external data | PARTIAL (real test exists but contradicts) |
| Pre-registration documented | YES |
| Multiple validation tests | YES |

---

## Critical Findings

### Finding 1: CONFLICTING RESULTS - SEVERE

**Synthetic Data Test:**
- R^2_cv = **0.8617** (PASS > 0.7 threshold)
- Verdict: "SIGMA_PREDICTABLE"
- 22 synthetic datasets across 7 domains

**Real Data Test:**
- R^2_cv = **0.0000** (FAIL < 0.5 threshold)
- Verdict: "SIGMA_IRREDUCIBLY_EMPIRICAL"
- 9 real datasets from HuggingFace + NCBI GEO

This is a MAJOR inconsistency. The markdown claims Q25 is "RESOLVED - SIGMA IS PREDICTABLE" but the real data test FALSIFIES this claim.

### Finding 2: SYNTHETIC DATA PROBLEM

The synthetic test uses **generated embeddings** with known properties:
- `generate_nlp_like()` - Creates artificial NLP-like embeddings
- `generate_market_like()` - Creates artificial market embeddings
- `generate_image_like()` - Creates artificial image embeddings
- etc.

These are NOT real NLP, market, or image embeddings - they are synthetic approximations designed to have certain statistical properties. The high R^2 on synthetic data may reflect the model learning the patterns of the synthetic generators, not real data structure.

### Finding 3: REAL DATA SHOWS NO PREDICTABILITY

From `q25_real_data_results.json`:

| Dataset | Domain | Optimal Sigma |
|---------|--------|---------------|
| stsb | nlp_similarity | 2.72 |
| sst2 | nlp_sentiment | 2.72 |
| ag_news | nlp_news | 2.42 |
| imdb | nlp_reviews | 2.42 |
| snli | nlp_inference | 2.42 |
| mnli | nlp_multi_genre | 1.92 |
| emotion | nlp_emotion | 2.42 |
| squad | qa | 2.72 |
| geo_expression | gene_expression | **39.44** |

**Key observations:**
- NLP datasets all cluster in 1.9-2.7 range (narrow variance)
- Gene expression is a massive outlier (39.44)
- Cross-validation R^2 = 0.0 because the GEO outlier is unpredictable

The formula found works only because GEO is so different - it's essentially detecting "is this gene expression data or NLP?"

### Finding 4: FORMULA IS NOT GENERAL

The claimed formula from synthetic test:
```
log(sigma) = 3.4560 + 0.9396 * log(mean_dist) - 0.0872 * log(effective_dim) - 0.0212 * eigenvalue_ratio
```

Applied to real data:
- Training R^2 = 0.989 (very high - overfitting)
- CV R^2 = 0.0 (complete failure to generalize)

This is classic overfitting: the formula memorizes the training data (mostly dominated by the GEO outlier) but cannot predict held-out data.

---

## Data Sources Verification

### Synthetic Test (q25_results.json):
- **All data is synthetically generated** via numpy random
- NOT from external sources
- Designed to have diverse properties

### Real Data Test (q25_real_data_results.json):
| Dataset | Source | Verified Real? |
|---------|--------|----------------|
| stsb | HuggingFace/mteb/stsbenchmark-sts | YES |
| sst2 | HuggingFace/stanfordnlp/sst2 | YES |
| ag_news | HuggingFace/fancyzhx/ag_news | YES |
| imdb | HuggingFace/stanfordnlp/imdb | YES |
| snli | HuggingFace/stanfordnlp/snli | YES |
| mnli | HuggingFace/nyu-mll/glue/mnli | YES |
| emotion | HuggingFace/dair-ai/emotion | YES |
| squad | HuggingFace/squad | YES |
| geo_expression | NCBI_GEO/GSE45267 | YES |

Real data sources are legitimate.

---

## Verdict

**STATUS: UNRESOLVED - CONTRADICTORY EVIDENCE**

The current status ("RESOLVED - SIGMA IS PREDICTABLE") is **incorrect**.

### What the evidence actually shows:

1. **Synthetic data:** Sigma CAN be predicted (R^2=0.86)
   - But this may be artifact of synthetic data design

2. **Real data:** Sigma CANNOT be predicted (R^2=0.00)
   - Formula fails completely on cross-validation
   - Only works because it detects the GEO outlier

### Correct Conclusion:

Sigma is **partially predictable within domain** (NLP datasets cluster tightly) but **not predictable across domains** (GEO is unpredictable from NLP properties).

The question should be marked:
- **STATUS: PARTIALLY CONFIRMED** with caveats
- OR **STATUS: DOMAIN-DEPENDENT**

### Recommendations:

1. Update question status to reflect conflicting evidence
2. Run more real data tests (more domain diversity)
3. Test within-domain prediction specifically
4. Remove or downweight synthetic data results

---

## Bullshit Check

| Red Flag | Found? |
|----------|--------|
| Synthetic data passed as validation | YES (main result is from synthetic) |
| Real data contradicts but ignored | YES (status not updated) |
| Overfitting (high train, zero CV) | YES |
| Cherry-picked metrics | POSSIBLE (synthetic over real) |
| Claim not supported by evidence | YES (status says RESOLVED but isn't) |

**Overall:** This question needs status revision. The synthetic-data result should not be trusted over the real-data result.

---

## Files Examined

- `experiments/open_questions/q25/test_q25_sigma.py` (1148 lines)
- `experiments/open_questions/q25/q25_results.json` (543 lines)
- `experiments/open_questions/q25/test_q25_real_data.py` (exists)
- `experiments/open_questions/q25/q25_real_data_results.json` (280 lines)
- `research/questions/lower_priority/q25_what_determines_sigma.md` (115 lines)
