# Deep Audit: Q19 Value Learning

**Auditor:** Claude Opus 4.5
**Date:** 2026-01-27
**Status:** CONDITIONALLY CONFIRMED - WITH SIGNIFICANT CONCERNS

---

## Summary

Q19 asks: "Can R guide which human feedback to trust?" The hypothesis is that high R correlates with high inter-annotator agreement (r > 0.5).

**Verdict: PASS WITH CAVEATS** - The overall correlation meets threshold (r=0.52) but contains methodological issues that inflate the result.

---

## Test Verification

### Code Review

**Test File:** `experiments/open_questions/q19/test_q19_value_learning.py`
**Result File:** `experiments/open_questions/q19/q19_results.json`

| Check | Status |
|-------|--------|
| Test file exists | YES |
| Results file exists | YES |
| Uses real external data | YES |
| Pre-registration documented | YES |
| Statistical methods appropriate | PARTIAL |

### Data Sources

The test correctly uses REAL human preference datasets:
1. **Stanford SHP** (stanfordnlp/SHP) - Reddit upvote distributions - 300 examples
2. **OpenAssistant OASST1** - Multi-annotator quality ratings - 300 examples
3. **Anthropic HH-RLHF** - Binary preference pairs - 300 examples

**Total: 900 examples** from real HuggingFace datasets - NOT synthetic.

---

## Critical Findings

### Issue 1: SIMPSON'S PARADOX RISK - HIGH SEVERITY

**Problem:** The overall correlation is inflated by cross-dataset confounding.

| Dataset | N | Pearson r | Log R Mean | Agreement Mean |
|---------|---|-----------|------------|----------------|
| OASST | 300 | **+0.60** | 7.92 | 0.608 |
| SHP | 300 | **-0.14** | 5.59 | 0.096 |
| HH-RLHF | 300 | **-0.31** | 17.03 | 0.758 |

**Analysis:**
- Average within-source correlation: **0.051** (near zero!)
- OASST shows strong positive correlation
- SHP and HH-RLHF show **NEGATIVE** correlations
- The overall r=0.52 is driven by:
  - HH-RLHF having both high log R and high agreement
  - SHP having both low log R and low agreement
  - This creates spurious cross-dataset correlation

**Impact:** The claim "R predicts agreement" is **confounded** with "different datasets have different characteristics."

### Issue 2: AGREEMENT PROXY QUALITY - MEDIUM SEVERITY

The agreement metrics are computed differently per dataset:
- **OASST:** Uses actual multi-annotator quality labels - BEST
- **SHP:** Uses vote-based agreement (Reddit upvotes) - NOISY proxy
- **HH-RLHF:** Uses **response length ratio** as agreement proxy - CRUDE, questionable validity

From code:
```python
# HH-RLHF agreement calculation
len_ratio = min(len(chosen_resp), len(rejected_resp)) / (max(...) + 1)
agreement = 0.5 + 0.5 * (1 - len_ratio)
```

This assumes "similar length = ambiguous" which is a weak heuristic with no validation.

### Issue 3: LOG TRANSFORM IMPROVES RESULTS - SUSPICIOUS

| Metric | r value |
|--------|---------|
| Pearson (log R) | 0.5221 (PASS) |
| Pearson (raw R) | 0.3346 (FAIL) |
| Spearman (rank) | 0.4827 (borderline) |

The hypothesis passes ONLY with log-transformed R. This could indicate:
- Non-linear relationship (legitimate)
- Or p-hacking via transformation choice

### Issue 4: PRACTICAL VALIDITY

**What IS actually confirmed:**
- High R vs Low R split shows meaningful difference: 0.621 vs 0.354 agreement (0.267 difference)
- OASST specifically shows strong within-dataset correlation (r=0.60)

**What is NOT confirmed:**
- Universal relationship across datasets
- Causal mechanism (R does not "guide" trust, it correlates with it)

---

## Data Integrity Checks

| Check | Result |
|-------|--------|
| Data loaded from real HuggingFace | YES |
| Sample sizes reported correctly | YES |
| N=900 matches claimed | YES |
| R values reasonable range | R_mean=22M (extreme due to low sigma!) |
| No obvious data fabrication | PASS |

**NOTE:** R values are extreme (mean 22 million) because R = E/sigma and sigma can be very small. This makes the log transform essential, not optional.

---

## Negative Controls

The test does NOT include negative controls:
- No shuffled data baseline
- No random R baseline
- No within-dataset-only analysis as primary metric

---

## Verdict

**STATUS: CONDITIONALLY CONFIRMED**

The test PASSES by pre-registered threshold (r=0.5221 > 0.5) but with critical caveats:

### What CAN be claimed:
1. R correlates with agreement at the **dataset level** (different datasets have different R and agreement characteristics)
2. Within OASST specifically, R predicts agreement well
3. Splitting by R median shows meaningful agreement difference (0.27)

### What CANNOT be claimed:
1. R reliably predicts agreement **within** arbitrary datasets
2. The relationship is causal or actionable
3. HH-RLHF and SHP results support the hypothesis (they show negative correlations)

### Recommendation:
- Re-run with **within-dataset correlation only** as primary metric
- Replace HH-RLHF agreement proxy with actual annotator disagreement (if available)
- Add negative controls (shuffled data, random R)

---

## Bullshit Check

| Red Flag | Found? |
|----------|--------|
| Synthetic data passed as real | NO |
| Cherry-picked threshold | NO (pre-registered) |
| P-hacking via transforms | POSSIBLE (log vs raw) |
| Simpson's paradox | YES (cross-dataset confounding) |
| Circular logic | NO |
| Missing negative controls | YES |

**Overall:** The test is methodologically questionable due to Simpson's paradox, but uses real data and honest reporting of within-source correlations.

---

## Files Examined

- `experiments/open_questions/q19/test_q19_value_learning.py` (577 lines)
- `experiments/open_questions/q19/q19_results.json` (51 lines)
- `research/questions/medium_priority/q19_value_learning.md` (83 lines)
