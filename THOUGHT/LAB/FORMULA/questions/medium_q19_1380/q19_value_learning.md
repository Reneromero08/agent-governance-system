# Question 19: Value learning (R: 1380)

**STATUS: CONDITIONALLY CONFIRMED**

## Question
Can R guide which human feedback to trust? (High R = reliable signal, low R = ambiguous/contested)

## Pre-registration
- **Hypothesis**: Pearson r > 0.5 between R and inter-annotator agreement (IAA)
- **Prediction**: High R correlates with high agreement; Low R correlates with disputed examples
- **Falsification**: r < 0.3

## Test Results (2026-01-28)

### Overall Results
| Metric | Value |
|--------|-------|
| N examples | 900 |
| Pearson r (log R) | **0.5221** |
| Pearson r (raw R) | 0.3346 |
| Spearman rho | 0.4827 |
| P-value | 4.32e-64 |
| **Verdict** | **PASS** |

### By Dataset
| Dataset | N | Pearson r | P-value | Log R Mean | Agreement Mean |
|---------|---|-----------|---------|------------|----------------|
| OASST | 300 | **0.6018** | 6.01e-31 | 7.92 | 0.608 |
| SHP | 300 | -0.1430 | 0.013 | 5.59 | 0.096 |
| HH-RLHF | 300 | -0.3056 | 6.60e-08 | 17.03 | 0.758 |

### Key Findings

1. **Overall correlation meets threshold**: r = 0.52 > 0.5 (PASS)

2. **Critical caveat - Within-source divergence**:
   - Average within-source correlation: **0.051** (near zero)
   - Only OASST shows strong positive correlation (r = 0.60)
   - SHP and HH-RLHF show **negative** correlations

3. **High R vs Low R agreement**:
   - High R (above median): mean agreement = 0.621
   - Low R (below median): mean agreement = 0.354
   - Difference: 0.267 (meaningful split)

4. **Cross-source confounding detected**:
   - HH-RLHF: High log R (17.0), High agreement (0.76)
   - SHP: Low log R (5.6), Low agreement (0.10)
   - This creates spurious overall correlation

## Interpretation

The hypothesis **passes by the pre-registered threshold** but with important caveats:

### What the data shows:
- The R metric CAN distinguish between high-agreement and low-agreement feedback
- The 0.27 difference in mean agreement between high/low R groups is practically meaningful
- OASST shows genuine within-dataset correlation (r = 0.60)

### What to be cautious about:
- **Simpson's paradox risk**: The overall correlation is inflated by cross-dataset confounding
- **Dataset-specific behavior**: R works well for OASST but not for SHP or HH-RLHF
- **Agreement proxy quality**: SHP uses vote-based agreement (noisy), HH-RLHF uses length ratio (crude proxy)

### Implications for Value Learning:
1. R may be useful for **within-domain** feedback filtering (if validated per-domain)
2. Cross-domain application requires careful calibration
3. Best results expected for datasets with explicit multi-annotator labels (like OASST)

## Follow-up Questions
- Q19a: Does R correlate with annotator confidence scores (where available)?
- Q19b: Can R identify ambiguous edge cases in reward model training?
- Q19c: How does R perform on adversarial/contested value statements?

## Data Sources
- Stanford SHP (stanfordnlp/SHP) - Reddit upvote distributions
- OpenAssistant (OpenAssistant/oasst1) - Multi-annotator quality ratings
- Anthropic HH-RLHF (Anthropic/hh-rlhf) - Binary preference pairs

## Files
- Test: `questions/19/test_q19_value_learning.py`
- Results: `questions/19/q19_results.json`
