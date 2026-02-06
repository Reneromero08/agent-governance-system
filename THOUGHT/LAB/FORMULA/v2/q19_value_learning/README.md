# Q19: R Correlates with Human Value Agreement

## Hypothesis

R computed over human feedback data correlates with inter-annotator agreement (IAA). Specifically, high R indicates reliable, high-agreement feedback signal, while low R indicates ambiguous or contested examples. This means R can guide which human feedback to trust during value learning: high-R examples provide cleaner training signal, and low-R examples flag disputed or ambiguous cases that need additional annotation or should be downweighted.

## v1 Evidence Summary

Tested on 3 real human preference datasets from HuggingFace (N=900 total):

| Dataset | N | Pearson r (log R) | P-value | Agreement Proxy |
|---------|---|-------------------|---------|-----------------|
| OASST | 300 | 0.6018 | 6.01e-31 | Multi-annotator quality ratings |
| SHP | 300 | -0.1430 | 0.013 | Reddit upvote distributions |
| HH-RLHF | 300 | -0.3056 | 6.60e-08 | Response length ratio (invalid) |

Overall pooled: Pearson r(log R) = 0.5221 (PASS by pre-registered threshold of r > 0.5).
Average within-source correlation: 0.051 (near zero).
High R vs Low R agreement difference: 0.267.

A resolved re-test correctly identified Simpson's Paradox and changed verdict to INCONCLUSIVE:
- Within-dataset average r = 0.168
- Resolved confirmation threshold lowered to r > 0.3
- HH-RLHF excluded due to invalid agreement proxy

## v1 Methodology Problems

The Phase 6B verification found critical issues:

1. **Simpson's Paradox dominates the result.** The overall r=0.52 is an ecological fallacy. HH-RLHF has both high R and high agreement; SHP has both low R and low agreement. This creates a spurious positive correlation across datasets. Within each dataset, R and agreement are uncorrelated or negatively correlated (2 of 3 datasets show NEGATIVE within-dataset correlation).

2. **PASS only with log transform.** Pre-registration says "Pearson r > 0.5" but does not specify log transform. Raw R gives r=0.3346 (FAIL). Using log(R) is a degree of freedom. R values span from near-zero to millions (mean 22 million), so log transform is arguably necessary, but should have been pre-registered.

3. **HH-RLHF agreement proxy is invalid.** Using response length ratio as a proxy for annotator agreement has no validation. Response length is not a measure of human disagreement. The resolved test correctly excludes this dataset.

4. **Only OASST shows genuine within-dataset signal** (r=0.505 in resolved test). OASST also has the best agreement metric (actual multi-annotator labels). But N=1 dataset is insufficient for a general claim.

5. **No precision/recall analysis** for an R-based feedback filter. Even OASST r=0.505 explains only ~25% of variance. Practical utility is undemonstrated.

6. **Resolved test lowered confirmation threshold** from r > 0.5 to r > 0.3 (a degree of freedom favoring the hypothesis).

Verdict recommended status INCONCLUSIVE, R from 1380 to 700-800.

## v2 Test Plan

### Phase 1: Pre-Register Transform and Thresholds

Before any testing:
1. Explicitly state whether log(R) or raw R will be the primary metric
2. Pre-register the confirmation threshold (we use r > 0.3 within-dataset, matching the resolved test's methodology)
3. Pre-register that the primary metric is WITHIN-dataset correlation, not pooled
4. Pre-register that agreement must be measured from actual multi-annotator labels, not proxies

### Phase 2: Multi-Annotator Dataset Test

1. Select at least 3 datasets with genuine multi-annotator labels:
   - OpenAssistant (oasst1) -- quality ratings from multiple annotators
   - WMT human evaluation -- translation quality from multiple raters
   - SemEval STS tasks -- similarity ratings from multiple annotators
2. For each dataset:
   - Embed response texts using standard embedding model
   - Compute R (GLOSSARY-defined) over response groups
   - Compute inter-annotator agreement (Krippendorff's alpha or Fleiss' kappa)
   - Compute within-dataset Pearson and Spearman correlations between log(R) and IAA
3. Compare R to bare E and random baseline

### Phase 3: Practical Utility Test

1. For datasets where R shows positive correlation with IAA:
   - Define R-based filter thresholds (high/medium/low R)
   - Compute precision/recall for identifying high-agreement examples
   - Generate ROC curves
   - Compute the actual practical improvement from using R to filter training data
2. Compare to simpler filters:
   - Length-based filtering
   - Agreement-based filtering (using a subsample of annotations)
   - Random filtering

### Phase 4: Cross-Dataset Generalization

1. Calibrate R thresholds on Dataset A
2. Apply frozen thresholds to Dataset B and C
3. Test whether the R-agreement relationship transfers without retuning
4. This is the key test for whether R has general value-learning utility

## Required Data

- **OpenAssistant (oasst1)** -- multi-annotator quality ratings (best agreement labels)
- **WMT Human Evaluation** -- translation quality with multi-rater annotations
- **SemEval STS** (2012-2017) -- semantic similarity with multi-annotator ratings
- **Chatbot Arena** (LMSYS) -- pairwise preferences with annotator metadata
- **UltraFeedback** -- multi-aspect LLM output ratings

All datasets must have genuine multi-annotator labels (not proxies like length ratios or vote counts).

## Pre-Registered Criteria

- **Success (confirm):** Within-dataset Spearman rho > 0.3 between log(R) and inter-annotator agreement on at least 2 of 3+ datasets, AND R outperforms bare E (rho difference > 0.1) on at least 1 dataset
- **Failure (falsify):** Within-dataset Spearman rho < 0.1 on all datasets, OR bare E matches or exceeds R on all datasets, OR the correlation is negative on 2+ datasets
- **Inconclusive:** Within-dataset rho is 0.1-0.3 on most datasets, or results are mixed (positive on some, negative on others, with no clear pattern)

## Baseline Comparisons

1. **Bare E** (mean pairwise cosine similarity)
2. **1/sigma** (inverse standard deviation)
3. **Random baseline** (shuffled R values)
4. **Response length** (crude but commonly used filter)
5. **Agreement subsample** (using a fraction of annotations as the filter, to benchmark against)

## Salvageable from v1

- **OASST correlation result** (within-dataset r=0.505 in resolved test): This is a genuine signal on a properly annotated dataset. The OASST pipeline can be directly reused.
- **Simpson's Paradox detection methodology**: The self-correction process is a model for v2 quality control. The resolved test's within-dataset approach should be the default.
- **The negative results** (SHP r=-0.143, HH-RLHF r=-0.306) are valuable: they demonstrate that R does NOT universally correlate with agreement, which constrains the hypothesis.
- **Test code** at `v1/questions/medium_q19_1380/tests/` for scaffolding
- **Results data** at `v1/questions/medium_q19_1380/results/q19_results.json` and `q19_resolved_results.json` for comparison
