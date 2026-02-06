# N1: Does E/grad_S Outperform Bare E?

## Why This Is Question #1

If R doesn't beat bare cosine similarity, the formula's entire theoretical apparatus is moot. Q10 found raw E gives 4.33x discrimination vs R's 1.79x for alignment detection. But that was one task with potentially wrong E. Nobody ran the direct, controlled head-to-head across multiple tasks.

## Hypothesis

**H0:** R = E/grad_S provides better discrimination than bare E on standard NLP benchmarks.

**H1 (alternative):** R performs equal to or worse than bare E, meaning the grad_S normalization adds no value.

## Pre-Registered Test Design

### Datasets (minimum 4, all public)

| Dataset | Task | Source | Metric |
|---------|------|--------|--------|
| STS-B | Semantic similarity | HuggingFace `stsb_multi_mt` | Spearman r with human scores |
| SNLI | Textual entailment | HuggingFace `snli` | AUC for entailment vs contradiction |
| SST-2 | Sentiment | HuggingFace `sst2` | Correlation with sentiment labels |
| MNLI | Multi-genre NLI | HuggingFace `multi_nli` | AUC matched/mismatched |

### Procedure

1. For each dataset, encode texts using `all-MiniLM-L6-v2` (sentence-transformers)
2. For each pair/group, compute:
   - **E** = mean pairwise cosine similarity
   - **R_simple** = E / grad_S (where grad_S = std of pairwise cosine similarities)
   - **R_full** = (E / grad_S) * sigma^Df (if computable without overflow)
3. Correlate E, R_simple, R_full with ground truth labels
4. Report: which metric best predicts the ground truth?

### Success Criteria

- **R wins:** R_simple or R_full achieves higher correlation/AUC than bare E on >= 3/4 datasets (p < 0.05 per comparison, Bonferroni corrected)
- **E wins:** Bare E achieves higher correlation/AUC than both R variants on >= 3/4 datasets
- **Tie:** Mixed results, no clear winner

### Baseline

- Random baseline (shuffled labels)
- Bare E (cosine similarity alone)

## Dependencies

- v2/Q1 (grad_S definition) should be reviewed first, but this test can proceed independently
- Uses v2/GLOSSARY.md E definition throughout

## Expected Outcome

Unknown. This is a genuine question. Q10's E>R finding might hold, in which case the formula needs rethinking. Or Q10 might have been an anomaly of that particular task.

## Related

- v2/Q1 (Why grad_S?)
- v2/Q10 (Alignment detection -- where E>R was first found)
- v2/Q20 (Tautology risk -- if R=E/sigma, is it just SNR?)
- N2 (What is grad_S? -- explains WHY E beats R, if it does)
