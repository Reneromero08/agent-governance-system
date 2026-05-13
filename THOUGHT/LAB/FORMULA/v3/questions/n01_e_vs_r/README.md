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

## Current Status

Status: EXECUTED ON 2026-03-09

Paper trail:
- `PREREGISTRATION.md` -- locked hypothesis, prediction, falsification criteria, and fixed parameters
- `RUNLOG.md` -- prior work reviewed, local environment notes, exact execution command, and run outcome
- `code/test_n01_e_vs_r.py` -- executable harness for the one-shot evaluation
- `results/n01_e_vs_r_results.json` -- raw metrics and bootstrap deltas
- `results/n01_e_vs_r_report.md` -- human-readable result summary

Prior work reviewed for this pass:
- `THOUGHT/LAB/FORMULA/v2/q10_alignment/README.md`
- `THOUGHT/LAB/FORMULA/v2/q01_grad_s/README.md`
- `THOUGHT/LAB/FORMULA/v2/q20_tautology/README.md`
- `THOUGHT/LAB/FORMULA/v2/GLOSSARY.md`
- `THOUGHT/LAB/FORMULA/v2/METHODOLOGY.md`

This pass used one fixed design across four external benchmarks:
- STS-B validation
- SST-2 validation
- SNLI validation
- MNLI validation_matched

Each benchmark is converted into externally labeled pure-vs-mixed clusters with fixed sampling rules, then `E`, `R_simple`, and `R_full` are compared by AUC on that purity task. No parameter search is allowed. If `E` wins, that is the result.

Observed result from the registered run:
- `E` wins: 0 datasets
- `R_simple` wins: 0 datasets
- ties: 4 datasets
- overall status: INCONCLUSIVE / mixed

Under this fixed test design, `E` and `R_simple` were statistically indistinguishable on all four datasets.
