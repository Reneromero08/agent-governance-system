# N1 Exploratory Follow-Up

Date: 2026-03-09
Status: EXECUTED

The preregistered `N1` run on STS-B, SST-2, SNLI, and MNLI was inconclusive because the chosen cluster construction was a weak observable: both `E` and `R_simple` were near random on multiple datasets.

This follow-up is explicitly exploratory. It does **not** overwrite the preregistered result.

## Goal

Get a real boundary answer by testing `E` vs `R_simple` on datasets where class labels directly define semantic purity.

## Fixed Exploratory Design

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Datasets:
  - `ag_news` test
  - `dair-ai/emotion` test
  - `glue/sst2` validation
  - `snli` validation
  - `glue/mnli` validation_matched
- Cluster size: `16`
- Per-class cap: `800`
- Recipes:
  - binary datasets: pure, skewed `12/4`, balanced `8/8`
  - multi-class datasets: pure, skewed `12/4`, balanced-2 `8/8`, balanced-4 `4/4/4/4`
- Metrics:
  - Spearman correlation with actual cluster purity
  - AUC for `pure` vs `not pure`
  - bootstrap CI for `E - R_simple` on both statistics

## Interpretation Rule

- If `E` wins with CI excluding zero: `E` is better on that dataset.
- If `R_simple` wins with CI excluding zero: `R_simple` is better on that dataset.
- Otherwise: tie on that dataset.

## What This Follow-Up Can Answer

- Whether `R_simple` helps on high-heterogeneity topic tasks
- Whether `E` dominates on compact/binary tasks
- Whether the answer to N1 is "depends on dataset geometry" rather than a universal winner

## Outcome

Outputs:

- `results/n01_label_purity_followup.json`
- `results/n01_label_purity_followup.md`

Observed result:

- `ag_news`: `E` beats `R_simple` decisively
  - Spearman delta `0.0641 [0.0295, 0.1065]`
  - AUC delta `0.0311 [0.0043, 0.0612]`
- `emotion`: tie
- `sst2`: tie
- `snli`: tie
- `mnli`: tie

Bottom line:

- No dataset in this follow-up shows `R_simple` beating `E`.
- One dataset (`ag_news`) shows `E` beating `R_simple` with confidence.
- Four datasets are ties.

Practical answer:

- `R_simple` is not a generally stronger metric than bare `E`.
- If anything, the current evidence points the other way: `E` is at least as good everywhere tested here and clearly better on `ag_news`.
