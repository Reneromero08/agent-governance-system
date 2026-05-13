# N1 Pre-Registration

Locked: 2026-03-09
Question: Does `R_simple = E / grad_S` outperform bare `E` on real NLP benchmarks?

## Prior Work Read Before Registration

- `THOUGHT/LAB/FORMULA/v3/questions/n01_e_vs_r/README.md`
- `THOUGHT/LAB/FORMULA/v2/q10_alignment/README.md`
- `THOUGHT/LAB/FORMULA/v2/q01_grad_s/README.md`
- `THOUGHT/LAB/FORMULA/v2/q20_tautology/README.md`
- `THOUGHT/LAB/FORMULA/v2/GLOSSARY.md`
- `THOUGHT/LAB/FORMULA/v2/METHODOLOGY.md`

## One-Sentence Pre-Registration

1. HYPOTHESIS: Bare `E` will match or beat `R_simple` on at least 3 of 4 fixed benchmark evaluations, and `R_full` will not beat `R_simple`.
2. PREDICTION: `AUC(E) - AUC(R_simple) >= 0.02` on at least 3 of 4 datasets, with bootstrap 95% CI for `AUC(E) - AUC(R_simple)` entirely above `0.0` on at least 2 of 4 datasets.
3. FALSIFICATION: This hypothesis is falsified if `R_simple` beats `E` with bootstrap 95% CI entirely above `0.0` on at least 3 of 4 datasets.
4. DATA SOURCE: HuggingFace datasets `glue/stsb` validation, `glue/sst2` validation, `snli` validation, `glue/mnli` validation_matched.
5. SUCCESS THRESHOLD: For a metric to "win" a dataset, its AUC advantage must have bootstrap 95% CI excluding `0.0`. For the overall hypothesis to hold, `E` must win at least 3 of 4 datasets.

## Fixed Design

No grid search. No changing cluster sizes after seeing results. No switching datasets after failures.

### Model

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Representation for single-sentence tasks: sentence embedding
- Representation for pair tasks: arithmetic mean of sentence-1 and sentence-2 embeddings

### Randomness

- Seed: `20260309`
- Bootstrap samples: `1000`

### Dataset Construction

- Cluster size: `16`
- Pure clusters per dataset: `120`
- Mixed clusters per dataset: `120`
- Evaluation label: `1 = pure`, `0 = mixed`

### Pure/Mixed Rules

- STS-B:
  - Pure cluster: all 16 examples from one human-score bin
  - Score bins: `[0,1)`, `[1,2)`, `[2,3)`, `[3,4)`, `[4,5.1]`
  - Mixed cluster: 4 examples from each of 4 distinct score bins
- SST-2:
  - Pure cluster: all 16 examples from one sentiment label
  - Mixed cluster: 8 positive + 8 negative
- SNLI:
  - Use only `entailment` and `contradiction`
  - Pure cluster: all 16 examples from one label
  - Mixed cluster: 8 entailment + 8 contradiction
- MNLI:
  - Use only `entailment` and `contradiction` from `validation_matched`
  - Pure cluster: all 16 examples from one label
  - Mixed cluster: 8 entailment + 8 contradiction

### Metrics Under Test

- `E`
- `R_simple = E / grad_S`
- `R_full = (E / grad_S) * sigma^Df`
- Random baseline

### Statistical Comparison

- Primary metric: AUC for pure-vs-mixed discrimination
- Uncertainty: paired bootstrap over clusters with `1000` resamples
- Comparisons:
  - `E` vs `R_simple`
  - `R_simple` vs `R_full`
  - each metric vs random baseline

## Anti-Pattern Checklist

- Ground truth is not derived from `E`, `R_simple`, or `R_full`: YES
- Parameters are fixed before running: YES
- No grid search for desired outcome: YES
- Negative result will be reported unchanged: YES
- No goalpost moving from the registered hypothesis: YES

## Stop Conditions

- Run the script once after the interpreter is repaired.
- If any dataset fails to load, record the failure and stop.
- If `R_full` produces non-finite values, record that and keep the result instead of silently dropping it.
