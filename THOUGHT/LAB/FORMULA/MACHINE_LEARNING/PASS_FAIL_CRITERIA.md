# Pass / Fail Criteria

## Experiment 01 Standard

Task:
discriminate externally labeled pure clusters from mixed clusters.

Primary metric:
ROC AUC.

### Confirmed Utility

For a given dataset/model pair:

- `R_simple` or `R_full` achieves AUC >= 0.70
- and beats `E` by at least 0.03 AUC
- and beats at least three other non-random baselines

### Weak Positive

- `R_simple` or `R_full` achieves AUC >= 0.65
- and is statistically tied with the best baseline

### Falsified Utility In That Setting

- both `R_simple` and `R_full` are below 0.60 AUC
- or `E` beats them by >= 0.03 AUC
- or a simpler baseline wins consistently across reruns

## Program-Level ML Evidence

The ML mapping becomes interesting only if:

1. at least two datasets show confirmed or weak-positive utility;
2. the effect survives a model swap;
3. no term redefinition was needed.

Otherwise the responsible conclusion is that the current ML operationalization
is not useful enough yet.
