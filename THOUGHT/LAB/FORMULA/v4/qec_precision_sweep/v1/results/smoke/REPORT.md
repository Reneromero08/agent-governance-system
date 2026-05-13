# QEC Precision Sweep Report

Run id: `smoke_api_check`
UTC time: `2026-05-13T09:12:27.237442+00:00`
Git revision: `bb04bbf7293cfe963da2b9d69311287498b49e31`

## Preregistered Mapping

- `R`: logical-error suppression, `physical_error_rate / logical_error_rate_laplace`.
- `E`: normalized initial logical-state integrity, fixed at `1.0` for this first sweep.
- `grad_S`: binary entropy of the physical error rate, `H2(p)`.
- `sigma`: entropy-to-correction efficiency proxy, `1 / sqrt(H2(p))`.
- `Df`: surface-code distance `d`.

Formula score:

```text
R_hat = (E / grad_S) * sigma ** Df
```

## Sweep

- Conditions: `4`
- Total shots: `400`
- Distances: `[3, 5]`
- Held-out distances: `[5]`
- Physical error rates: `[0.005, 0.02]`
- Bases: `['x']`

## Model Comparison

| Model | Train MAE | Test MAE | Train R2 | Test R2 |
|---|---:|---:|---:|---:|
| `p_only` | 0.0000 | 0.6962 | 1.0000 | -0.1159 |
| `distance_only` | 1.1636 | 0.7612 | 0.0000 | -0.8364 |
| `formula_components` | 0.0000 | 2.4197 | 1.0000 | -10.1819 |
| `formula_score` | 0.0000 | 2.7644 | 1.0000 | -13.4877 |
| `standard_qec_scaling` | 0.0000 | 3.2732 | 1.0000 | -18.5677 |

## Verdict

Best held-out model by MAE: `p_only`.

The preregistered formula mapping did not beat the strongest held-out baseline in this run.

This is evidence about the current QEC domain mapping, not proof of the whole formula.
A stronger result would require larger sweeps, more noise models, and a frozen mapping carried across them.

## Files

- Raw condition CSV: `results\smoke_api_check\conditions.csv`
- Full JSON payload: `results\smoke_api_check\qec_precision_sweep.json`
- This report: `REPORT.md`
