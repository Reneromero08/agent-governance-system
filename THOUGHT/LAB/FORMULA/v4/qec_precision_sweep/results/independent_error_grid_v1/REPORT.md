# QEC Precision Sweep Report

Run id: `independent_error_grid_v1`
UTC time: `2026-05-13T09:28:45.814497+00:00`
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

- Conditions: `56`
- Total shots: `1120000`
- Distances: `[3, 5, 7, 9]`
- Held-out distances: `[7, 9]`
- Physical error rates: `[0.00075, 0.0015, 0.003, 0.006, 0.012, 0.024, 0.048]`
- Bases: `['x', 'z']`

## Model Comparison

| Model | Train MAE | Test MAE | Train R2 | Test R2 |
|---|---:|---:|---:|---:|
| `standard_qec_scaling` | 0.3519 | 0.8740 | 0.8988 | 0.8052 |
| `formula_components` | 0.3998 | 0.9946 | 0.8652 | 0.7613 |
| `p_only` | 0.5449 | 1.3232 | 0.7960 | 0.5828 |
| `formula_score` | 0.4537 | 1.3555 | 0.8419 | 0.5696 |
| `distance_only` | 1.1900 | 2.3544 | 0.0235 | -0.0020 |

## Verdict

Best held-out model by MAE: `standard_qec_scaling`.

The preregistered formula mapping did not beat the strongest held-out baseline in this run.

This is evidence about the current QEC domain mapping, not proof of the whole formula.
A stronger result would require larger sweeps, more noise models, and a frozen mapping carried across them.

## Files

- Raw condition CSV: `results\independent_error_grid_v1\conditions.csv`
- Full JSON payload: `results\independent_error_grid_v1\qec_precision_sweep.json`
- This report: `REPORT.md`
