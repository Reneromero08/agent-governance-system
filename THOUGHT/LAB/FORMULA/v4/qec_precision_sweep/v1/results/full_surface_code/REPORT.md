# QEC Precision Sweep Report

Run id: `full_surface_code_v1`
UTC time: `2026-05-13T09:13:18.692295+00:00`
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

- Conditions: `48`
- Total shots: `960000`
- Distances: `[3, 5, 7, 9]`
- Held-out distances: `[7, 9]`
- Physical error rates: `[0.001, 0.002, 0.005, 0.01, 0.02, 0.04]`
- Bases: `['x', 'z']`

## Model Comparison

| Model | Train MAE | Test MAE | Train R2 | Test R2 |
|---|---:|---:|---:|---:|
| `standard_qec_scaling` | 0.4078 | 0.9832 | 0.8590 | 0.7609 |
| `formula_components` | 0.4807 | 1.0438 | 0.8128 | 0.7512 |
| `p_only` | 0.5841 | 1.2510 | 0.7511 | 0.6269 |
| `formula_score` | 0.5339 | 1.6255 | 0.7811 | 0.4585 |
| `distance_only` | 1.2253 | 2.3607 | 0.0198 | -0.0032 |

## Verdict

Best held-out model by MAE: `standard_qec_scaling`.

The preregistered formula mapping did not beat the strongest held-out baseline in this run.

This is evidence about the current QEC domain mapping, not proof of the whole formula.
A stronger result would require larger sweeps, more noise models, and a frozen mapping carried across them.

## Files

- Raw condition CSV: `results\full_surface_code_v1\conditions.csv`
- Full JSON payload: `results\full_surface_code_v1\qec_precision_sweep.json`
- This report: `REPORT.md`
