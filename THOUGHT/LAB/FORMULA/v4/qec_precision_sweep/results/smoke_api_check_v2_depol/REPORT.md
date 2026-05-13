# QEC Precision Sweep v2 -- Raw Sweep Report

Run id: `smoke_api_check_v2_depol`
Noise model: `depol`
UTC time: `2026-05-13T09:56:28.961360+00:00`
Git revision: `bb04bbf7293cfe963da2b9d69311287498b49e31`

## Preregistered Mapping (v2, frozen p_th)

- `E`: `1 - p` (initial single-qubit survival probability)
- `grad_S`: `p / p_th` with `p_th = 0.007071067811865475`
- `sigma`: `sqrt(p_th / p)`
- `Df`: surface-code distance `d`
- `R`: `ln(physical_error_rate / logical_error_rate_laplace)`

Formula score:

```text
R_hat = (E / grad_S) * sigma ** Df
```

## Sweep

- Conditions: `4`
- Total shots: `800`
- Distances: `[3, 5]`
- Held-out distances: `[5]`
- Physical error rates: `[0.005, 0.02]`
- Bases: `['x']`

## Preliminary Model Comparison (built-in evaluation)

| Model | Train MAE | Test MAE | Train R2 | Test R2 |
|---|---:|---:|---:|---:|
| `formula_score_v2` | 0.0000 | 0.6780 | 1.0000 | -1.5119 |
| `formula_components_v2` | 0.0000 | 0.6990 | 1.0000 | -1.4980 |
| `p_only` | 0.0000 | 0.8069 | 1.0000 | -1.7061 |
| `distance_only` | 0.6474 | 0.8069 | 0.0000 | -2.6174 |
| `formula_components_v1` | 0.0000 | 1.7657 | 1.0000 | -12.0670 |
| `formula_score_v1` | 0.0000 | 1.9575 | 1.0000 | -15.0711 |
| `standard_qec_scaling` | 0.0000 | 2.2406 | 1.0000 | -19.7145 |

## Preliminary Verdict (built-in evaluation only)

Best held-out model by MAE: `formula_score_v2`.

Note: The authoritative evaluation uses the separate evaluator script
(`evaluate_qec_v2.py`) which applies the full pass/fail criteria from
PREREGISTRATION_v2.md, including bootstrap confidence intervals.

## Files

- Raw condition CSV: `results\smoke_api_check_v2_depol\conditions.csv`
- Full JSON payload: `results\smoke_api_check_v2_depol\qec_precision_sweep_v2.json`
- This report: `REPORT.md`
