# QEC Precision Sweep v2 -- Evaluator Report

Report id: `smoke_eval_v2_depol`
UTC time: `2026-05-13T09:56:41.769982+00:00`
Frozen p_th: `0.007071067811865475`

## Overall Cross-Model Verdict

**AMBIGUOUS**

## Noise Model: depol

Run id: `smoke_api_check_v2_depol`
Per-model verdict: **AMBIGUOUS**

### Model Metrics

| Model | Train MAE | Test MAE | 95% CI | Train R2 | Test R2 |
|---|---:|---:|---|---:|---:|
| `formula_score` | 0.0000 | 0.6780 | [0.2716, 1.0844] | 1.0000 | -1.5119 |
| `formula_components` | 0.0000 | 0.6990 | [0.3346, 1.0634] | 1.0000 | -1.4980 |
| `p_only` | 0.0000 | 0.8069 | [0.6583, 0.9555] | 1.0000 | -1.7061 |
| `distance_only` | 0.6474 | 0.8069 | [0.3081, 1.3057] | 0.0000 | -2.6174 |
| `standard_qec_scaling` | 0.0000 | 2.2406 | [1.8762, 2.6050] | 1.0000 | -19.7145 |

### Pass/Fail Criteria

**Pass conditions met:**
- formula_score MAE (0.6780) <= standard_qec_scaling MAE * 1.05 (2.3526)
- formula_score MAE (0.6780) < p_only MAE (0.8069)

**Fail conditions triggered:**
- formula_components R2 (-1.4980) < 0.0

## Falsification Check

**Falsified: False**

## Confirmation Check

**Confirmed: False**
Reason: Second noise model not run


## Preregistration

The pass/fail criteria applied here are defined in:
`THOUGHT/LAB/FORMULA/v4/qec_precision_sweep/PREREGISTRATION_v2.md`

No post-hoc remapping occurred. The frozen p_th = 0.007071067811865475
was fixed before any v2 sweep ran.

