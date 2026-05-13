# QEC Precision Sweep v2 -- Evaluator Report

Report id: `qec_v2_cross_eval`
UTC time: `2026-05-13T09:58:03.995140+00:00`
Frozen p_th: `0.007071067811865475`

## Overall Cross-Model Verdict

**MIXED**

## Noise Model: depol

Run id: `qec_v2_depol`
Per-model verdict: **PASS**

### Model Metrics

| Model | Train MAE | Test MAE | 95% CI | Train R2 | Test R2 |
|---|---:|---:|---|---:|---:|
| `formula_score` | 0.4170 | 0.8252 | [0.5889, 1.0807] | 0.8736 | 0.8048 |
| `standard_qec_scaling` | 0.3661 | 0.8422 | [0.6072, 1.0818] | 0.8951 | 0.8030 |
| `formula_components` | 0.3662 | 0.8603 | [0.6026, 1.1330] | 0.8936 | 0.7751 |
| `p_only` | 0.4929 | 1.2076 | [0.8892, 1.5530] | 0.8299 | 0.6088 |
| `distance_only` | 1.2259 | 2.2734 | [1.8970, 2.6111] | 0.0217 | 0.0016 |

### Pass/Fail Criteria

**Pass conditions met:**
- formula_score MAE (0.8252) <= standard_qec_scaling MAE * 1.05 (0.8843)
- formula_score MAE (0.8252) < p_only MAE (1.2076)
- formula_components R2 (0.7751) >= 0.0

## Noise Model: meas

Run id: `qec_v2_meas`
Per-model verdict: **FAIL**

### Model Metrics

| Model | Train MAE | Test MAE | 95% CI | Train R2 | Test R2 |
|---|---:|---:|---|---:|---:|
| `standard_qec_scaling` | 0.2201 | 1.0852 | [0.7164, 1.5669] | 0.9678 | 0.4992 |
| `formula_score` | 0.4701 | 1.4811 | [1.1934, 1.7976] | 0.8773 | 0.4897 |
| `p_only` | 0.5933 | 1.5707 | [1.1545, 2.0089] | 0.7962 | 0.3107 |
| `formula_components` | 0.4122 | 1.8363 | [1.5124, 2.1748] | 0.9103 | 0.2519 |
| `distance_only` | 1.2694 | 2.1183 | [1.6787, 2.5853] | 0.1043 | -0.0676 |

### Pass/Fail Criteria

**Pass conditions met:**
- formula_score MAE (1.4811) < p_only MAE (1.5707)
- formula_components R2 (0.2519) >= 0.0

**Fail conditions triggered:**
- formula_score MAE (1.4811) > standard_qec_scaling MAE * 1.05 (1.1394)
- standard_qec_scaling MAE (1.0852) < formula_score MAE * 0.90 (1.3330)

## Falsification Check

**Falsified: False**

### depol
- formula_score / standard_qec MAE ratio: 0.9799
- fs > 1.5 * sqec: False
- fc R2 < 0: False
- p_only better than fs: False
- All three falsification conditions: False

### meas
- formula_score / standard_qec MAE ratio: 1.3649
- fs > 1.5 * sqec: False
- fc R2 < 0: False
- p_only better than fs: False
- All three falsification conditions: False

## Confirmation Check

**Confirmed: False**
Reason: Both noise models must PASS


## Preregistration

The pass/fail criteria applied here are defined in:
`THOUGHT/LAB/FORMULA/v4/qec_precision_sweep/PREREGISTRATION_v2.md`

No post-hoc remapping occurred. The frozen p_th = 0.007071067811865475
was fixed before any v2 sweep ran.

