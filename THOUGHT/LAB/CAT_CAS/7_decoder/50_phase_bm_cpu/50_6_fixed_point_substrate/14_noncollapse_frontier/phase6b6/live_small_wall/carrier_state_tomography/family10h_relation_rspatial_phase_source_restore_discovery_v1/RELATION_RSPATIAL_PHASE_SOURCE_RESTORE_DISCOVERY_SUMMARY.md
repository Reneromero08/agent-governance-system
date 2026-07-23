# Relation R-Spatial Phase Source-Restore Discovery

Run ID: `family10h_relation_rspatial_phase_source_restore_discovery_v1_0`
Archive SHA-256: `1b4d5d6c6e4c57d1fcf58a03a24005aeb74d2e21b0b9921191d96551f96b92b0`
Analysis SHA-256: `7ecdd52cf1adb19b8df007610f9b9bd1f33237144f27e642915f99b5cff899f7`

| Method | Primary R | Sham R | D primary-minus-sham | abs(D)/abs(alive D) | max stratum ratio | all strata <= 0.25 |
|---|---:|---:|---:|---:|---:|---:|
| alive | 0.072281074 | 0.023249986 | 0.049031088 | 1.000 | 1.000 | False |
| dead | 0.030497046 | -0.008053936 | 0.038550982 | 0.786 | 1.206 | False |
| reset_double_flush | -0.011032768 | 0.000803345 | -0.011836113 | 0.241 | 1.503 | False |
| restore_same | 0.002429590 | -0.007338937 | 0.009768527 | 0.199 | 1.605 | False |
| restore_opposite | -0.003502237 | -0.007841610 | 0.004339373 | 0.089 | 0.812 | False |

| Method | prep/query phase rad | prep/query mag ratio | diag/skew phase rad | diag/skew mag ratio |
|---|---:|---:|---:|---:|
| alive | -0.734917 | 1.000 | 1.000626 | 1.000 |
| dead | -0.764185 | 0.472 | 0.753013 | 0.582 |
| reset_double_flush | 3.092983 | 0.414 | -2.038188 | 0.289 |
| restore_same | -1.656115 | 0.169 | 0.705016 | 0.141 |
| restore_opposite | 0.594344 | 0.279 | 0.751784 | 0.065 |

Discovery interpretation:
- alive D positive: `True`
- dead D positive: `True`
- source death kills below 0.25 x alive D: `False`
- dead all one-factor strata positive: `True`
- best reset method: `reset_double_flush`
- best reset kills aggregate below 0.25 x alive D: `True`
- best reset kills all one-factor strata below 0.25 x alive D: `False`
- best reset reduces dead D abs: `True`
- persistent source-written-state candidate with reset method: `False`
- restore-same rebuilds positive D: `True`
- restore-opposite flips D negative: `False`
- phase source-restore candidate: `False`

This is exploratory evidence only. It emits no prospective scientific claim and does not promote `SMALL_WALL_CROSSED`.
