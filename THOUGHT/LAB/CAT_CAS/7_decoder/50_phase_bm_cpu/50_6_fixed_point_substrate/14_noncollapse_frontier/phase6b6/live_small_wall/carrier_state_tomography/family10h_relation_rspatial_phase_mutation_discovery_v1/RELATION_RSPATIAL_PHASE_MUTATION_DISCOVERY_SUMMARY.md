# Relation R-Spatial Phase Mutation Discovery

Run ID: `family10h_relation_rspatial_phase_mutation_discovery_v1_0`
Archive SHA-256: `bf2124cef1ab747e6fa69fa794b5b3b4e284cf2441d6cf8280f0b1a1ad97b969`
Analysis SHA-256: `3bcef8ff4470413d7ab35e84f9131a95d4de8e2052734e4b047928ec3db5c021`

| Method | Primary R | Sham R | D primary-minus-sham | abs(D)/abs(alive D) | max stratum ratio | all strata <= 0.25 |
|---|---:|---:|---:|---:|---:|---:|
| alive | 0.074075559 | 0.006326067 | 0.067749492 | 1.000 | 1.000 | False |
| dead | 0.039950541 | -0.014699625 | 0.054650166 | 0.807 | 1.116 | False |
| reset_double_flush | 0.011861280 | -0.017078544 | 0.028939824 | 0.427 | 0.620 | False |
| mutate_same | 0.073939491 | -0.009108821 | 0.083048312 | 1.226 | 1.555 | False |
| mutate_opposite | 0.063733166 | -0.004267678 | 0.068000845 | 1.004 | 1.393 | False |

| Method | prep/query phase rad | prep/query mag ratio | diag/skew phase rad | diag/skew mag ratio |
|---|---:|---:|---:|---:|
| alive | -1.033756 | 1.000 | 0.887792 | 1.000 |
| dead | -0.446822 | 0.481 | 0.619964 | 0.626 |
| reset_double_flush | -2.899312 | 0.373 | -0.520105 | 0.311 |
| mutate_same | -0.580589 | 0.452 | 0.429258 | 0.851 |
| mutate_opposite | -0.674650 | 0.728 | 0.740703 | 0.858 |

Discovery interpretation:
- alive D positive: `True`
- dead D positive: `True`
- source death kills below 0.25 x alive D: `False`
- dead all one-factor strata positive: `True`
- best reset method: `reset_double_flush`
- best reset kills aggregate below 0.25 x alive D: `False`
- best reset kills all one-factor strata below 0.25 x alive D: `False`
- best reset reduces dead D abs: `True`
- persistent source-written-state candidate with reset method: `False`
- mutate-same rebuilds positive D: `True`
- mutate-opposite flips D negative: `False`
- phase mutation candidate: `False`

This is exploratory evidence only. It emits no prospective scientific claim and does not promote `SMALL_WALL_CROSSED`.
