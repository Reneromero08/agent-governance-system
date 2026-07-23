# Relation R-Spatial Phase Reset Reuse Discovery

Run ID: `family10h_relation_rspatial_phase_reset_reuse_discovery_v1_0`
Archive SHA-256: `3b419aa7972d1c59377f0760cc64604098cf9ea4c18ecd7dee0573ee23f0c3f5`
Analysis SHA-256: `8641781645975eb7399b1ff199b437de8149d767248486f72aae07d85f3901cd`

| Method | Primary R | Sham R | D primary-minus-sham | abs(D)/abs(alive D) | max stratum ratio | all strata <= 0.25 |
|---|---:|---:|---:|---:|---:|---:|
| alive | 0.059115784 | -0.032260763 | 0.091376547 | 1.000 | 1.000 | False |
| dead | 0.042990277 | 0.000923901 | 0.042066375 | 0.460 | 0.565 | False |
| reset_double_flush | 0.003393021 | -0.024230894 | 0.027623915 | 0.302 | 0.435 | False |
| restore_same | -0.006907413 | 0.009901311 | -0.016808724 | 0.184 | 0.289 | False |
| restore_opposite | 0.006324063 | 0.007522203 | -0.001198140 | 0.013 | 0.177 | True |

| Method | prep/query phase rad | prep/query mag ratio | diag/skew phase rad | diag/skew mag ratio |
|---|---:|---:|---:|---:|
| alive | -1.051885 | 1.000 | 0.742518 | 1.000 |
| dead | -0.996827 | 0.271 | 0.501431 | 0.387 |
| reset_double_flush | -2.963830 | 0.168 | -0.292808 | 0.233 |
| restore_same | -1.912491 | 0.221 | 2.686148 | 0.151 |
| restore_opposite | 2.785075 | 0.018 | -2.268914 | 0.015 |

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
- restore-same rebuilds positive D: `False`
- restore-opposite flips D negative: `True`
- phase reset/reuse candidate: `False`

This is exploratory evidence only. It emits no prospective scientific claim and does not promote `SMALL_WALL_CROSSED`.
