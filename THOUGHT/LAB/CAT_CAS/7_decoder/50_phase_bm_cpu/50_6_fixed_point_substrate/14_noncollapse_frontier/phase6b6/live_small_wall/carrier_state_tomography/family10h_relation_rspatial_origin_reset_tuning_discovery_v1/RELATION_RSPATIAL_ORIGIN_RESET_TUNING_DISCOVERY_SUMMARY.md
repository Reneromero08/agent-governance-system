# Relation R-Spatial Origin Reset Tuning Discovery

Run ID: `family10h_relation_rspatial_origin_reset_tuning_discovery_v1_0`
Archive SHA-256: `eb06cf068cf930d07d86771b02b93623bfdfea73bfeb7343b088b76a6a82d058`
Analysis SHA-256: `17f086450b216cd5ba0a3f417f8c3a06c62ee7d6cac1da49369bc0c5c9a2d948`

| Method | Primary R | Sham R | D primary-minus-sham | abs(D)/abs(alive D) | max stratum ratio | all strata <= 0.25 |
|---|---:|---:|---:|---:|---:|---:|
| alive | 0.000703925 | -0.001309360 | 0.002013285 | 1.000 | 1.000 | False |
| dead | 0.001847787 | 0.007394256 | -0.005546469 | 2.755 | 1.227 | False |
| reset_flush | 0.001289088 | -0.002594455 | 0.003883543 | 1.929 | 1.247 | False |
| reset_prefault_flush | -0.003470089 | -0.000919750 | -0.002550339 | 1.267 | 0.973 | False |
| reset_double_flush | -0.000512234 | 0.001798197 | -0.002310431 | 1.148 | 0.817 | False |
| reset_prefault_double_flush | -0.001976719 | 0.000241497 | -0.002218216 | 1.102 | 1.069 | False |
| reset_prefault_triple_flush | -0.002862713 | 0.003150526 | -0.006013239 | 2.987 | 2.083 | False |
| reset_lane_a_flush | 0.001845668 | -0.000592445 | 0.002438113 | 1.211 | 1.998 | False |

Discovery interpretation:
- alive D positive: `True`
- dead D positive: `False`
- source death kills below 0.25 x alive D: `False`
- dead all one-factor strata positive: `False`
- best reset method: `reset_double_flush`
- best reset kills aggregate below 0.25 x alive D: `False`
- best reset kills all one-factor strata below 0.25 x alive D: `False`
- best reset reduces dead D abs: `True`
- wrong lane-A reset also kills all strata: `False`
- persistent source-written-state candidate with reset method: `False`

This is exploratory evidence only. It emits no prospective scientific claim and does not promote `SMALL_WALL_CROSSED`.
