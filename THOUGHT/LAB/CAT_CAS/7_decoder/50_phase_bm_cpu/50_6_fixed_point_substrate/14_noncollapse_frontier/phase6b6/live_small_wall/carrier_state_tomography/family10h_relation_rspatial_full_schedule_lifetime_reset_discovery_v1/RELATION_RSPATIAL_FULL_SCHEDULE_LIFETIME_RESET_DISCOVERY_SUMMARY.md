# Relation R-Spatial Full Schedule Lifetime Reset Discovery

Run ID: `family10h_relation_rspatial_full_schedule_lifetime_reset_discovery_v1_0`
Archive SHA-256: `e4689f226695ef0d44d75fda0fe07c230a60f5bb4e4d8e3b943e8bfc979df6f7`
Analysis SHA-256: `932ec23f3d507a9dc7a1908da527341c6cb0ff7842a4a1daf742fe11907c4c40`

| Method | Primary R | Sham R | D primary-minus-sham | abs(D)/abs(alive D) | max stratum ratio | all strata <= 0.25 |
|---|---:|---:|---:|---:|---:|---:|
| alive | 0.066774616 | -0.032292981 | 0.099067597 | 1.000 | 1.000 | False |
| dead | 0.016027892 | -0.009659396 | 0.025687289 | 0.259 | 0.408 | False |
| reset_prefault_flush | -0.002020601 | 0.012861359 | -0.014881960 | 0.150 | 0.289 | False |
| reset_double_flush | -0.005821925 | -0.013917581 | 0.008095657 | 0.082 | 0.204 | True |

Discovery interpretation:
- alive D positive: `True`
- dead D positive: `True`
- source death kills below 0.25 x alive D: `False`
- dead all one-factor strata positive: `True`
- best reset method: `reset_double_flush`
- best reset kills aggregate below 0.25 x alive D: `True`
- best reset kills all one-factor strata below 0.25 x alive D: `True`
- best reset reduces dead D abs: `True`
- prefault+flush kills all strata: `False`
- persistent source-written-state candidate with reset method: `True`

This is exploratory evidence only. It emits no prospective scientific claim and does not promote `SMALL_WALL_CROSSED`.
