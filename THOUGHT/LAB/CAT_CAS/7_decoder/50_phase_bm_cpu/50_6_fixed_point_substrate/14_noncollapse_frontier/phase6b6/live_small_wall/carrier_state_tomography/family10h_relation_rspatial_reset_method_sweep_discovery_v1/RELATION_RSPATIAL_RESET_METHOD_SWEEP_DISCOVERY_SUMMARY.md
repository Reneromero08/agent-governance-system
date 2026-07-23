# Relation R-Spatial Reset Method Sweep Discovery

Run ID: `family10h_relation_rspatial_reset_method_sweep_discovery_v1_0`
Archive SHA-256: `602deff9a6d0143ff3e25c53fee23108d63b68e97bc8c1afc18eb1bacfda8f8a`
Analysis SHA-256: `9447cec15cf1a6ce2b7149830fb7249c6d532e97a77bdc3438f0e1bac2a8540d`

| Method | Primary R | Sham R | D primary-minus-sham | abs(D)/abs(alive D) | max stratum ratio | all strata <= 0.25 |
|---|---:|---:|---:|---:|---:|---:|
| alive | 0.077981642 | -0.000966425 | 0.078948067 | 1.000 | 1.000 | False |
| dead | 0.021298466 | -0.016608916 | 0.037907382 | 0.480 | 0.935 | False |
| reset_flush | 0.005448627 | -0.026385078 | 0.031833705 | 0.403 | 1.730 | False |
| reset_prefault_flush | 0.009888845 | 0.011237586 | -0.001348741 | 0.017 | 0.577 | False |
| reset_double_flush | -0.006977819 | 0.011098189 | -0.018076008 | 0.229 | 0.468 | False |
| reset_lane_a_flush | 0.026899312 | -0.020245300 | 0.047144612 | 0.597 | 1.104 | False |

Discovery interpretation:
- alive D positive: `True`
- dead D positive: `True`
- source death kills below 0.25 x alive D: `False`
- dead all one-factor strata positive: `True`
- best reset method: `reset_double_flush`
- best reset kills aggregate below 0.25 x alive D: `True`
- best reset kills all one-factor strata below 0.25 x alive D: `False`
- best reset reduces dead D abs: `True`
- wrong lane-A reset also kills all strata: `False`
- persistent source-written-state candidate with reset method: `False`

This is exploratory evidence only. It emits no prospective scientific claim and does not promote `SMALL_WALL_CROSSED`.
