# Relation R-Spatial Full Schedule Source-Off Reset Crosscheck Discovery

Run ID: `family10h_relation_rspatial_full_schedule_source_off_reset_crosscheck_discovery_v1_0`
Archive SHA-256: `bd83588936568d4b7068fa52e62fd35be18ccb904ad9373ff30124e66e5b2d84`
Analysis SHA-256: `35a3f25e4b32bd7b3b9b10dde85b912ddcfd68b88f1c347f54f84ec166013cbe`

| Method | Primary R | Sham R | D primary-minus-sham | abs(D)/abs(alive D) | max stratum ratio | all strata <= 0.25 |
|---|---:|---:|---:|---:|---:|---:|
| alive | 0.065496687 | -0.001952647 | 0.067449334 | 1.000 | 1.000 | False |
| dead | 0.035943278 | -0.002100590 | 0.038043868 | 0.564 | 0.766 | False |
| reset_double_flush | 0.002079935 | -0.010985203 | 0.013065138 | 0.194 | 0.505 | False |

Discovery interpretation:
- alive D positive: `True`
- dead D positive: `True`
- source death kills below 0.25 x alive D: `False`
- dead all one-factor strata positive: `True`
- source-off kills aggregate below 0.25 x alive D: `False`
- source-off kills all one-factor strata below 0.25 x alive D: `False`
- best reset method: `reset_double_flush`
- best reset kills aggregate below 0.25 x alive D: `True`
- best reset kills all one-factor strata below 0.25 x alive D: `False`
- best reset reduces dead D abs: `True`
- persistent source-written-state candidate with reset method: `False`

This is exploratory evidence only. It emits no prospective scientific claim and does not promote `SMALL_WALL_CROSSED`.
