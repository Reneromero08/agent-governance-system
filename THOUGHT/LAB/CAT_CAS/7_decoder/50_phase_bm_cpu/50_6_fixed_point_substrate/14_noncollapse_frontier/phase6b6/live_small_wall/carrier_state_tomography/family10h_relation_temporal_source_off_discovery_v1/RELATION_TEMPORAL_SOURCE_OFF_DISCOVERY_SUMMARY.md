# Relation Temporal Source-Off Discovery

Run ID: `family10h_relation_temporal_source_off_discovery_v1_0`
Archive SHA-256: `13224a796dd2cf4539b6db37eb2a88cb6bde2ede9b3ebed38172ddf5aef91405`
Analysis SHA-256: `d185efbe95d8ca54ce4b01fa6ef87f6695131f8a34388cbe5b747a92e72942cd`

| Method | Primary R | Sham R | D primary-minus-sham | abs(D)/abs(alive D) | max stratum ratio | all strata <= 0.25 |
|---|---:|---:|---:|---:|---:|---:|
| alive | 0.030529443 | -0.001683890 | 0.032213334 | 1.000 | 1.000 | False |
| source_off | 0.118397424 | -0.004027911 | 0.122425335 | 3.800 | 9.764 | False |

Discovery interpretation:
- alive D positive: `True`
- source-off D positive: `True`
- source-off kills below 0.25 x alive D: `False`
- source-off all one-factor strata below 0.25 x alive D: `False`
- source-written temporal candidate: `False`

This is exploratory evidence only. It emits no prospective scientific claim and does not promote `SMALL_WALL_CROSSED`.
