# Relation Fixed-Marginal Order Discovery

Run ID: `family10h_relation_fixed_marginal_order_discovery_v1_0`
Archive SHA-256: `7a3a208de21b377b2f3b0c9202b19c6e75b510e29d95b7e3e7a2ded21dd91963`
Analysis SHA-256: `47bdf5e9fa48ffe28da904a782c128d379aaa78c016b1a7a499385b60d098276`

| Pair | Primary R | Sham R | D primary-minus-sham |
|---|---:|---:|---:|
| ab | 0.068704935 | -0.003769913 | 0.072474848 |
| ba | 0.055989850 | -0.006048924 | 0.062038774 |
| dead_ab | 0.007823122 | -0.008970911 | 0.016794033 |
| dead_ba | 0.022966489 | -0.007010010 | 0.029976500 |
| source_off_ab | 0.112698382 | -0.004423530 | 0.117121912 |
| source_off_ba | 0.113426444 | 0.002247243 | 0.111179201 |
| neutral | 0.053333205 | -0.006899493 | 0.060232698 |
| random | 0.060933442 | 0.019611079 | 0.041322362 |

Order coordinates:
- alive D order AB minus BA: `0.010436074`
- dead D order AB minus BA: `-0.013182467`
- source-off D order AB minus BA: `0.005942711`
- source-off abs/alive abs: `0.569`
- dead abs/alive abs: `1.263`

Interpretation:
- fixed-marginal order candidate: `False`
- source-off collapses below 0.25 x alive order: `False`
- dead preserves at least 0.25 x alive order: `True`
- one-factor strata same sign: `True`

This is exploratory evidence only. It emits no positive scientific claim and does not promote `SMALL_WALL_CROSSED`.
