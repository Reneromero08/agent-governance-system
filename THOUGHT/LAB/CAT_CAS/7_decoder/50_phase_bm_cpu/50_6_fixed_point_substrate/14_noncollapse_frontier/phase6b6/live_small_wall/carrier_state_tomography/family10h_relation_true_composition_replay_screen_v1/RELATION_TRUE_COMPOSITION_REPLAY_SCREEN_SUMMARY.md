# Relation Composition Replay-Exclusion Discovery

Run ID: `family10h_relation_true_composition_replay_screen_v1_0`
Archive SHA-256: `78c97be57e38ccc470453229690f5237803a212c22871cd60a65730a20d067e7`
Analysis SHA-256: `f0a0295074f48a0d5f5fcfae57c87ea2a2475db18eea0577463c3fe5dfd57734`

| Pair | Primary R | Sham R | D primary-minus-sham |
|---|---:|---:|---:|
| adjacent_r0r1 | 0.091543946 | 0.026089405 | 0.065454541 |
| adjacent_r1r0 | 0.068023130 | -0.013236929 | 0.081260059 |
| dead_adjacent_r0r1 | 0.074904616 | -0.009394909 | 0.084299524 |
| dead_adjacent_r1r0 | 0.098427076 | 0.008560341 | 0.089866736 |
| source_off_adjacent_r0r1 | 0.109491249 | -0.001454957 | 0.110946206 |
| source_off_adjacent_r1r0 | 0.117677811 | 0.004271313 | 0.113406499 |
| gapped_r0r1 | 0.091592869 | -0.000638906 | 0.092231776 |
| gapped_r1r0 | 0.081308107 | -0.024711015 | 0.106019122 |
| balanced_alt_a | 0.069031338 | -0.009195177 | 0.078226515 |
| balanced_alt_b | 0.096140854 | -0.003865046 | 0.100005900 |

Composition coordinates:
- adjacent Omega composition R0->R1 minus R1->R0: `-0.015805518`
- dead Omega composition R0->R1 minus R1->R0: `-0.005567211`
- source-off Omega composition R0->R1 minus R1->R0: `-0.002460293`
- gapped Omega composition R0->R1 minus R1->R0: `-0.013787347`
- balanced-alt Omega phase A minus phase B: `-0.021779385`
- source-off abs/alive abs: `0.156`
- dead abs/alive abs: `0.352`
- gapped abs/adjacent abs: `0.872`
- balanced-alt abs/adjacent abs: `1.378`

Interpretation:
- composition replay-exclusion candidate: `False`
- source-off collapses below 0.25 x alive composition: `True`
- dead preserves at least 0.25 x alive composition: `True`
- gapped replay collapses below 0.25 x adjacent: `False`
- balanced-alt replay collapses below 0.25 x adjacent: `False`
- one-factor strata same sign: `True`

This is exploratory evidence only. It emits no positive scientific claim and does not promote `SMALL_WALL_CROSSED`.
