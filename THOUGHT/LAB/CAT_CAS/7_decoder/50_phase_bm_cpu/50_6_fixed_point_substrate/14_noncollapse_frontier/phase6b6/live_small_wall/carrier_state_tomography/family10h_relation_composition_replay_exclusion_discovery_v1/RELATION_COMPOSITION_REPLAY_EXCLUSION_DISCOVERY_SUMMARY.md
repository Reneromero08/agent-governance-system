# Relation Composition Replay-Exclusion Discovery

Run ID: `family10h_relation_composition_replay_exclusion_discovery_v1_0`
Archive SHA-256: `12c4c59e32acb11a2bea3555680bb174e599604dccd5b3964bc2f88ff257d7f5`
Analysis SHA-256: `8a47c60aed8d5b8627307acee02b589f0fe6c7b670b600896892baa2bd8db281`

| Pair | Primary R | Sham R | D primary-minus-sham |
|---|---:|---:|---:|
| adjacent_r0r1 | 0.051810439 | 0.011592843 | 0.040217596 |
| adjacent_r1r0 | 0.077950955 | -0.009123467 | 0.087074422 |
| dead_adjacent_r0r1 | 0.027557958 | -0.000731067 | 0.028289025 |
| dead_adjacent_r1r0 | 0.041791289 | -0.004403003 | 0.046194292 |
| source_off_adjacent_r0r1 | 0.119923066 | -0.005320744 | 0.125243810 |
| source_off_adjacent_r1r0 | 0.117156635 | -0.001812173 | 0.118968808 |
| gapped_r0r1 | 0.073439742 | 0.001493177 | 0.071946565 |
| gapped_r1r0 | 0.062695978 | -0.010617417 | 0.073313396 |
| balanced_alt_a | 0.069980493 | -0.003003763 | 0.072984256 |
| balanced_alt_b | 0.056620544 | 0.001374999 | 0.055245544 |

Composition coordinates:
- adjacent Omega composition R0->R1 minus R1->R0: `-0.046856826`
- dead Omega composition R0->R1 minus R1->R0: `-0.017905266`
- source-off Omega composition R0->R1 minus R1->R0: `0.006275002`
- gapped Omega composition R0->R1 minus R1->R0: `-0.001366830`
- balanced-alt Omega phase A minus phase B: `0.017738712`
- source-off abs/alive abs: `0.134`
- dead abs/alive abs: `0.382`
- gapped abs/adjacent abs: `0.029`
- balanced-alt abs/adjacent abs: `0.379`

Interpretation:
- composition replay-exclusion candidate: `False`
- source-off collapses below 0.25 x alive composition: `True`
- dead preserves at least 0.25 x alive composition: `True`
- gapped replay collapses below 0.25 x adjacent: `True`
- balanced-alt replay collapses below 0.25 x adjacent: `False`
- one-factor strata same sign: `True`

This is exploratory evidence only. It emits no positive scientific claim and does not promote `SMALL_WALL_CROSSED`.
