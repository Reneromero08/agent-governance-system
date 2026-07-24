# Relation Composition Loop Discovery

Run ID: `family10h_relation_composition_loop_discovery_v1_0`
Archive SHA-256: `e842faf701cb3c8837a1777ace991bda145155f65436e77d8df83968988ca494`
Analysis SHA-256: `0f17b1d691427cd43684d2ad463591f81e7652c329d625cad3862e1f8b7d2df5`

| Pair | Primary R | Sham R | D primary-minus-sham |
|---|---:|---:|---:|
| r0r1 | 0.070606324 | 0.005392471 | 0.065213853 |
| r1r0 | 0.053006165 | 0.012873997 | 0.040132169 |
| dead_r0r1 | 0.039250375 | -0.004658611 | 0.043908986 |
| dead_r1r0 | 0.031162072 | -0.005274407 | 0.036436479 |
| source_off_r0r1 | 0.116113413 | -0.002128955 | 0.118242368 |
| source_off_r1r0 | 0.106103700 | -0.010551988 | 0.116655688 |
| neutral | 0.073762608 | -0.008213680 | 0.081976289 |
| random | 0.068657126 | -0.012820977 | 0.081478102 |

Composition coordinates:
- alive Omega composition R0->R1 minus R1->R0: `0.025081684`
- dead Omega composition R0->R1 minus R1->R0: `0.007472507`
- source-off Omega composition R0->R1 minus R1->R0: `0.001586680`
- source-off abs/alive abs: `0.063`
- dead abs/alive abs: `0.298`

Interpretation:
- composition-loop candidate: `True`
- source-off collapses below 0.25 x alive composition: `True`
- dead preserves at least 0.25 x alive composition: `True`
- one-factor strata same sign: `True`

This is exploratory evidence only. It emits no positive scientific claim and does not promote `SMALL_WALL_CROSSED`.
