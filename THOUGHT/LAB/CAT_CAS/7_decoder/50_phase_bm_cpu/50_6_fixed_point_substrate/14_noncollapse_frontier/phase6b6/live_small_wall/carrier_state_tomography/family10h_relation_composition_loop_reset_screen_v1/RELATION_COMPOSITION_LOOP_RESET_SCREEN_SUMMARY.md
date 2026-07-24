# Relation Composition Loop Reset Screen

Run ID: `family10h_relation_composition_loop_reset_screen_v1_2`
Archive SHA-256: `b85aa8ca86bc549f3ed55ad77cbf860f2768f389b1a32040a760aa0e292f3d26`
Analysis SHA-256: `60c86c8aea2bd1aec7399c43d042ec8e3707a9da825164eb6f714daed8650ff7`

| Pair | Primary R | Sham R | D primary-minus-sham |
|---|---:|---:|---:|
| r0r1 | 0.055205037 | -0.009638540 | 0.064843577 |
| r1r0 | 0.061834969 | 0.016991008 | 0.044843961 |
| dead_r0r1 | 0.014035425 | -0.007524982 | 0.021560407 |
| dead_r1r0 | 0.022418869 | -0.007860341 | 0.030279210 |
| reset_r0r1 | 0.016216908 | 0.024268552 | -0.008051644 |
| reset_r1r0 | 0.003649617 | -0.001638517 | 0.005288134 |
| source_off_r0r1 | 0.116440574 | -0.007267633 | 0.123708207 |
| source_off_r1r0 | 0.108299122 | 0.001042286 | 0.107256836 |
| neutral | 0.063545353 | -0.005155038 | 0.068700392 |
| random | 0.072615329 | -0.028778337 | 0.101393666 |

Composition coordinates:
- alive Omega composition R0->R1 minus R1->R0: `0.019999616`
- dead Omega composition R0->R1 minus R1->R0: `-0.008718803`
- reset double-flush Omega composition R0->R1 minus R1->R0: `-0.013339778`
- source-off Omega composition R0->R1 minus R1->R0: `0.016451371`
- source-off abs/alive abs: `0.823`
- dead abs/alive abs: `0.436`
- reset abs/alive abs: `0.667`

Interpretation:
- composition-loop reset candidate: `False`
- source-off collapses below 0.25 x alive composition: `False`
- dead preserves at least 0.25 x alive composition: `True`
- reset double-flush collapses below 0.25 x alive composition: `False`
- reset double-flush all one-factor strata below 0.25 x matched alive: `False`
- one-factor strata same sign: `False`

This is exploratory evidence only. It emits no positive scientific claim and does not promote `SMALL_WALL_CROSSED`.
