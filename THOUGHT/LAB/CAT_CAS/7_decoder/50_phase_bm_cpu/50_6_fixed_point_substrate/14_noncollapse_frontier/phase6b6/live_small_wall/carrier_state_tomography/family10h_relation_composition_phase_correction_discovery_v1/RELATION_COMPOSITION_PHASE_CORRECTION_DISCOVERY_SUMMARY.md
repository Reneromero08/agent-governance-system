# Relation Composition Phase-Correction Discovery

Run ID: `family10h_relation_composition_phase_correction_discovery_v1_0`
Archive SHA-256: `39f5a85d55cf154f6c7199d9c1452bd9700a74c5a3cf7ce311d6d00f47625ecc`
Analysis SHA-256: `27cf7a31788487262f8866f3ccebe64e8461c1dcc29feed12b84435306a74c28`

| Pair | Primary R | Sham R | D primary-minus-sham |
|---|---:|---:|---:|
| adjacent_r0r1 | 0.063454351 | -0.014217581 | 0.077671932 |
| adjacent_r1r0 | 0.084635001 | 0.000389157 | 0.084245845 |
| balanced_alt_a | 0.085996227 | 0.012637440 | 0.073358786 |
| balanced_alt_b | 0.095487870 | 0.011339831 | 0.084148039 |
| dead_adjacent_r0r1 | 0.067387183 | 0.001608283 | 0.065778900 |
| dead_adjacent_r1r0 | 0.065476195 | 0.008940538 | 0.056535656 |
| dead_balanced_alt_a | 0.067152007 | -0.017589059 | 0.084741066 |
| dead_balanced_alt_b | 0.087465062 | -0.005161586 | 0.092626648 |
| source_off_adjacent_r0r1 | 0.097401920 | 0.000711161 | 0.096690759 |
| source_off_adjacent_r1r0 | 0.097678004 | 0.008091984 | 0.089586020 |
| source_off_balanced_alt_a | 0.094202209 | 0.004618327 | 0.089583881 |
| source_off_balanced_alt_b | 0.098981529 | -0.003695546 | 0.102677074 |
| gapped_r0r1 | 0.052790272 | -0.015120922 | 0.067911193 |
| gapped_r1r0 | 0.063492465 | -0.002488291 | 0.065980756 |
| gapped_balanced_alt_a | 0.079445888 | -0.004375939 | 0.083821826 |
| gapped_balanced_alt_b | 0.096838946 | 0.004837984 | 0.092000962 |

Composition coordinates:
- adjacent Omega composition R0->R1 minus R1->R0: `-0.006573912`
- dead Omega composition R0->R1 minus R1->R0: `0.009243244`
- source-off Omega composition R0->R1 minus R1->R0: `0.007104739`
- gapped Omega composition R0->R1 minus R1->R0: `0.001930438`
- balanced-alt Omega phase A minus phase B: `-0.010789253`
- dead balanced-alt Omega phase A minus phase B: `-0.007885582`
- source-off balanced-alt Omega phase A minus phase B: `-0.013093193`
- gapped balanced-alt Omega phase A minus phase B: `-0.008179135`
- alive Psi adjacent-minus-balanced: `0.004215341`
- dead Psi adjacent-minus-balanced: `0.017128826`
- source-off Psi adjacent-minus-balanced: `0.020197932`
- gapped Psi adjacent-minus-balanced: `0.010109573`
- source-off Psi abs/alive Psi abs: `4.792`
- dead Psi abs/alive Psi abs: `4.063`
- gapped Psi abs/alive Psi abs: `2.398`
- diagnostic balanced-alt abs/adjacent Omega abs: `1.641`

Interpretation:
- composition phase-correction candidate: `False`
- source-off collapses below 0.25 x alive Psi: `False`
- dead preserves at least 0.25 x alive Psi: `True`
- gapped collapses below 0.25 x alive Psi: `False`
- corrected one-factor strata same sign: `False`

This is exploratory evidence only. It emits no positive scientific claim and does not promote `SMALL_WALL_CROSSED`.
