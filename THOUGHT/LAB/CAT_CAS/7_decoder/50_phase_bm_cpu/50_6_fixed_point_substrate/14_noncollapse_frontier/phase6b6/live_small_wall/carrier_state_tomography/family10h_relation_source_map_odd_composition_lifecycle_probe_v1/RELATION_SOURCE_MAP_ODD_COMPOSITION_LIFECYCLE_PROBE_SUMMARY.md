# Relation Source Map Odd Composition Lifecycle Probe

Run ID: `family10h_relation_source_map_odd_composition_lifecycle_probe_v1_0`
Archive SHA-256: `334f4857a54cd522d80087d31372e61e7003b9d45e9e8f38ad4f49dcd8ff9d4a`
Analysis SHA-256: `f9304a998e9caeb622e949c97e72231c8f65f2923524c71925dd7560b5eb2b6a`

| Pair | Primary R | Sham R | D primary-minus-sham |
|---|---:|---:|---:|
| native_alive_r0r1 | 0.079960664 | -0.000489023 | 0.080449687 |
| native_alive_r1r0 | 0.081863733 | 0.003541661 | 0.078322072 |
| native_dead_r0r1 | 0.091384563 | -0.000114910 | 0.091499473 |
| native_dead_r1r0 | 0.083350618 | 0.011263233 | 0.072087386 |
| native_source_off_r0r1 | 0.110955140 | -0.005316207 | 0.116271346 |
| native_source_off_r1r0 | 0.115071038 | 0.002051638 | 0.113019400 |
| swapped_alive_r0r1 | 0.098599127 | -0.012901246 | 0.111500373 |
| swapped_alive_r1r0 | 0.082561043 | -0.015720437 | 0.098281481 |
| swapped_dead_r0r1 | 0.070964784 | -0.015039190 | 0.086003974 |
| swapped_dead_r1r0 | 0.077861471 | 0.004327042 | 0.073534429 |
| swapped_source_off_r0r1 | 0.123059200 | 0.002379365 | 0.120679836 |
| swapped_source_off_r1r0 | 0.110204341 | 0.001274700 | 0.108929641 |

Source-map-odd composition coordinates:
- Omega native alive: `0.002127615`
- Omega swapped alive: `0.013218892`
- Gamma alive: `-0.005545639`
- Gamma source-off: `-0.004249124`
- Gamma dead: `0.003471271`
- source-off |Gamma| / alive |Gamma|: `0.766`
- dead |Gamma| / alive |Gamma|: `0.626`
- alive even-mode |E| / |Gamma|: `1.384`

Interpretation:
- source-map-odd composition lifecycle candidate: `False`
- native/swapped reversal aggregate: `False`
- source-off Gamma collapses below 0.25 x alive Gamma: `False`
- dead Gamma preserves same sign and >= 0.25 x alive Gamma: `False`
- alive Gamma same sign in one-factor strata: `False`
- source-off Gamma collapses in one-factor strata: `False`
- dead Gamma preserves in one-factor strata: `False`

This is exploratory evidence only. It emits no positive scientific claim and does not promote `SMALL_WALL_CROSSED`.
