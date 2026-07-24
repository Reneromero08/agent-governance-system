# Offset-2 Signed Projection Source-Off Screen Discovery

Run ID: `family10h_relation_offset2_signed_projection_source_off_screen_discovery_v1_0`
Archive SHA-256: `a95ba0ea8f4f6e0b25c52b4baa769e6c0e60f41f1b3b2c987cc7a4cf1275687e`
Analysis SHA-256: `c1ed9f13aea38aa2d1ab00fbb5097a20fda4bf125b75a7df392e15f441a3779d`

Receiver projection: relation-matrix contrast of per-row `mean(B_first_touch_cycles - A_first_touch_cycles)` at signed offset 2.

| Variant | Lifecycle | C offset2 signed | abs/abs(alive) | max stratum ratio |
|---|---|---:|---:|---:|
| alive_offset2_signed | alive | -2.759185791 | 1.000 | 1.000 |
| source_off_offset2_signed | source_off | -0.553924561 | 0.201 | 34.642 |
| dead_offset2_signed | dead | -0.484313965 | 0.176 | 13.197 |
| reset_double_flush_offset2_signed | reset_double_flush | 6.248352051 | 2.265 | 47.050 |

Discovery interpretation:
- source-off collapses below 0.25 x alive: `True`
- source-off all one-factor strata below 0.25 x alive: `False`
- dead preserves at least 0.25 x alive: `False`
- reset double flush collapses below 0.25 x alive: `False`
- offset2 signed projection candidate: `False`

This is exploratory evidence only. It emits no prospective scientific claim and does not promote `SMALL_WALL_CROSSED`.
