# Offset Signed Projection Source-Off Sweep Discovery

Run ID: `family10h_relation_offset_signed_source_off_sweep_discovery_v1_0`
Archive SHA-256: `2c07e333b6f73a28de7057804efb7e68145c233e14fd8252fc67ebd930dc1bf7`
Analysis SHA-256: `8ef788e71031e7e44550b6df3bf538d66a86db8323e72c7e533381052bb80a76`

Receiver projection: relation-matrix contrast of per-row `mean(B_first_touch_cycles - A_first_touch_cycles)` at signed offset sweep.

| Variant | Offset | Lifecycle | C offset signed | abs/abs(matched alive) | max stratum ratio |
|---|---:|---|---:|---:|---:|
| alive_offset2_signed | 2 | alive | 2.601928711 | 1.000 | 1.000 |
| source_off_offset2_signed | 2 | source_off | 4.311462402 | 1.657 | 16.117 |
| alive_offset4_signed | 4 | alive | 0.213836670 | 1.000 | 1.000 |
| source_off_offset4_signed | 4 | source_off | -1.113677979 | 5.208 | 12.828 |
| alive_offset8_signed | 8 | alive | 1.619110107 | 1.000 | 1.000 |
| source_off_offset8_signed | 8 | source_off | 1.337280273 | 0.826 | 1.573 |
| alive_offset16_signed | 16 | alive | -2.154510498 | 1.000 | 1.000 |
| source_off_offset16_signed | 16 | source_off | 0.425933838 | 0.198 | 8.953 |
| alive_offset1024_signed | 1024 | alive | 1.382171631 | 1.000 | 1.000 |
| source_off_offset1024_signed | 1024 | source_off | -0.016448975 | 0.012 | 13.964 |

Discovery interpretation:
- source-off screen hit offsets: `[]`
- best source-off ratio offset: `1024`
- best source-off abs/alive abs: `0.012`
- any source-off screen hit: `False`
- followup needed for dead/reset persistence: `False`

This is exploratory evidence only. It emits no prospective scientific claim and does not promote `SMALL_WALL_CROSSED`.
