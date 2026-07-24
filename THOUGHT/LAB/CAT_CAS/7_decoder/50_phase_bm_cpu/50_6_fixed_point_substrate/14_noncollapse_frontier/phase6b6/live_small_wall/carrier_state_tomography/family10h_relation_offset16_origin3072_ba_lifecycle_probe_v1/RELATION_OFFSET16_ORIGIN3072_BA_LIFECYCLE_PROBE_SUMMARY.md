# Offset16 Origin3072 BA Lifecycle Probe

Run ID: `family10h_relation_offset16_origin3072_ba_lifecycle_probe_v1_0`
Archive SHA-256: `f4a56d40bef4f59e84d96338b35a1f3a01e87204cb3a89cde838ece2ca262c65`
Analysis SHA-256: `40427fe3c05d36207a21d7bfd1f7d20efa00d4dd28d80fd0366f93809fa25046`

Receiver projection: relation-matrix contrast of per-row `mean(B_first_touch_cycles - A_first_touch_cycles)` at signed offset 16.

| Variant | Offset | Lifecycle | C offset signed | abs/abs(matched alive) | max stratum ratio |
|---|---:|---|---:|---:|---:|
| alive_offset16_signed | 16 | alive | -0.569108073 | 1.000 | 1.000 |
| source_off_offset16_signed | 16 | source_off | -2.512695312 | 4.415 | 40.228 |
| dead_offset16_signed | 16 | dead | 2.015559896 | 3.542 | 8.289 |
| reset_double_flush_offset16_signed | 16 | reset_double_flush | -5.644270833 | 9.918 | 153.400 |

Discovery interpretation:
- alive nonzero: `True`
- alive one-factor strata same sign: `False`
- source-off collapses below 25pct alive: `False`
- source-off one-factor strata collapse: `False`
- dead preserves above 25pct alive same-sign: `False`
- dead one-factor strata preserve same-sign: `False`
- reset double flush collapses below 25pct alive: `False`
- reset double flush one-factor strata collapse: `False`
- lifecycle candidate: `False`

This is exploratory evidence only. It emits no prospective scientific claim and does not promote `SMALL_WALL_CROSSED`.
