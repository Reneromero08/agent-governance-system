# Offset16 Origin2048 AB Lifecycle Probe

Run ID: `family10h_relation_offset16_origin2048_ab_lifecycle_probe_v1_0`
Archive SHA-256: `55de9f050f49534c1f7b6fcfe16ae3e3e7a20dc1c2c00b406e6c8e77874841db`
Analysis SHA-256: `21ff4d08015e8d2f0fa0273871abf065c48023959bc55726c2ec5bf69dcaed04`

Receiver projection: relation-matrix contrast of per-row `mean(B_first_touch_cycles - A_first_touch_cycles)` at signed offset 16.

| Variant | Offset | Lifecycle | C offset signed | abs/abs(matched alive) | max stratum ratio |
|---|---:|---|---:|---:|---:|
| alive_offset16_signed | 16 | alive | -2.746972656 | 1.000 | 1.000 |
| source_off_offset16_signed | 16 | source_off | -0.495768229 | 0.180 | 1.562 |
| dead_offset16_signed | 16 | dead | 1.727213542 | 0.629 | 1.561 |
| reset_double_flush_offset16_signed | 16 | reset_double_flush | -11.853678385 | 4.315 | 10.100 |

Discovery interpretation:
- alive nonzero: `True`
- alive one-factor strata same sign: `True`
- source-off collapses below 25pct alive: `True`
- source-off one-factor strata collapse: `False`
- dead preserves above 25pct alive same-sign: `False`
- dead one-factor strata preserve same-sign: `False`
- reset double flush collapses below 25pct alive: `False`
- reset double flush one-factor strata collapse: `False`
- lifecycle candidate: `False`

This is exploratory evidence only. It emits no prospective scientific claim and does not promote `SMALL_WALL_CROSSED`.
