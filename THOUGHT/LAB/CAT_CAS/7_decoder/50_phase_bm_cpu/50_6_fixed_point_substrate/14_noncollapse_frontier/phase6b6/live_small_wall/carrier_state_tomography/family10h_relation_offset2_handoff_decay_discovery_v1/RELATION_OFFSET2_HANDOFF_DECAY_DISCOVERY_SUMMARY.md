# Offset-2 Handoff Decay Discovery

Run ID: `family10h_relation_offset2_handoff_decay_discovery_v1_0`
Archive SHA-256: `362f33fa01341eb1c8beadc98815a22d5d56063371f167f0dfd547f86ed7aad2`
Analysis SHA-256: `12908e58b7668be85eeb2d78c759667b2c0d30f9f954d22d642d1f5ffd8003f7`

Receiver projection: relation-matrix contrast of per-row `mean(B_first_touch_cycles - A_first_touch_cycles)` at signed offset 2.

| Variant | Source lifetime | Delay ns | C offset2 signed | abs/abs(alive) | one-factor same sign |
|---|---|---:|---:|---:|---|
| alive_offset2_signed | alive_during_query | 0 | -1.536590576 | 1.000 | `False` |
| source_off_no_prep_offset2_signed | source_off_no_preparation | 0 | -7.144897461 | 4.650 | `False` |
| dead_after_exit_0ns_offset2_signed | prepared_source_exited_before_query_delay_0ns | 0 | -0.847778320 | 0.552 | `False` |
| dead_after_exit_10us_offset2_signed | prepared_source_exited_before_query_delay_10000ns | 10000 | 0.841278076 | 0.547 | `False` |
| dead_after_exit_100us_offset2_signed | prepared_source_exited_before_query_delay_100000ns | 100000 | -2.497955322 | 1.626 | `False` |
| dead_after_exit_1ms_offset2_signed | prepared_source_exited_before_query_delay_1000000ns | 1000000 | -5.710601807 | 3.716 | `False` |
| dead_after_exit_10ms_offset2_signed | prepared_source_exited_before_query_delay_10000000ns | 10000000 | -1.135528564 | 0.739 | `False` |

Discovery interpretation:
- true source-off no-prep collapses below 0.25 x alive: `False`
- true source-off no-prep all one-factor strata below 0.25 x alive: `False`
- post-source-exit handoff candidate variants: `[]`

This is exploratory evidence only. It emits no prospective scientific claim and does not promote `SMALL_WALL_CROSSED`.
