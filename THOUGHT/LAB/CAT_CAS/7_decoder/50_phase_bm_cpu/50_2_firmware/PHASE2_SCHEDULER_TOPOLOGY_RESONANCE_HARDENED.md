# PHASE2_SCHEDULER_TOPOLOGY_RESONANCE_HARDENED

## Verdict

`SCHEDULER_TOPOLOGY_RESONANCE_NOT_REPRODUCED`

The core-pair scheduler/topology probe produced sparse timing candidates, but
they did not reproduce across fresh seed windows or core pairs.

## Artifacts

- Harness: `50_2_phase_locked_network/src/scheduler_topology_resonance.c`
- Analyzer: `50_2_phase_locked_network/src/analyze_scheduler_topology_resonance.py`
- Initial report: `50_2_firmware/PHASE2_SCHEDULER_TOPOLOGY_RESONANCE.md`
- Row CSVs: `50_2_firmware/PHASE2_SCHEDULER_TOPOLOGY_RESONANCE_*.csv`
- Analysis CSVs: `50_2_firmware/PHASE2_SCHEDULER_TOPOLOGY_RESONANCE_*_analysis.csv`

## Validation Summary

| Run | Result | Top feature | bAcc | Shuffle p95 | Margin |
|---|---|---|---:|---:|---:|
| core2_3 seed20000 | candidate | `sum_ns_threshold` | 0.633333 | 0.583333 | 0.050000 |
| core2_3 seed30000 | not confirmed | `elapsed_a_ns_threshold` | 0.569196 | 0.578125 | -0.008929 |
| core2_3 seed40000 | not confirmed | `elapsed_b_ns_threshold` | 0.595106 | 0.584175 | 0.010931 |
| core4_5 seed20000 | candidate | `sum_ns_threshold` | 0.666667 | 0.566667 | 0.100000 |
| core4_5 seed30000 | not confirmed | `delta_abs_ns_threshold` | 0.602679 | 0.575083 | 0.027595 |
| core4_5 seed40000 | not confirmed | `mode_majority` | 0.555617 | 0.611235 | -0.055617 |
| core4_5 seed50000 | not confirmed | `elapsed_b_ns_threshold` | 0.555061 | 0.600667 | -0.045606 |
| core4_5 seed60000 | not confirmed | `carrier_low_majority` | 0.500000 | 0.538721 | -0.038721 |

Candidate rate: `2/8`.

All accepted row files had zero restoration failures.

## Interpretation

The scheduler/topology phase-offset mechanism behaves like sparse timing
variance. It can produce candidate windows, but those windows do not hold across
fresh seeds or across the `2,3` and `4,5` core pairs.

This closes the current scheduler/topology threshold route as CPU-sings
evidence.

## Route Impact

Route 5 moves to:

```text
SCHEDULER_TOPOLOGY_RESONANCE_NOT_REPRODUCED
```

This is not:

- `CPU_SINGS`
- `BYTE_READY_HUMAN_REVIEW`
- `SOFTWARE_FIRMWARE_TRUE_WALL`

## Next Exact Action

`FIRMWARE_P4_SEPARATE_SOURCE_SEARCH`

The runtime software classifiers have now rejected:

- P4 asymmetry oracle
- state-window oracle
- state-label elapsed threshold
- state-label modal features
- scheduler/topology phase-offset threshold

The next non-repeating route should return to firmware-side search for a
separate P4-affecting source, without repeating the already decoded runtime-MSR
helper chain.

## Boundary

- No platform setting changes.
- No P0-P3 modification.
- No candidate image construction.
- No external instrumentation.
