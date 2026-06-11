# PHASE2_STATE_LABEL_TIMING_EDGE_HARDENED_REPORT

## Verdict

`STATE_LABEL_TIMING_EDGE_COLLAPSED_UNDER_HARD_NULL`

The state-label timing edge survived a few small windows, but it did not survive
the harder reproducibility gates. The useful result is negative but precise:
restoration remains stable, internal state labels remain observable, and the
timing edge is not reproducible enough to count as CPU-sings evidence.

## Artifact Chain

- Selector map: `cpu_sing_3/PHASE2_EFFECTIVE_STATE_SELECTOR_MAP_REPORT.md`
- First coupling probe: `cpu_sing_3/PHASE2_STATE_LABEL_PHASE_COUPLING_REPORT.md`
- Compact sweep: `cpu_sing_3/PHASE2_STATE_LABEL_TIMING_EDGE_STABILITY_SWEEP.md`
- Dense 4096-round sweep: `cpu_sing_3/PHASE2_STATE_LABEL_TIMING_EDGE_NARROWING_SWEEP.md`
- Shuffled-answer hard-null sweep: `cpu_sing_3/PHASE2_STATE_LABEL_HARDNULL_SWEEP.md`
- Higher-row focus sweep: `cpu_sing_3/PHASE2_STATE_LABEL_HARDNULL_FOCUS_SWEEP.md`

## Result Summary

| Stage | Runs | Candidate runs | Key result |
|---|---:|---:|---|
| Compact stability sweep | 6 | 2 | Timing edge remained live but unstable. |
| Dense 4096-round narrowing sweep | 16 | 2 | Mean margin over mode/core controls was only `0.025910`; only seeds `5000` and `6000` passed. |
| Shuffled-answer hard-null sweep | 5 | 1 | Only seed `6000` passed both mode/core and shuffled-answer gates. Seed `5000` missed shuffled-null margin. |
| Higher-row focus sweep | 3 | 0 | Seed `6000` did not reproduce with 16 trials per case; seed `6250` beat shuffled null but failed mode/core margin. |

All stages had zero restoration failures.

## Interpretation

The edge is best classified as unstable timing variance:

```text
state labels observable -> timing edge appears in sparse windows -> hard nulls
and higher-row reruns collapse the candidate
```

This does not erase the state-label selector discovery. It does close the
current timing-threshold classifier as a CPU-sings route.

## Route Impact

Route 5 moves from:

```text
STATE_LABEL_TIMING_EDGE_NOT_STABLE_YET
```

to:

```text
STATE_LABEL_TIMING_EDGE_COLLAPSED_UNDER_HARD_NULL
```

Follow-up modal feature validation:

```text
STATE_LABEL_MODAL_FEATURE_NOT_CONFIRMED
```

`cpu_sing_3/PHASE2_STATE_LABEL_MODAL_VALIDATION.md` tested eight fresh seed
windows. It found candidate feature rows, including `elapsed_quantile` and
`elapsed_state_quantile`, but no modal feature family survived the shuffled
answer criterion across three distinct fresh seed starts. The state-label
surface remains observable, but this modal classifier is not a CPU-sings route.

This is not:

- `CPU_SINGS`
- `BYTE_READY_HUMAN_REVIEW`
- `SOFTWARE_FIRMWARE_TRUE_WALL`

## Next Exact Action

`NEXT_SOFTWARE_ROUTE_SEARCH`

Do not keep rerunning elapsed-threshold or modal classifiers over the same
state-label rows. The next software route should move to a different mechanism,
for example scheduler/topology resonance or a separate P4-affecting firmware
source.

- preserve zero restoration failures for catalytic rows
- require hard nulls
- require reproduction across seed windows
- do not reuse a classifier already closed here as the main success claim

## Boundary

- No platform setting changes.
- No P0-P3 modification.
- No candidate image construction.
- No external instrumentation.
