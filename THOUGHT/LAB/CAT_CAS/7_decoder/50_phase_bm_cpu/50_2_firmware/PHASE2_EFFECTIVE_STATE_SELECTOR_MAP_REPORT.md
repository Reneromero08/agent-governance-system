# PHASE2_EFFECTIVE_STATE_SELECTOR_MAP_REPORT

## Verdict

`READ_ONLY_EFFECTIVE_STATE_SELECTOR_FOUND__VID_STILL_FIXED`

The read-only selector map produced a useful result: ordinary load selectors
move internal frequency-state labels, but the VID label remains fixed at `0x12`.

This advances the runtime route without supporting a direct requested-VID
physical-cause claim. The usable carrier is now internal state selection plus
timing/CV behavior, not decoded P4 VID.

## Artifact

- JSON: `50_2_firmware/PHASE2_EFFECTIVE_STATE_SELECTOR_MAP.json`
- Runner: `50_1_subthreshold_msr/src/msr_effective_state_selector_map.py`

## Run Summary

| Item | Result |
|---|---:|
| setting changes | false |
| cases | 30 |
| cores | 0-5 |
| modes | baseline, self_load, neighbor_load, other_load, all_load |
| samples per case | 72 |
| workload iterations | 512 |
| COFVID label count | 4 |
| PSTATE label count | 4 |
| effective state moved | true |

Observed COFVID labels:

```text
fid=00/did=0/vid=12/raw=0x0180000140032400
fid=00/did=1/vid=12/raw=0x0180000140042440
fid=08/did=0/vid=12/raw=0x0180000140022408
fid=10/did=0/vid=12/raw=0x0180000140012410
```

Observed PSTATE labels:

```text
fid=00/did=0/vid=00/raw=0x0000000000000000
fid=01/did=0/vid=00/raw=0x0000000000000001
fid=02/did=0/vid=00/raw=0x0000000000000002
fid=03/did=0/vid=00/raw=0x0000000000000003
```

## Timing Carrier

| Mode | Timing mean ns | Timing CV |
|---|---:|---:|
| baseline | 195711.551 | 0.069653154 |
| neighbor_load | 183639.836 | 0.049428477 |
| other_load | 234508.921 | 0.477827230 |
| all_load | 356899.880 | 0.073923713 |
| self_load | 394680.903 | 0.228885465 |

## Interpretation

The VID field did not move: every COFVID label retained `vid=12`. That keeps the
runtime clamp result intact and prevents claiming that decoded P4 VID labels
became effective silicon state.

However, FID/DID/PSTATE labels did move across the read-only load matrix. This
means there is still a software-visible selector surface:

```text
ordinary load selector -> internal state label distribution -> timing/CV carrier
```

That is narrower than the original P4 route, but it is live and measurable from
software.

## Route Impact

Route 5 advances from:

```text
RUNTIME_P4_ASYMMETRY_ORACLE_CLOSED
```

to:

```text
READ_ONLY_EFFECTIVE_STATE_SELECTOR_FOUND__VID_STILL_FIXED
```

This is not:

- `CPU_SINGS`
- `BYTE_READY_HUMAN_REVIEW`
- `SOFTWARE_FIRMWARE_TRUE_WALL`

## Next Exact Action

`STATE_LABEL_PHASE_COUPLING_PROBE`

Use the observed FID/DID/PSTATE state labels as bins and test whether a
core-pinned reversible/catalytic workload shows answer-predictive or
phase-coupled structure after null comparison.

Acceptance criteria:

- record per-row COFVID and PSTATE labels
- keep VID fixed as a control, not as a cause
- compare state-label bins against cyclic core-label nulls
- require answer/coupling separation above nulls
- preserve restoration/hash status for catalytic rows

## Boundary

- No platform setting changes.
- No P0-P3 modification.
- No candidate image construction.
- No external instrumentation.
