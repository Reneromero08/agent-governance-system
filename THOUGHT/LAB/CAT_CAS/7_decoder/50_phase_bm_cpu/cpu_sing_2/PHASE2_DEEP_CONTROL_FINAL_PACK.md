# PHASE2_DEEP_CONTROL_FINAL_PACK

## Verdict

PHASE2_DEEP_CONTROL_SURFACES_CLASSIFIED

No accepted Kuramoto lock, Ising descent, GOE structure, or voltage reopening was produced in this pass. The successful result is classification of the deeper non-destructive control surfaces and preparation of the next measurement path.

## Route Table

| Route | Artifact | Verdict |
|---|---|---|
| 1 Firmware targeting | `PHASE2_DEEP_1_FIRMWARE.md` | `FIRMWARE_ROUTE_CLASSIFIED_DESIGN_ONLY` |
| 2 Clamp map | `PHASE2_DEEP_2_CLAMP_MAP.md` | `CLAMP_MAP_READONLY_CLASSIFIED` |
| 3 External measurement | `PHASE2_DEEP_3_EXTERNAL_MEASURE.md` | `EXTERNAL_OBSERVABILITY_PLAN_READY` |
| 4 Markers | `PHASE2_DEEP_4_MARKERS.md` | `PHASE_MARKER_HARNESS_READY` |
| 5 Retest | `PHASE2_DEEP_5_RETEST.md` | `RETEST_PROTOCOL_READY_WAITING_FOR_BETTER_OBSERVABILITY` |

## Best Control Surface

Best immediate control surface: external observability with marker correlation.

Reason:

- Prior software-only metrics did not separate from nulls.
- Runtime VID is clamped.
- Firmware is not byte-ready for P4-only safety.
- The marker harness now creates repeatable workload state transitions for waveform alignment.

## Firmware Status

Design-only. Known target:

```text
AmdProcessorInitPeim GUID DE3E049C-A218-4891-8658-5FC0FA84C788
file offset 0x00340048
PE32 offset 0x00340088
normalizer window 0x00366DF0-0x00366E8F
loop limit compare at 0x00366E82
global branch byte at 0x00366E3E is not P4-safe
```

No flash action was prepared.

## Measurement Status

Marker harness sanity pass succeeded:

```text
segment,tsc,state,edge,c3,c4,c5
0,71291455504002,0x111,1,1311768467463986931,1311768467463528181,1311768467463593719
1,71291487920205,0x121,2,14901733374053020589,3774310008201819642,12100212319698572849
```

Safety after marker run:

```text
P4C3 8000013540003440
P4C4 8000013540003440
k10temp +48.5 C
```

## Exact Next Human Action

Connect a non-invasive oscilloscope probe to a CPU VRM output capacitor with a short ground spring. Capture at 100 MS/s if available while running:

```sh
./phase2_marker_harness 256 50000 > phase2_marker_log.csv
```

Save the raw waveform and marker CSV, then run marker-aligned analysis. Accept a signal only if it changes by marker state and survives shuffled-label nulls.

## Blocked Routes

- Runtime VID path: clamped at CpuVid `0x1A`.
- Global firmware branch edit: not P4-safe.
- Direct NB PCI writes: field meaning is not proven safe for active-core control.
- Prior software-only Kuramoto sweeps: no real/null separation.

## Missing Artifacts

- Raw external waveform capture.
- Probe point photo.
- Board VRM controller marking.
- P4-only firmware byte plan with P0-P3 safety proof.
- UEFITool validation output for any future edited image.

## Do Not Do

- Do not flash BIOS.
- Do not do board modifications.
- Do not write unknown PCI fields.
- Do not run unattended voltage sweeps.
- Do not rerun prior software sweeps without better observability.

