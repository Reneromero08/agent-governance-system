# PHASE2_RUNTIME_P4_ASYMMETRY_ORACLE_REPORT

## Verdict

`RUNTIME_P4_ASYMMETRY_ORACLE_CLOSED`

The read-only P4 asymmetry oracle ran on the Phenom II target and found
no actionable P4-definition asymmetry after the target restart. Earlier runtime
observation saw cores 3/4 with a different `MSRC001_0068` value; this run found
all six cores normalized to the same P4 definition.

This closes the specific angle:

```text
existing per-core P4 definition asymmetry -> timing/state oracle
```

It does not close software/firmware Phase 2 overall.

## Command

```powershell
modprobe msr
Get-Content -Raw 50_1_subthreshold_msr\src\msr_p4_asymmetry_oracle.py |
  <target-shell> `
  "python3 - --cores 0-5 --samples 48 --workload-iters 384 --modes baseline,self_load,neighbors_load,other_group_load,all_load" |
  Out-File -Encoding ascii cpu_sing_3\PHASE2_RUNTIME_P4_ASYMMETRY_ORACLE.json
```

Direct all-core P4 readback:

```text
core0 8000013540003440
core1 8000013540003440
core2 8000013540003440
core3 8000013540003440
core4 8000013540003440
core5 8000013540003440
```

## Safety

- No MSR setting changes.
- No setting-write tool.
- No uncontrolled setting changes.
- No platform image action.
- No P0-P3 modification.
- No external instrumentation.
- `modprobe msr` was used only to expose `/dev/cpu/*/msr` for read access.

## Result Summary

| Metric | Result |
|---|---:|
| cases | 30 |
| modes | baseline, self_load, neighbors_load, other_group_load, all_load |
| samples per case | 48 |
| workload iterations | 384 |
| P4 groups | `p4_did=1/vid=26` only |
| P4 raw values | `0x8000013540003440` only |
| COFVID VID values | `0x12` only |
| oracle candidates | 0 |

Mode timing means by the single P4 group:

| Mode | P4 group mean ns | Candidate |
|---|---:|---|
| baseline | 162739.819 | false |
| self_load | 345343.170 | false |
| neighbors_load | 141508.062 | false |
| other_group_load | 235911.042 | false |
| all_load | 254030.267 | false |

Because there was only one P4 group, cyclic null comparison had no two-group
label to test. The oracle therefore closes for this runtime state.

## Interpretation

The prior P4-definition asymmetry was real in the earlier observer report, but
it was not stable across restart. Current runtime state is:

```text
P4 definition MSR: uniform, VID 0x1A
COFVID status VID: uniform, VID 0x12
```

That matters for both firmware and software:

- Firmware still reconstructs the constructor-relevant P4 field from runtime
  `MSRC001_0068`.
- Runtime `MSRC001_0068` is readable and writable in principle, but this
  read-only run shows no persistent per-core P4 definition split to use.
- COFVID status remains clamped at VID `0x12` under this load matrix.
- Any future software carrier must use runtime transition/load/selector
  behavior, not the vanished old P4-asymmetry label.

## Route Impact

Route 5 advances to:

```text
RUNTIME_P4_ASYMMETRY_ORACLE_CLOSED
```

This is not:

- `CPU_SINGS`
- `BYTE_READY_HUMAN_REVIEW`
- `SOFTWARE_FIRMWARE_TRUE_WALL`

## Next Exact Action

`P4_EFFECTIVE_STATE_CONTROL_PROOF`

The next software/firmware route should verify whether any safe, P4-scoped
runtime control actually changes effective silicon state as seen through
COFVID/PSTATE/timing carriers. Do not call decoded P4 VID a real effective-state effect
unless COFVID/status or a stronger internal carrier moves with it.
