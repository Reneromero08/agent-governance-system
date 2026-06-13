# PHASE2_RUNTIME_TRANSITION_JITTER_REPORT

## Verdict

`RUNTIME_TRANSITION_JITTER_CHARACTERIZED`

The read-only transition/jitter experiment ran over SSH on the Phenom II target. It confirmed that COFVID VID transitions are observable from software under scheduler/load changes, but this run did not show a stable timing-jitter separation that can be used as a phase/Kuramoto success signal.

## Command

```powershell
Get-Content -Raw session_scripts\phase1_msr\msr_transition_jitter_probe.py | ssh -o BatchMode=yes -o ConnectTimeout=5 root@192.168.137.100 "python3 - --cores 0-5 --modes baseline,self_load,neighbor_load,all_load --samples 160 --delay 0.001 --transition-sample-limit 8 --summary-only"
```

## Safety

- No MSR writes.
- No voltage writes.
- No BIOS flash.
- No P0-P3 modification.
- No Tier 3 physical instrumentation.
- Reads only `/dev/cpu/<core>/msr` for `PSTATE_STATUS`, `COFVID_STATUS`, and P4 `MSRC001_0068`.

## Global Summary

| Metric | Result |
|---|---:|
| cases | 24 |
| samples per case | 160 |
| total PSTATE_STATUS transitions | 26 |
| PSTATE transition cases | 17 |
| total COFVID VID transitions | 3 |
| VID transition cases | 3 |

VID transition cases:

| Mode | Core | VID set | Transition count | Transition sample |
|---|---:|---|---:|---|
| `baseline` | 4 | `0x12`, `0x1A` | 1 | `0x12 -> 0x1A`, PSTATE `0 -> 3`, delta `1157144 ns` |
| `self_load` | 1 | `0x12`, `0x14` | 1 | `0x14 -> 0x12`, PSTATE `2 -> 1`, delta `1102561 ns` |
| `self_load` | 2 | `0x12`, `0x14` | 1 | `0x14 -> 0x12`, PSTATE `1 -> 1`, delta `1097558 ns` |

The P4 definition remained stock in all cases:

```text
MSRC001_0068 = 0x8000013540003440
P4 FID=0 DID=1 VID=0x1A
```

## Interpretation

This confirms a software-visible runtime state machine:

- P4 remains `VID=0x1A`.
- COFVID can expose `VID=0x1A`, `VID=0x14`, and `VID=0x12` depending on mode/core.
- PSTATE_STATUS transitions are common enough to observe internally.
- COFVID VID transitions are rare in this short run.
- The measured transition deltas are close to steady sample deltas in this run, so the transition itself is not yet a usable timing oracle.

The useful signal is state observability, not transition jitter. Software can classify PSTATE/COFVID windows from MSR reads, but the current timing channel does not yet show phase-lock or Ising evidence.

## Route Impact

Route 5 advances from `RUNTIME_LOAD_AFFINITY_CHARACTERIZED` to:

`RUNTIME_TRANSITION_JITTER_CHARACTERIZED`

This is not `CPU_SINGS`, not `BYTE_READY_HUMAN_REVIEW`, and not `SOFTWARE_FIRMWARE_TRUE_WALL`.

## Next Action

`RUNTIME_STATE_WINDOW_ORACLE`

Build a read-only state-windowed software harness that:

- samples COFVID/PSTATE_STATUS,
- bins timing observations by runtime state,
- compares state-conditioned distributions against deterministic nulls,
- does not use external measurement,
- does not write MSRs,
- accepts success only if state-conditioned timing distributions separate reproducibly beyond nulls.
