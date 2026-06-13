# Undervolt Pathway 5: Runtime K10 P-state MSR Control

## Status

`RUNTIME_MSR_PATHWAY_VERIFIED_CANDIDATE`

Candidate quality: `STRONG_CANDIDATE`

Risk tags: `HUMAN_WRITE_ONLY`

## Why This Is A Separate Candidate

The NB PCI register path explains the floor, but the executable software path on AMD Family 10h is runtime manipulation of P-state MSRs and `COFVID` state. This is also the route used by historical K10 undervolt tools such as k10ctl and TurionPowerControl.

Local evidence already proves part of the path:

- P-state definition MSRs are readable and writable enough for DID/frequency changes.
- DID divisors from 100 MHz to 3200 MHz were confirmed locally.
- P-state entry/exit cycling is required for a definition write to take effect.
- The existing lower-VID attempts may have failed because hardware MinVid/MaxVid filtering clamps the request, or because both `CpuVid` and `NbVid` were not advanced through a safe incremental test.

## Exact Control Registers

- `MSRC001_0064` through `MSRC001_0068`: P-state definition registers.
- `MSRC001_0062`: P-state command/control.
- `MSRC001_0071`: COFVID status/readback.
- `MSRC001_0070`: COFVID control; roadmap says direct writes are overridden, so use P-state definition + transition first.

## Safe Human-run Step

The next test should not jump to VID `0x3A`. It should test VID `0x20` at low frequency with both `CpuVid` and `NbVid` set to `0x20`, then read `COFVID_STS` to prove whether the hardware accepts or clamps it.

Use the command block in [UNDERVOLT_PATHWAY_2_NB_PCI_CONFIG.md](UNDERVOLT_PATHWAY_2_NB_PCI_CONFIG.md). The exact test P4 value is:

```text
0x80000135400040c0
```

Decode:

- P-state: P4 (`MSRC001_0068`)
- FID: `0x00`
- DID: `0x03`, about 200 MHz
- CpuVid: `0x20`
- NbVid: `0x20`

Rollback value:

```text
0x8000013540003440
```

## Acceptance Criteria

Candidate succeeds if:

- `rdmsr -p 4 0xc0010068` reads back `80000135400040c0`.
- After P-state cycling, `rdmsr -p 4 0xc0010071` reports DID `3` and VID `0x20` or a clearly lower-voltage accepted VID.
- Temperature remains below the lab limit.
- SSH and housekeeping cores remain responsive.

Candidate fails or remains clamped if:

- `MSRC001_0068` accepts the write but `COFVID_STS` returns VID `0x1A` or another floor value.
- The core ignores the request or the controller overwrites it.
- A watchdog, lockup, or thermal event occurs.

## Next Values If VID 0x20 Is Accepted

Do not jump directly to `0x3A`. Human-run sequence:

1. `0x20` at DID `3`.
2. `0x24` at DID `3`.
3. `0x28` at DID `3`.
4. `0x2C` at DID `3`.
5. `0x30` at DID `3`.
6. Stop before `0x3A` unless every previous step is stable and readback-proven.

Each step needs a rollback to `0x8000013540003440` and a `COFVID_STS` readback.

## Decision

This is the best next non-flash route. It has the lowest recovery burden and directly tests whether the observed floor is a controller clamp or an artifact of previous write method.
