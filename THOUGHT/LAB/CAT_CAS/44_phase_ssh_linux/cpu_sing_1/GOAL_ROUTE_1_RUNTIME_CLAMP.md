# GOAL_ROUTE_1_RUNTIME_CLAMP

## Verdict

RUNTIME_VID_ROUTE_CLAMPED_FINAL

## Evidence

`RUNTIME_VID_DECIDER_PACK.md` contains the decisive runtime VID test.

Observed:

- Core 4 stock P4 before write: `8000013540003440`.
- Test P4 written and read back: `80000135400040c0`.
- Requested CpuVid in test value: `0x20`.
- COFVID_STS after P-state transition: `1800001400434c0`.
- Decoded COFVID_STS CpuVid after transition: `0x1A`.
- Rollback P4: `8000013540003440`.

## Decision

The P-state definition MSR write mechanism works, but hardware output VID does not go below the observed floor. Runtime VID sweep is rejected unless contradictory evidence appears.

