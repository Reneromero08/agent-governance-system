# PHASE2_VID_FIRMWARE

## Verdict

VID_FIRMWARE_ROUTE_DESIGN_ONLY_HUMAN_APPROVAL_REQUIRED

## Runtime VID Status

The runtime VID route is already clamped. Prior decider evidence:

- P4 write accepted `0x80000135400040c0`.
- COFVID_STS stayed at CpuVid `0x1A`.
- P4 was rolled back to `0x8000013540003440`.

No runtime VID sweep was rerun.

## Firmware Status

The AGESA normalizer location is known in `AmdProcessorInitPeim`, but the global `JBE -> JAE` patch is unsafe because it is not P4-only and can affect high-frequency P-states.

Allowed future design requirements:

- P4-only/table-specific logic.
- P0-P3 safety proof.
- UEFI checksum verification.
- External recovery plan.
- Human approval before any risky action.

No BIOS flash was performed.

