# Phase-QEMU V0 P0 Findings

## Result

`PHASE_QEMU_V0_P0_REFERENCE_PROCESS_GEOMETRY_AND_SOURCE_ISOLATED_UPSTREAM_ENERGIZED_RINGDOWN_CALIBRATION`

The actual QEMU 10.2.4 PCI device executed the selected P0 process geometry
with hidden source, carrier, detector, environment, and pending-boundary state.
Its final boundary remained unreadable until explicit diagnostic release.
Owner and program tag mismatches were rejected before source isolation without
mutating the barrier. These guest-writable tags provide nominal command
consistency, not authenticated or multi-controller custody.

For the sealed 64-step ringdown arms:

| arm | I (Q30) | Q (Q30) | energy (Q30) | barrier | cycles |
|---|---:|---:|---:|---:|---:|
| 0 | 581926496 | 8791306 | 315453609 | 8 | 20480 |
| pi | -581926504 | -8791401 | 315453619 | 8 | 20480 |
| carrier removed | 1021 | 0 | 0 | 8 | 20480 |

The independent recurrence reproduced every boundary exactly and measured
positive carrier dissipation in both present-carrier arms. The small residual
removed-carrier I value is detector feedthrough from the still-energized
upstream source, not mechanical carrier storage.

Verification classification:

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level:

`SEPARATE_REFERENCE_PARITY`

Restoration classification:

`NO_RESTORATION_CLAIM`

## Actual migration sham

The migration control captured the hidden state after 16 of 64 ringdown
steps, loaded it into a second QEMU process, recovered the PCI BAR and memory
enable state without guest configuration replay, kept the boundary locked,
and completed to the exact uninterrupted boundary. The sealed stream is a
positive retained artifact of roughly 576 KB; its exact framing varies across
reexecution and is not treated as an entropy or compression measure.

Recovery classification:

`SNAPSHOT_RELOAD`

It is not restoration: backing identity changes, process recreation and a
retained state stream are required, no inverse executes, and restored reuse
remains rejected.

## Obstruction

The selected natural-ringdown recurrence is dissipative and has no implemented
exact reverse. This does not reject every possible physical P0 echo law.
Clearing or migration can recreate state but cannot qualify as catalytic
restoration. The independent fixed-point recurrence is an
equal-access deterministic classical shadow, so this QEMU model does not
escape M257.

The route should not continue with more P0 phase arms, ringdown durations, or
reset variants. The next model must change the dynamics and test an actual
history-free echo/reversal law on a growing relational family.

## Durable evidence

- `evidence/PHASE_QEMU_V0_P0_QTEST_RESULTS.json`
- `evidence/PHASE_QEMU_V0_P0_SEPARATE_REFERENCE.json`
- `evidence/PHASE_QEMU_V0_P0_MIGRATION_SHAM.json`
- `evidence/PHASE_QEMU_V0_BUILD_RECEIPT.json`

The migration stream, compiled QEMU binary, and unpacked upstream source stay
in managed disk-backed Scratch and are not committed as scientific payload.
