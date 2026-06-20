# Phase 6 Combined PDN Runner

`combined_pdn_runner.c` is the schedule-driven executor boundary for the frozen combined-observability campaign.

Current status:

```text
ENGINEERING_HARDENING_IMPLEMENTED
TARGET_REVALIDATION_REQUIRED
SCIENTIFIC_ACQUISITION_NOT_AUTHORIZED
PHYSICAL_CARRIER_RESTORATION_NOT_CLAIMED
```

The runner enforces the non-negotiable validation and hardware invariants:

- consumes compiled `session.json`, `windows.jsonl`, and `session_manifest.json` directly;
- verifies the session manifest before reading scientific windows;
- refuses an existing output directory;
- requires contiguous `window_index` order;
- rejects sender-off windows with any active drive;
- rejects unsupported measurement modes;
- accepts only a nonzero lowercase 40-character hexadecimal executor commit in hardware mode;
- supports hardware-free `--validate-only` and explicit `--hardware` modes;
- pins the receiver and arms the sender before choosing a future shared TSC origin;
- records sender-ready, sender-epoch, first-drive, receiver-epoch, first-sample, and last-sample TSC values;
- rejects sender or receiver epoch skew beyond the declared engineering tolerance;
- applies schedule-provided physical tone, codeword source, actual mode, phase, and amplitude without reordering;
- creates and joins a sender per driven window and proves no sender is alive for sender-off capture;
- snapshots, pins, restores, and verifies all cpufreq policies plus boost on every exit path;
- constructs `run.json` once, flushes and `fsync`s it, then hashes it without mutation;
- labels cpufreq/boost cleanup as host control-state restoration, not physical carrier restoration;
- writes immutable evidence and a self-excluding SHA-256 run manifest.

`first_drive_tsc` is the first active ALU burst under the scheduled phase waveform. It may occur after the first receiver sample when the declared phase begins in an idle quadrant. Synchronization is therefore proved by `sender_epoch_tsc` and `receiver_epoch_tsc` relative to the shared origin, not by requiring `first_drive_tsc <= first_sample_tsc`.

## Evidence and authorization

`make_executor_source_bundle.py` must be run from a clean repository at the exact executor commit. It reconstructs committed sources with `git show`, builds the strict and sanitizer binaries, runs the test suites, compiles and validates all 12 schedules, and produces a hash-bound engineering bundle.

`catcas_preflight.py` reports two separate states:

```text
engineering_ready
acquisition_ready
```

`engineering_ready=true` does not authorize scientific acquisition. `acquisition_ready=true` additionally requires a separate `CAT_CAS_PHASE6_ACQUISITION_AUTHORIZATION_V1` artifact bound to:

- the exact source-bundle SHA-256;
- executor commit;
- campaign-plan SHA-256;
- authorized output root;
- `restoration_authorized=false`;
- an explicit project-owner identity.

Gate R remains non-authorizing. No hardware scientific campaign may start from engineering evidence alone.

## Build and test

```bash
make test
```
