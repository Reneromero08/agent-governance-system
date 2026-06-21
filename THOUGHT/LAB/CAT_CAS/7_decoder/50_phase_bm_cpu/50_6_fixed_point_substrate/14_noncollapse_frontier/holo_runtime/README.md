# Phase 6 Combined PDN Runner

`combined_pdn_runner.c` is the schedule-driven executor boundary for the frozen combined-observability campaign.

Current status:

```text
ENGINEERING_HARDENING_IMPLEMENTED
TARGET_REVALIDATION_REQUIRED
SCIENTIFIC_ACQUISITION_NOT_AUTHORIZED
PHYSICAL_CARRIER_RESTORATION_NOT_CLAIMED
```

## Executor contract

The runner:

- consumes the compiled `session.json`, `windows.jsonl`, and `session_manifest.json` process object directly;
- verifies the session manifest before reading any window;
- refuses an existing output directory;
- requires contiguous window order and one of the two frozen routes (`v4s5` or `v2s3`);
- rejects sender-off windows containing an active drive;
- accepts only a nonzero lowercase 40-character hexadecimal executor commit in hardware mode;
- supports hardware-free `--validate-only`, authorization-gated `--hardware`, exact-shape `--engineering-smoke`, and test-only `--mock-hardware` modes;
- refuses every real scientific hardware schedule unless an acquisition artifact matches the executor commit, plan digest, owner, and authorized output root;
- permits unauthorized real hardware only for the exact two-low-amplitude-plus-one-sender-off engineering-smoke schedule;
- pins and readies the sender before publishing a future shared TSC origin;
- records sender-ready, sender-epoch, first-drive, receiver-epoch, first-sample, and last-sample TSC values;
- fails closed when sender or receiver epoch skew exceeds the declared tolerance;
- applies the schedule-provided physical tone, codeword source, mode, phase, and amplitude without reordering;
- bounds capture allocation before integer conversion or memory allocation;
- creates and joins one sender per driven window and preserves true sender-off capture;
- snapshots, pins, restores, and independently reads back all six cpufreq policies plus boost;
- constructs final `run.json` only after data and log synchronization status is settled;
- flushes and `fsync`s `run.json`, then hashes it without mutation;
- labels cpufreq/boost cleanup as host control-state restoration, never physical carrier restoration;
- writes immutable evidence and a self-excluding SHA-256 run manifest.

`first_drive_tsc` is the first active ALU burst under the scheduled phase waveform. It may occur after the first receiver sample when the declared phase begins in an idle quadrant. Shared-origin synchronization is therefore judged by `sender_epoch_tsc` and `receiver_epoch_tsc`, not by requiring `first_drive_tsc <= first_sample_tsc`.

## Deployment boundary

The full Git repository and Git executable are **not** required on CAT_CAS.

```text
local clean Git checkout
→ commit-bound source-transfer bundle
→ transfer only required files to CAT_CAS
→ target compile/tests/validation/smoke
→ target evidence manifest
→ return target evidence locally
→ final sealed engineering bundle
→ engineering-only preflight
```

`make_executor_source_bundle.py` performs the local Git proof and builds the source-transfer bundle. On CAT_CAS, the same script verifies the transferred hashes; target identity is established by the sealed bundle rather than a target Git checkout.

`collect_target_engineering_evidence.py` derives—not merely asserts—the target evidence:

- `validation-report` recomputes the 12/12 validation-only result from the run directories and rejects any hardware touch;
- `snapshot` records raw host/cpufreq/MSR/temperature/process state before the smoke;
- `finalize` recomputes the three-window smoke contract, late-sender fail-closed result, cpufreq/boost restoration, and runner-process cleanup.

The local final sealer binds those returned artifacts to the original source-transfer digest and exact executor commit. `catcas_preflight.py` then independently recomputes the same facts from the sealed raw files.

## Authorization boundary

Preflight reports separate states:

```text
engineering_ready
acquisition_ready
```

`engineering_ready=true` proves only the declared engineering packet. It does not authorize the scientific campaign.

`acquisition_ready=true` additionally requires a distinct `CAT_CAS_PHASE6_ACQUISITION_AUTHORIZATION_V1` artifact bound to the final bundle, executor commit, campaign plan, authorized output root, explicit project-owner identity, and `restoration_authorized=false`.

Gate R remains non-authorizing. The 12-session scientific hardware campaign and every restoration experiment remain blocked until separately authorized.

## Local build and test

```bash
gcc -std=c11 -O2 -pthread -Wall -Wextra -Werror \
  combined_pdn_runner.c combined_pdn_hardware.c \
  -o combined_pdn_runner -lm
python3 test_combined_pdn_runner.py
```

## Evidence replay caveats

- Archive formats such as ZIP may not preserve Linux executable mode bits. Restore the runner's executable mode before replaying target tests, but never alter its bytes; verify the recorded SHA-256 after extraction.
- `engineering_preflight.json` is generated after the final engineering bundle is sealed. Replay preflight against an untouched clone of the sealed bundle and write the new report outside that clone, or remove the prior post-sealing report before rerunning.
- Every generated bundle-local `.sha256` sidecar must name the file as it exists in the same directory and must pass `sha256sum -c` from that directory. Renaming a manifest therefore requires rewriting its sidecar, not merely renaming the sidecar file.
- These replay rules preserve engineering evidence. They do not authorize scientific acquisition or physical restoration claims.
