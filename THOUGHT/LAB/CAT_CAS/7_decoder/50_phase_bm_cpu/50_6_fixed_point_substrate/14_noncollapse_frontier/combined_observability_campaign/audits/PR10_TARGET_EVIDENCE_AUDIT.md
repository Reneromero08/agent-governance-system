# PR #10 CAT_CAS Target Evidence Audit

## Audit subject

- PR: `Reneromero08/agent-governance-system#10`
- Qualified head: `81ea84f341b29c41b93667d0e0fb98e0975bcbcf`
- Audit packet: `pr10_target_audit_packet_81ea84f3.zip`
- Audit packet SHA-256: `7fc96e3979bf0a5757fed20b48e30a9d34b5975af9fce8d984eb9ae8e0e260ba`

## Verdict

`AUDIT_PASS_WITH_ONE_NONBLOCKING_PACKAGING_DEFECT`

The target engineering disposition remains:

`ENGINEERING_READY__ACQUISITION_UNAUTHORIZED`

No scientific acquisition authorization was present. No scientific acquisition was executed. No physical carrier restoration was claimed.

## Independently verified

### Archive and manifest integrity

- 706 ZIP entries
- No duplicate paths
- No absolute paths or `..` traversal
- No symbolic links
- Three expected top-level roots
- Source-transfer manifest: 64/64 payload files matched by size and SHA-256
- Target-evidence manifest: 224/224 payload files matched by size and SHA-256
- Final-bundle manifest: 291/291 payload files matched by size and SHA-256
- Target evidence copied into final bundle byte-for-byte
- Source-transfer files copied into final bundle byte-for-byte

### Reported top-level digests

All recomputed exactly:

- Frozen plan: `eb5a46d0a37d66910649467cf0d4e3cf947dee11fab94a36e9bdfed388455e53`
- Source-transfer bundle: `f0c8314b27ac1b853921be7fa3606f3319fe079da7b34fa6935eda51bf03375f`
- Strict binary: `0fc846a52d34b2395e254cc5a2db0bb715d1cd1ede77f8a5f6a2e940dab63037`
- Sanitizer binary: `b506f8ce1e6331c6d3cf5fcae24eacda5b709fb47c07659547d5e07b01d66ebf`
- Target evidence manifest: `a867a6d58d56d49e7ba4558a623725e7c81c2d84500b1045ff565192a8689d54`
- Final bundle: `5c6588a51ce6b806e1b7b269bafd1981256795653415e012592ad3b6313fdaca`
- Original preflight report: `7b1358513b9c847174aa02787e3d9c973268634ef196e9e9e5b1b35f21ca57a0`
- Smoke `run.json`: `5ab12d12c5ceeba40f0dd1b66e96e1ca7fd6c134e10fc5b7e5a02b8b572e12ab`
- Smoke run manifest: `8a26b0a32590b97856af66137c6513d7cbaf71b6508a56a0c0e2bd1674fa3069`

### Exact-head source and binary binding

Critical transferred source files were compared against GitHub blob IDs at exact head `81ea84f3`, including:

- `combined_pdn_hardware.c`
- `combined_pdn_hardware.h`
- `combined_pdn_runner.c`
- `make_engineering_smoke_schedule.py`
- `test_combined_pdn_runner.py`
- `catcas_preflight.py`
- `collect_target_engineering_evidence.py`
- `make_executor_source_bundle.py`
- `run_combined_campaign.py`
- `verify_run_manifests.py`
- `compile_session_schedule.py`
- `campaign_plan.py`
- `generate_campaign_plan.py`
- `verify_combined_plan_binding.py`

All checked Git blob IDs matched the sealed files.

The strict binary was rebuilt from the sealed source and reproduced the target binary SHA-256 exactly:

`0fc846a52d34b2395e254cc5a2db0bb715d1cd1ede77f8a5f6a2e940dab63037`

### Test execution

Independently rerun from the audit packet:

- Python suite: 35/35 PASS
- Rebuilt strict C suite: 29/29 PASS
- Rebuilt ASan/UBSan suite: 29/29 PASS
- Supplied target strict binary: 29/29 PASS
- Supplied target sanitizer binary: 29/29 PASS
- No sanitizer findings

The supplied ELF metadata is consistent with GCC 14.2.0. The sanitizer executable links `libasan.so.8` and `libubsan.so.1`.

### Frozen plan and validation-only campaign

- Frozen plan SHA-256 matched binding and manifest
- Plan schema validated with no errors
- Exact session set: `v2s3_seed0..5` and `v4s5_seed0..5`
- All 12 target validation inputs were semantically identical to the sealed compiled schedules
- Newline normalization changed bytes from CRLF to LF, but not any JSON object or window record
- Each session contained exactly 8,288 windows
- Each validation output contained exactly 8,288 contiguous `VALIDATED` rows
- Every validation row had `hardware_executed=0`
- Every validation `run.json` declared `VALIDATION_ONLY_HARDWARE_NOT_EXECUTED`
- Raw samples and telemetry were empty for all 12 validation runs
- Every run manifest was independently rehashed
- Validation report was regenerated from the raw directories and matched exactly
- Result: 12/12 PASS, hardware touched false

### Engineering smoke

The schedule was exactly:

1. Driven basis window
2. Driven rotation window
3. True sender-off window

Raw binary verification:

- 96,000 bytes
- 6,000 records
- 2,000 records per window
- Each record decoded as one little-endian TSC plus one IEEE-754 double
- Timestamps strictly increased within each window
- First and last sample TSCs matched CSV exactly
- All sample values were finite
- Raw mean, minimum, and maximum recomputed exactly
- Lock-in I, Q, magnitude, and floor independently recomputed from the raw samples and matched to floating-point roundoff

Timing:

| Window | Sender epoch skew | Receiver epoch skew | Ready lead | First drive after origin | Capture overrun |
|---|---:|---:|---:|---:|---:|
| 0 | 0.0498 µs | 0.0544 µs | 50.0004 ms | 0.000089 ms | 64.75 µs |
| 1 | 0.0588 µs | 0.0740 µs | 50.0004 ms | 21.3822 ms | 72.31 µs |
| 2 off | n/a | 0.0535 µs | n/a | n/a | 76.99 µs |

Window 1's first drive occurred after the first sample. This is allowed by the executor contract because the scheduled waveform may begin in an idle quadrant.

Sender-off verification:

- `sender_started=0`
- `sender_alive_at_capture=0`
- `first_drive_tsc=0`
- I/Q/magnitude/floor were null
- Sender cleanup state was stopped

The APERF/MPERF ratios were approximately 0.4974, 0.5000, and 0.5000, consistent with the recorded 1.6 GHz pin against the approximately 3.2 GHz reference domain.

### Late-sender fail-closed test

- Exit code: 5
- Failure: `SENDER_EPOCH_ALIGNMENT_FAILURE`
- Execution class: `MOCK_HARDWARE_TEST`
- `hardware_executed=false`
- Automatic retry disabled
- Host control-state restoration true
- No physical restoration claim
- Raw samples empty
- No result rows emitted

### Host control-state evidence

The raw before snapshot matched the collector report.

Before and after:

- Host: `catcas`
- Effective UID: 0
- CPU count: 6
- `constant_tsc`: present
- `nonstop_tsc`: present
- k10temp path: present
- MSR readable: cores 0 through 5
- cpufreq controls available: cores 0 through 5
- Minimum frequencies: all `800000`
- Maximum frequencies: all `3200000`
- Boost: `1`
- Remaining runner processes: none

This supports:

`HOST CONTROL-STATE RESTORATION VERIFIED`

It does not establish physical carrier restoration.

### V4 preflight

The original report was not merely trusted. Preflight was rerun against a clean clone of the sealed bundle, with the post-sealing report removed and executable mode restored after ZIP extraction.

The recomputed report matched the original semantically:

- Every engineering check true
- Every target check true
- `engineering_ready=true`
- `acquisition_authorized=false`
- `acquisition_ready=false`
- `scientific_acquisition_started=false`
- `physical_carrier_restoration_claimed=false`
- Only expected authorization error: acquisition authorization artifact not supplied

No authorization artifact was present anywhere in the packet.

## Finding

### Nonblocking packaging defect

`source_transfer_bundle.sha256` was copied from the source-transfer directory without rewriting its filename field. Its contents are:

```text
f0c8314b27ac1b853921be7fa3606f3319fe079da7b34fa6935eda51bf03375f  source_bundle.json
```

Inside the final bundle, that digest belongs to `source_transfer_bundle.json`, not the final `source_bundle.json`.

Therefore:

```text
sha256sum -c source_transfer_bundle.sha256
```

fails in the final bundle even though the digest value itself is correct.

This does not invalidate the preflight or evidence chain because the code verifies the source-transfer manifest by its content digest. It should be repaired in a follow-up change without rewriting the already-qualified head unless the target qualification is intentionally rerun.

Recommended corrected sidecar:

```text
f0c8314b27ac1b853921be7fa3606f3319fe079da7b34fa6935eda51bf03375f  source_transfer_bundle.json
```

## Replay caveats

- The Windows-created ZIP stored the Linux binaries as non-executable files. Independent replay required restoring executable mode.
- `engineering_preflight.json` is a post-sealing output and is not part of the final bundle's bound file set. Replaying preflight in place requires removing the old report or writing the report outside the sealed bundle.
- These are packaging and replay concerns, not evidence-semantic failures.

## Epistemic limit

This audit independently verifies integrity, derivation, internal consistency, executable behavior, raw numerical calculations, and claim boundaries.

It cannot cryptographically prove that the recorded physical measurements originated from the named machine at the recorded moment because the target does not provide a trusted hardware signature, TPM quote, or independent observer. The APERF/MPERF, temperature, TSC, cpufreq, and raw-sample evidence is strongly coherent with the reported target execution, but remains host-produced evidence.

## Final claim ceiling

### Proven by the audited packet

- Exact-head source and strict-binary binding
- Target compatibility evidence
- Strict and sanitizer test behavior
- Twelve-session validation-only integrity
- Shared-origin engineering-smoke timing
- Sender-off control behavior
- Late-sender fail-closed behavior
- Raw sample to reported-observable derivation
- Sealed provenance and manifest integrity
- Host control-state cleanup evidence
- V4 engineering readiness

### Not proven or authorized

- Scientific campaign result
- Twelve-session physical acquisition
- Physical observable-state restoration
- Physical carrier restoration
- Identified physical operator
- Target-to-carrier coupling
- Fold-odd invariant
- Orientation recovery
- Small Wall crossing
- Broader physical or ontological claims
