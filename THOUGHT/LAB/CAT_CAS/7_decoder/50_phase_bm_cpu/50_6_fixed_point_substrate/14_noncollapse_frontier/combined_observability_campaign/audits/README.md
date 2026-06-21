# Phase 6 Post-Acquisition Audits

This directory preserves forward-only audit results for the completed Phase 6 combined-observability acquisition.

The historical executor, plan, authorization, bundles, and evidence remain immutable.

## Audit sequence

1. `PHASE6_EXECUTOR_CODE_AUDIT_81ea84f3.md`
   Static audit of the exact executor source.

2. `PHASE6_METADATA_AUDIT_81ea84f3.md`
   Audit of all twelve session schedules, 99,456 result rows, telemetry, manifests, and raw-size closure without the large binary payloads.

3. `PHASE6_RAW_RECOVERY_V2S3_SEED0.md`
   First raw-session audit and implementation-recovery result on a training session.

4. `PHASE6_RAW_RECOVERY_V2S3_SEED0.json`
   Machine-readable result for the first raw session.

## Current state

```text
PROVENANCE_VALID
METADATA_INTEGRITY_PASS
FIRST_RAW_SESSION_PASS
REQUESTED_F_COORDINATE_REJECTED
F_OVER_4_CARRIER_SUPPORTED
EXACT_GATE_PHASE_SUPPORTED
SCRAMBLE_NULL_INVALID
IMMEDIATE_RERUN_NOT_REQUIRED
ALL_SESSION_RAW_AUDIT_REQUIRED
SCIENTIFIC_VERDICT_PENDING
```

## Analysis code

The sibling `analysis/` directory contains:

- `waveform_reference.py`
- `audit_raw_session.py`
- `test_waveform_reference.py`

These tools model the waveform that actually executed. They do not mutate or reinterpret the historical evidence object.
