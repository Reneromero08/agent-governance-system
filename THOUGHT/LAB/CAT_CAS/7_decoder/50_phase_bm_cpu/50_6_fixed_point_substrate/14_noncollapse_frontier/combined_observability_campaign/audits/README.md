# Phase 6 Post-Acquisition Audits

This directory preserves forward-only audit results for the completed Phase 6 combined-observability acquisition.

The historical executor, plan, authorization, bundles, and evidence remain immutable.

## Current audit sequence

1. Static executor source audit.
2. Twelve-session metadata and manifest audit.
3. Raw-session waveform recovery, beginning with training session `v2s3_seed0`.
4. Full campaign recovery analysis with training-frozen coordinates.
5. Original frozen adjudication and implementation-recovery adjudication kept in separate ledgers.

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

These tools model the waveform that actually executed. They do not mutate or rewrite the historical evidence object.

## Future executor

`../EXECUTOR_V2_REMEDIATION.md` defines the forward-only correction and qualification contract for any future acquisition.
