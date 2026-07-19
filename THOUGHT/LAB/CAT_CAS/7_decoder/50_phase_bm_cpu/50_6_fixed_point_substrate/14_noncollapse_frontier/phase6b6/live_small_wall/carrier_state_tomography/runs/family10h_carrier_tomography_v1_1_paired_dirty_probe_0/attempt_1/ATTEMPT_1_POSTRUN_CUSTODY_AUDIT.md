# Attempt 1 Postrun Custody Audit

Result: `FAMILY10H_PAIRED_DIRTY_PROBE_Q_READOUT_CONFIRMED_PROSPECTIVE`

Scientific claim: `PUBLIC_POST_SOURCE_SCALAR_CARRIER_Q_READOUT_CONFIRMED`

This is an offline additions-only custody closure. No target contact, SSH, SCP, PMU acquisition, runtime execution, cleanup, deployment, or live action was performed by this audit.

## Remote/Local Archive Equality

- Target-reported SHA-256: `0f92bcd4c00ee78b7e78e84c86bf375ee1caf4ca8c52ae49166ea809f16ff041`
- Committed local archive SHA-256: `0f92bcd4c00ee78b7e78e84c86bf375ee1caf4ca8c52ae49166ea809f16ff041`
- Evidence-inventory archive SHA-256: `0f92bcd4c00ee78b7e78e84c86bf375ee1caf4ca8c52ae49166ea809f16ff041`
- Adjudication source-evidence archive SHA-256: `0f92bcd4c00ee78b7e78e84c86bf375ee1caf4ca8c52ae49166ea809f16ff041`
- Target-reported size: `2726728`
- Committed local archive size: `2726728`
- Equality passed: `True`

The historical controller stored matching values, but did not itself parse and enforce target/local equality before setting copy-back passed. This postrun audit closes that equality retrospectively. The archive custody is valid because the recorded target, local, inventory, and adjudication values match exactly.

## Archived Ownership Marker

- Archive member: `family10h_carrier_tomography_v1_1_paired_dirty_probe_0/.family10h_carrier_tomography_v1_1_paired_dirty_probe_0_owner`
- Marker SHA-256: `07c5f9ec36b2f3325541f27bffc08e6fa533c7738dd018e25ddc41f8abe822be`
- Marker size: `194`
- Source-authority commit match: `True`
- Manifest freeze commit match: `True`
- Transaction/run identity match: `True`
- Marker verification passed: `True`

Remote canonical paths were checked absent before deployment. The controller created the root and wrote the marker during the same attempt. The archived marker has the correct identity. Cleanup confirmed marker presence and final path absence, but did not compare marker contents before deletion. This is a non-scientific cleanup-custody observation and does not rewrite the cleanup receipt.

## Authorization Token Handling

The completed controller embedded the one-shot live nonce in committed source and command receipts. Historical evidence was not deleted or rewritten. The nonce is permanently retired and forbidden from reuse. Future controllers must supply the nonce externally at execution time, retain only its hash, redact full command receipts, and must not treat a committed nonce as an independent secret authorization factor.

## Stale Diagnostic Semantics

The prospective adjudication inherited diagnostic names `attempt_3_observation`, `attempt_3_primary_passed`, and `attempt_3_secondary_channels_failed_same_law`. In the v1.1 prospective adjudication, those fields contain diagnostics computed from the v1.1 attempt-1 packet. They should be read as `prospective_attempt_1_observation`, `prospective_attempt_1_primary_passed`, and `prospective_attempt_1_secondary_channels_failed_same_law`. This is a metadata-label defect only and does not alter metrics, gates, result class, or scientific claim.

## Future Controller Fail-Closed Regressions

- target/local archive hash mismatch
- target/local archive size mismatch
- missing or malformed target archive report
- ownership marker missing
- ownership marker content mismatch
- ownership marker source-authority mismatch
- ownership marker freeze-commit mismatch
- ownership marker run-identity mismatch
- committed or reused full live nonce
- unredacted nonce in command receipts
