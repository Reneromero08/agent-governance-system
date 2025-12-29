# ADR-023: Capability Revocation Semantics (No-History-Break)

**Date:** 2025-12-27
**Status:** Accepted
**Confidence:** High
**Impact:** High
**Tags:** [catdpt, governance, capabilities, revocation]

## Context
Phase 6.6 introduced capability pinning and revocation to allow blocking compromised or deprecated skills. However, a global revocation list (`CAPABILITY_REVOKES.json`) would break the verification of historical runs that were valid at their execution time. We need "no-history-break" semantics where past runs remain verifiable while future use is blocked.

## Decision
We implement a unified policy snapshot mechanism:

1.  **Unified Policy Artifact:** Every pipeline execution must capture its governance state into a canonical `POLICY.json` file in the pipeline directory.
2.  **Revocation Snapshot:** At `ags run` time, the current list of revoked capabilities is snapshotted into `POLICY.json`.
3.  **Verification logic:** `pipeline_verify.py` must use the `revoked_capabilities` list found in the pipeline's `POLICY.json` instead of the global repo-state list. This ensures that if a capability is revoked *after* a run completes, that run still passes verification (as its snapshot list will not contain the new revocation).
4.  **Route enforcement:** `ags route` continues to use the global `CAPABILITY_REVOKES.json` to block the creation of *new* plans referencing revoked capabilities.

## Consequences
**Positive:**
- Historical verification is preserved (no-history-break).
- Governance policy is artifact-bound and immutable for each run.
- Unified storage for AGS preflight/admission verdicts and Catalytic revocation lists.

**Negative:**
- `POLICY.json` must be present for revocation checks to be accurate during verification.
- Slight increase in artifact size per pipeline.

## Compliance
- `TOOLS/ags.py` updated to write unified `POLICY.json`.
- `CATALYTIC-DPT/PIPELINES/pipeline_verify.py` updated to honor policy snapshots.
- `CATALYTIC-DPT/PIPELINES/pipeline_runtime.py` updated to use `POLICY.json` as the source of truth.
