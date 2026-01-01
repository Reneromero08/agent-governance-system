---
title: "Commit Plan Bundle"
section: "report"
author: "System"
priority: "Low"
created: "2025-12-28 12:00"
modified: "2025-12-28 12:00"
status: "Archived"
summary: "Commit plan for bundle system (Archived)"
tags: [commit_plan, bundle, archive]
---

<!-- CONTENT_HASH: ea3ab47599d354d5f20fc9155d27824888e24db1f787345aa359fe1a00279369 -->

# Commit Plan: Phase 5 - Bundle (Translation Protocol) MVP

## Overview
Implement deterministic, bounded, fail-closed executable bundle from completed jobs with integrity verification.

## Changed Files

### New Files (3)
1. `THOUGHT/LAB/CAT_CHAT/SCHEMAS/bundle.schema.json`
   - JSON Schema defining bundle structure
   - Enforces `additionalProperties: false` at all levels
   - Required fields: bundle_version, bundle_id, run_id, job_id, message_id, plan_hash, steps, inputs, artifacts, hashes, provenance

2. `THOUGHT/LAB/CAT_CHAT/catalytic_chat/bundle.py`
   - `BundleBuilder` class: deterministic bundle construction from completed jobs
   - `BundleVerifier` class: integrity verification with fail-closed validation
   - Enforces: job completeness gate, boundedness (no ALL slices), ordering, hash verification
   - Pure implementation: no wall-clock time in bundle output

3. `THOUGHT/LAB/CAT_CHAT/tests/test_bundle.py`
   - Unit tests for canonical JSON and SHA256 consistency
   - Integration tests for bundle build/verify

### Modified Files (1)
4. `THOUGHT/LAB/CAT_CHAT/catalytic_chat/cli.py`
   - Added `bundle` subcommand group
   - Added `cmd_bundle_build`: builds bundles from completed jobs
   - Added `cmd_bundle_verify`: verifies bundle integrity

## Governance Verification

### Determinism Enforcement
- **Canonical JSON**: `sort_keys=True`, separators=(",", ":"), no trailing spaces, single `\n` at EOF
- **bundle_id Algorithm**: SHA256 of pre-manifest (bundle_id=""), then serialize again with computed ID
- **root_hash**: SHA256 of `"\n".join(artifact_id + ":" + sha256) + "\n"`
- **Ordering**: Steps by (ordinal asc, step_id asc), artifacts by artifact_id asc, inputs lexicographically
- **Pure**: No wall-clock time in bundle output (completion_timestamp removed from provenance)

### Boundedness Enforcement
- **Job Completeness Gate**: All steps must be COMMITTED with exactly one receipt per step
- **No ALL slices**: Forbidden sentinel "all" rejected in both build and verify
- **Exact References**: Every artifact in bundle must be referenced by a step's refs+constraints
- **No Extras**: Step references validation ensures no artifacts beyond those referenced

### Fail-Closed Guarantees
- Missing refs → BundleError
- Schema mismatch → BundleError
- Hash mismatch → BundleError
- ALL slice → BundleError
- Unordered steps/artifacts → BundleError
- Forbidden fields (timestamp, cwd, etc.) → BundleError
- Absolute paths → BundleError

## Commit Message (Draft)

```
Phase 5: Bundle (Translation Protocol) MVP

Implement deterministic, bounded, fail-closed executable bundle from completed jobs.

- BundleBuilder: Construct bundles from completed jobs with completeness gate
- BundleVerifier: Verify integrity, ordering, hashes, and boundedness
- Bundle CLI: bundle build --run-id --job-id --out, bundle verify --bundle
- Schema: JSON Schema with additionalProperties: false enforcement

Determinism: Canonical JSON, bundle_id SHA256 algorithm, ordering rules
Boundedness: Job completeness gate, no ALL slices, exact step references
Fail-closed: Schema validation, hash verification, ordering enforcement
Pure: No wall-clock time in bundle output (completion_timestamp omitted)
```

## Testing

### Unit Tests (PASSING)
- `test_canonical_json_produces_deterministic_output`
- `test_sha256_produces_consistent_hash`

### Integration Tests (Pending)
- `test_bundle_build_deterministic_bytes` - Requires completed job fixture
- `test_bundle_verify_passes` - Requires completed job fixture
- `test_bundle_verify_fails_on_tamper` - Requires completed job fixture
- `test_bundle_is_bounded_and_rejects_all` - Requires completed job fixture
- `test_bundle_build_fails_if_job_not_complete` - Requires incomplete job fixture

Note: Integration tests require proper fixture setup with indexed sections and completed jobs.

## Verification Commands

```bash
# Plan a request
py -m catalytic_chat.cli plan request --request-file "THOUGHT/LAB/CAT_CHAT/tests/fixtures/plan_request_min.json"

# Build bundle from completed job
py -m catalytic_chat.cli bundle build --run-id <RUN_ID> --job-id <JOB_ID> --out "CORTEX/_generated/bundles/test_bundle"

# Verify bundle integrity
py -m catalytic_chat.cli bundle verify --bundle "CORTEX/_generated/bundles/test_bundle"
```

## Files to Commit

```
A  THOUGHT/LAB/CAT_CHAT/SCHEMAS/bundle.schema.json
A  THOUGHT/LAB/CAT_CHAT/catalytic_chat/bundle.py
A  THOUGHT/LAB/CAT_CHAT/tests/test_bundle.py
M  THOUGHT/LAB/CAT_CHAT/catalytic_chat/cli.py
```

## Notes

- Bundle system is read-only: build/verify do not write to system DB
- Artifacts are written only to output directory (CORTEX/_generated/bundles/)
- Completeness gate ensures only COMMITTED jobs can be bundled
- Determinism audit: Removed completion_timestamp from provenance (was wall-clock time)
