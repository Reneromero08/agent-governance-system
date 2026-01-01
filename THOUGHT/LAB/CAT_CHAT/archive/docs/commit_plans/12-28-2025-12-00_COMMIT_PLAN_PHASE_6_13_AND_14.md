---
title: "Commit Plan Phase 6.13 and 6.14"
section: "report"
author: "System"
priority: "Low"
created: "2025-12-28 12:00"
modified: "2025-12-28 12:00"
status: "Archived"
summary: "Commit plan for phases 6.13 and 6.14 (Archived)"
tags: [commit_plan, archive]
---

<!-- CONTENT_HASH: abc48383b5ee497419a05c474c4c7abfbc3a976636310f380e3e48a4b50c6047 -->

# Commit Plan: Phase 6.13 Multi-Validator Aggregation + Phase 6.14 External Verifier UX Improvements

## Scope
THOUGHT/LAB/CAT_CHAT/ - Multi-validator attestations with quorum validation (RECEIPT + MERKLE) and CI-friendly CLI output modes.

## Phases

### Phase 6.13: Multi-Validator Aggregation
**Goal**: Support multiple validator attestations with deterministic quorum semantics while maintaining backward compatibility.

#### Schemas (4 files)
1. `SCHEMAS/receipt.schema.json`
   - Add optional `attestations[]` array field
   - Single `attestation` field remains valid

2. `SCHEMAS/execution_policy.schema.json`
   - Add `receipt_attestation_quorum` object (required, scope)
   - Add `merkle_attestation_quorum` object (required, scope)

3. `SCHEMAS/aggregated_merkle_attestations.schema.json` (NEW)
   - Schema for array of merkle attestation objects
   - Enforces sorted order by (validator_id, public_key, build_id)

#### Code (2 files)
4. `catalytic_chat/attestation.py`
   - Add `verify_receipt_attestation_single()` - verifies single attestation
   - Add `verify_receipt_attestations_with_quorum()` - verifies multiple and enforces quorum
   - Deterministic ordering and rejection of unsorted arrays

5. `catalytic_chat/merkle_attestation.py`
   - Add `load_aggregated_merkle_attestations()` - loads aggregated attestations
   - Add `verify_merkle_attestations_with_quorum()` - verifies multiple and enforces quorum

#### Tests (1 file)
6. `tests/test_multi_validator_attestations.py` (NEW)
   - test_receipt_attestations_order_rejected_if_unsorted
   - test_receipt_quorum_passes_with_two_valid_of_two
   - test_receipt_quorum_fails_with_one_valid_of_two_when_required_two
   - test_merkle_quorum_passes_and_fails
   - test_single_attestation_backward_compatible

#### Documentation (1 file)
7. `PHASE_6.13_IMPLEMENTATION_SUMMARY.md` (NEW)
   - Implementation details and verification

### Phase 6.14: External Verifier UX Improvements
**Goal**: Add CI-friendly output modes and machine-readable summaries without changing verification semantics.

#### Code (3 files)
8. `catalytic_chat/cli_output.py` (NEW)
   - Standardized exit codes (0-3) documented in code comments
   - `format_json_report()` - format machine-readable JSON
   - `write_json_report()` - write JSON to stdout (only JSON + newline)
   - `write_info()` - write to stderr (suppressed in quiet mode)
   - `write_error()` - write to stderr (always displayed)
   - `classify_exit_code()` - classify and return appropriate exit code

9. `catalytic_chat/cli.py`
   - Add `--json` flag to `bundle verify`, `bundle run`, `trust verify`
   - Add `--quiet` flag to `bundle verify`, `bundle run`, `trust verify`
   - Update `cmd_bundle_verify()` with JSON output and exit code classification
   - Update `cmd_trust_verify()` with JSON output and exit code classification
   - Add missing variable definitions to `cmd_bundle_run()`

10. `catalytic_chat/__main__.py` (NEW)
   - Entry point for `python -m catalytic_chat.cli`

#### Tests (1 file)
11. `tests/test_cli_output.py` (NEW)
   - test_bundle_verify_json_stdout_purity
   - test_bundle_verify_json_purity_on_error
   - test_bundle_verify_exit_code_invalid_input
   - test_bundle_verify_exit_code_verification_failed
   - test_bundle_verify_quiet_mode
   - test_trust_verify_json_output
   - test_trust_verify_exit_code_invalid_input
   - test_json_deterministic_output
   - test_exit_codes_documented

#### Documentation (1 file)
12. `PHASE_6.14_IMPLEMENTATION_SUMMARY.md` (NEW)
   - Implementation details and verification

## Files Summary

### New Files (5)
- SCHEMAS/aggregated_merkle_attestations.schema.json
- catalytic_chat/cli_output.py
- catalytic_chat/__main__.py
- tests/test_multi_validator_attestations.py
- tests/test_cli_output.py
- PHASE_6.13_IMPLEMENTATION_SUMMARY.md
- PHASE_6.14_IMPLEMENTATION_SUMMARY.md

### Modified Files (4)
- SCHEMAS/receipt.schema.json
- SCHEMAS/execution_policy.schema.json
- catalytic_chat/attestation.py
- catalytic_chat/merkle_attestation.py
- catalytic_chat/cli.py

### Documentation Files (3)
- CHANGELOG.md (root and THOUGHT/LAB/CAT_CHAT/)
- CAT_CHAT_ROADMAP.md

Total: 12 files

## Commit Suggestion

### Single Commit (Recommended)
```
feat(THOUGHT/LAB/CAT_CHAT): Phase 6.13 multi-validator aggregation + Phase 6.14 external verifier UX

Phase 6.13: Add multi-validator attestations with deterministic quorum validation
- Add optional attestations[] array to receipt schema
- Add quorum policy fields to execution policy schema
- Implement aggregated merkle attestations schema
- Add verify_receipt_attestations_with_quorum() for receipt multi-validator verification
- Add verify_merkle_attestations_with_quorum() for merkle multi-validator verification
- Add comprehensive tests for ordering, quorum, and backward compatibility
- Reuses existing trust policy (strict_trust, strict_identity) for validation
- Purely additive: single attestation path unchanged

Phase 6.14: Add CI-friendly output modes and machine-readable summaries
- Add cli_output.py module with standardized exit codes (0-3)
- Add --json and --quiet flags to bundle verify/run/trust verify commands
- Add __main__.py entry point for python -m catalytic_chat.cli
- JSON output uses canonical_json_bytes() for deterministic field ordering
- No verification behavior changes; purely additive UX improvements

Verification:
- All 28 existing tests pass (backward compatibility confirmed)
- All 5 new multi-validator tests pass
- All 9 new CLI output tests pass
- Deterministic outputs confirmed for identical inputs
- Exit codes match documented values

Files changed:
- Modified: 4 files (schemas, attestation, merkle_attestation, cli)
- New: 5 files (schemas, modules, tests, documentation)
- Documentation: CHANGELOG.md, CAT_CHAT_ROADMAP.md updated
```

## Verification Commands

### Test All Tests
```bash
cd "D:\CCC 2.0\AI\agent-governance-system"
set PYTHONPATH=THOUGHT\LAB\CAT_CHAT
python -m pytest THOUGHT/LAB/CAT_CHAT/tests -v
```

### Test Multi-Validator Attestations
```bash
cd "D:\CCC 2.0\AI\agent-governance-system"
set PYTHONPATH=THOUGHT\LAB\CAT_CHAT
python -m pytest THOUGHT/LAB/CAT_CHAT/tests/test_multi_validator_attestations.py -v
```

### Test CLI Output
```bash
cd "D:\CCC 2.0\AI\agent-governance-system"
set PYTHONPATH=THOUGHT\LAB\CAT_CHAT
python -m pytest THOUGHT/LAB/CAT_CHAT/tests/test_cli_output.py -v
```

### Test Bundle Verify with JSON
```bash
cd "D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\CAT_CHAT"
set PYTHONPATH=THOUGHT\LAB\CAT_CHAT
python -m catalytic_chat.cli bundle verify --bundle <path> --json
```

## Rollback Plan

If issues arise:
1. Rollback Phase 6.14: Remove `--json`, `--quiet` flags; delete cli_output.py, __main__.py, test_cli_output.py
2. Rollback Phase 6.13: Remove attestations[] from receipt schema; remove quorum fields; delete multi-validator verification functions
3. Restore CHANGELOG.md and CAT_CHAT_ROADMAP.md to previous version

## Notes

- Both phases are purely additive and backward compatible
- No changes to core verification logic or trust primitives
- All changes localized to THOUGHT/LAB/CAT_CHAT/
- Deterministic outputs guaranteed via canonical_json_bytes()
- Standardized exit codes enable CI integration
- JSON output format stable and machine-readable
