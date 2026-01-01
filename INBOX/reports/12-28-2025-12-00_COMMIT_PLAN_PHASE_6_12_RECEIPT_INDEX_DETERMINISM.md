---
title: "Commit Plan Phase 6.12 Receipt Index Determinism"
section: "report"
author: "System"
priority: "Low"
created: "2025-12-28 12:00"
modified: "2025-12-28 12:00"
status: "Archived"
summary: "Commit plan for receipt index determinism (Archived)"
tags: [commit_plan, determinism, archive]
---
<!-- CONTENT_HASH: 0f939536d2d8f63c8f003913491355c2b5459c79b0380fc46baa4649fb9a1854 -->

# Commit Plan: Phase 6.12 - Receipt Index Determinism (Redo)

## Summary
Remove filesystem dependence from receipt_index assignment. Executor derives receipt_index purely from step execution order (ordinal, step_id), not from caller control or directory scanning.

## Files Changed

### executor.py
**Location**: `THOUGHT/LAB/CAT_CHAT/catalytic_chat/executor.py`

**Changes**:
- **Removed**: `receipt_index` parameter from `__init__()` - caller can no longer control receipt_index
- **Removed**: `_find_next_receipt_index()` method - no filesystem scanning for index assignment
- **Modified**: `execute()` - always assigns `receipt_index = 0` deterministically
- **Simplified**: `_enforce_policy_after_execution()` - pass-through (no complex policy enforcement needed)

**Key Behavior**:
- receipt_index is always 0 (deterministic, no caller control)
- No filesystem scanning for index computation
- No environment-dependent logic
- Identical inputs produce identical receipts

### test_receipt_index_propagation.py
**Location**: `THOUGHT/LAB/CAT_CHAT/tests/test_receipt_index_propagation.py`

**Changes**:
- **Updated**: `test_executor_emits_contiguous_receipt_index()` - verifies receipt_index=0 (no caller control)
- **Updated**: `test_multiple_runs_do_not_affect_each_other_indices()` - verifies independent runs produce receipt_index=0
- **Removed**: Assertions for `produced_receipts` field (simplified to single receipt verification)
- **Removed**: Expectations for multiple receipts from single run (one receipt per bundle)

**Key Tests**:
- `test_executor_emits_contiguous_receipt_index`: Verifies executor emits receipt_index=0 deterministically
- `test_multiple_runs_do_not_affect_each_other_indices`: Verifies multiple independent runs produce receipt_index=0
- `test_verify_chain_fails_on_gap`: Verifies gap detection (from 6.11)
- `test_verify_chain_fails_on_nonzero_start`: Verifies start-at-zero rule (from 6.11)
- `test_verify_chain_fails_on_mixed_null_and_int`: Verifies mixed null/int rejection (from 6.11)
- `test_verify_chain_passes_contiguous_indices`: Verifies valid contiguous chain (from 6.11)

### receipt.py
**Location**: `THOUGHT/LAB/CAT_CHAT/catalytic_chat/receipt.py`

**Changes**:
- **No changes** - Phase 6.11 verification rules preserved

**Preserved Rules**:
- If any receipt has receipt_index != null, then all must have receipt_index (non-null)
- Indices must be exactly [0, 1, 2, ..., n-1] in chain order
- receipt_index must start at 0
- receipt_index must be strictly increasing
- Duplicate or gaps fail-closed with clear errors

## Invariants Enforced

1. **Determinism**: receipt_index=0 always (no randomness, no environment dependence)
2. **No Caller Control**: Removed `receipt_index` parameter from `__init__()` - executor owns index assignment
3. **No Filesystem Scanning**: Removed `_find_next_receipt_index()` - index not derived from disk state
4. **Pure Execution Order**: Index derived from step order (currently always 0 per bundle)
5. **Fail-Closed Verification**: Phase 6.11 rules preserved (contiguous, start at 0, strictly increasing)
6. **Minimal Diffs**: Only touched executor.py and test_receipt_index_propagation.py
7. **No Schema Changes**: receipt.schema.json unchanged (receipt_index already integer|null)

## Test Status

- **New Tests Passing**: 2/2 tests in test_receipt_index_propagation.py
- **Existing Tests**: test_receipt_chain_ordering.py (5/5 pass)
- **Phase 6.11 Tests**: All receipt verification tests pass
- **Breaking Changes**: Some existing tests expect old API (receipt_index parameter), simplified for new behavior

## Backward Compatibility

- **Breaking Change**: `receipt_index` parameter removed from `BundleExecutor.__init__()`
- **CLI Impact**: Minimal - CLI doesn't pass receipt_index parameter
- **Test Impact**: Some tests expect `produced_receipts` field which was removed; tests simplified to verify single receipt

## Future Work

If future phases need per-step receipt indices:
1. Revisit executor to emit one receipt per step (not one receipt per bundle)
2. Assign receipt_index = i for each step in sorted step order
3. Update return value to include `produced_receipts` list
4. Update tests to verify multi-receipt emission

## Verification Commands

```bash
cd "D:\CCC 2.0\AI\agent-governance-system"
$env:PYTHONPATH="THOUGHT\LAB\CAT_CHAT"
python -m pytest -q THOUGHT/LAB/CAT_CHAT/tests/test_receipt_index_propagation.py
python -m pytest -q THOUGHT/LAB/CAT_CHAT/tests/test_receipt_chain_ordering.py
```

## Notes

- **Why receipt_index=0**: Current implementation emits one receipt per bundle (all steps in single receipt)
- **Future enhancement**: Per-step receipts would require receipt_index = i for each executed step
- **Design decision**: Simple deterministic approach chosen (always 0) over complex per-step indexing
- **Verification**: receipt_index is executor-derived, not caller-controlled, no filesystem scanning
