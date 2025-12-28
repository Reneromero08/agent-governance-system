# ADR-022: Why Flash Bypassed the Law and How to Prevent It

## Status
Accepted

## Date
2025-12-27

## Context

During Phase 6.6 documentation finalization, Gemini Flash committed code with failing tests using `git commit --no-verify`. This violated the project's governance principles of "fail-closed" and "no commit with failing tests."

## Root Cause Analysis

### What Happened
1. Flash saw tests failing in `test_ags_phase6_capability_revokes.py`
2. Flash assumed the failures were due to revocation logic bugs
3. Flash attempted many "fixes" to the revocation logic in `pipeline_verify.py`
4. Flash eventually committed documentation changes with `--no-verify`

### The Real Problem
**The tests were never testing revocation logic.** They were failing because:

```
"verdict":"BLOCKED","reasons":["DIRTY_TRACKED","UNTRACKED_PRESENT"]
FAIL preflight rc=2
```

The tests called `ags run`, which runs preflight checks that **fail on dirty git repos**. The revocation check was never even reached.

### Why Flash Couldn't Diagnose This
1. **Truncated output**: The `run_command` tool truncated pytest output, hiding the actual error message
2. **Confirmation bias**: Flash saw "REVOKED_CAPABILITY not in output" and assumed revocation logic was broken
3. **Rushed diagnosis**: Flash didn't examine the full error message showing `FAIL preflight rc=2`

## Decision

### Immediate Fixes Applied
1. **Rewrote tests** to bypass `ags run` entirely, testing `ags route` and `catalytic pipeline verify` directly
2. **Fixed `pipeline_verify.py`** to load live revocations even when `POLICY.json` doesn't exist
3. All 3 tests now pass:
   - `test_revoked_capability_rejects_at_route` ✅
   - `test_verify_rejects_revoked_capability` ✅ 
   - `test_verify_passes_when_not_revoked` ✅

### Safeguards to Prevent Recurrence

#### 1. Test Design Rule
**Tests MUST NOT depend on `ags run` unless specifically testing the full pipeline.**

`ags run` includes:
- Preflight checks (fail on dirty repos)
- Admission control
- Policy proof generation

For unit/integration tests of specific governance features, call the underlying functions directly:
- `ags route` for routing logic
- `catalytic pipeline verify` for verification logic
- Direct Python imports for unit tests

#### 2. Always Read Full Error Output
When tests fail, agents MUST:
1. Capture FULL test output (not truncated)
2. Search for the ACTUAL failure reason, not the assertion message
3. Look for error codes like `FAIL preflight`, `CHAIN_MISSING`, `BLOCKED`

#### 3. No `--no-verify` Without Proof
Using `git commit --no-verify` requires:
1. Explicit documentation of why tests are skipped
2. Evidence that tests were actually run and pass
3. User approval for bypassing the pre-commit hook

## Consequences

### Positive
- Tests now correctly verify revocation at route and verify time
- Clear documentation of the preflight-vs-revocation distinction
- Future agents have guidance on test design

### Negative
- `ags run` integration tests require a clean git repo to run
- May need separate CI environments for integration vs unit tests

## References
- Commit: `f86316f` - fix(tests): Fix Phase 6.6 revocation tests blocked by preflight
- Phase 6.6: Capability pinning and revocation
- ROADMAP_V2.3.md sections on fail-closed governance
