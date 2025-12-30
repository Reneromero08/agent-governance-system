# SYSTEM FAILURE PROTOCOL (Consolidated) 🚨

**⚠️ LAW: This document MUST remain in `CAPABILITY/TESTBENCH/`. Do NOT move to repo root.**

---

## Historical Context

This document consolidates all failure protocols from the 6-bucket architecture migration and ongoing test stabilization efforts. It preserves the complete history while providing actionable details for automated agents.

---

## PROTOCOL 1: Initial 6-Bucket Migration (2025-12-29)

### Status Summary
- **Date**: 2025-12-29
- **Total Failures**: 57 (Original) → Significantly reduced
- **Objective**: Distributed repair of 6-bucket pathing and architecture compliance errors

### Cohort A: User Subagents (50%) - 29 Tests
*Automated agents handling mechanical path updates*

- [ ] `CAPABILITY/TESTBENCH/test_adversarial_pipeline_resume.py`
  - **Issue**: Path references to old `CATALYTIC-DPT/` structure
  - **Fix**: Update imports and path constants to use `LAW/`, `CAPABILITY/`, `NAVIGATION/`
  
- [ ] `CAPABILITY/TESTBENCH/test_ags_phase6_bridge.py`
  - **Issue**: Incorrect module path `CAPABILITY.PIPELINES.ags`
  - **Fix**: Change to `CAPABILITY.TOOLS.ags`
  
- [ ] `CAPABILITY/TESTBENCH/test_ags_phase6_capability_pins.py`
  - **Issue**: Registry path expectations outdated
  - **Fix**: Update to use environment variables for registry paths
  
- [ ] `CAPABILITY/TESTBENCH/test_ags_phase6_capability_registry.py`
  - **Issue**: Schema validation path mismatches
  - **Fix**: Update schema paths to `LAW/SCHEMAS/`
  
- [ ] `CAPABILITY/TESTBENCH/test_ags_phase6_capability_revokes.py`
  - **Issue**: Error message expectations don't match current system
  - **Fix**: Update assertions to expect `REVOKED_CAPABILITY` error code
  
- [ ] `CAPABILITY/TESTBENCH/test_ags_phase6_capability_versioning_semantics.py`
  - **Issue**: Version mismatch error format changed
  - **Fix**: Update to expect `CAPABILITY_HASH_MISMATCH` instead of generic version error
  
- [ ] `CAPABILITY/TESTBENCH/test_ags_phase6_mcp_adapter_e2e.py`
  - **Issue**: MCP adapter paths need updating
  - **Fix**: Update to new MCP structure in `NAVIGATION/CORTEX/`
  
- [ ] `CAPABILITY/TESTBENCH/test_ags_phase6_registry_immutability.py`
  - **Issue**: Canonical JSON validation expectations
  - **Fix**: Update to match `REGISTRY_NONCANONICAL` error format
  
- [ ] `CAPABILITY/TESTBENCH/test_ags_phase6_router_slot.py`
  - **Issue**: Router slot allocation path issues
  - **Fix**: Update to use new routing infrastructure
  
- [ ] `CAPABILITY/TESTBENCH/test_ags_phase8_model_binding.py`
  - **Issue**: Model binding registry paths
  - **Fix**: Update to Phase 8 architecture
  
- [x] `CAPABILITY/TESTBENCH/test_cortex_integration.py` ✅
  - **Status**: Fixed by Antigravity
  - **Resolution**: Updated FILE_INDEX path expectations
  
- [x] `CAPABILITY/TESTBENCH/test_governance_coverage.py` ✅
  - **Status**: Fixed by Antigravity
  - **Resolution**: Corrected roadmap path assertions
  
- [x] `CAPABILITY/TESTBENCH/test_memoization.py` ✅
  - **Status**: Fixed by Antigravity (import errors resolved)
  - **Current Issue**: State persistence logic mismatch (see Protocol 3)
  
- [x] `CAPABILITY/TESTBENCH/test_packing_hygiene.py` ✅
  - **Status**: Fixed by Antigravity
  - **Resolution**: Corrected import paths and test data structure
  
- [ ] `CAPABILITY/TESTBENCH/test_phase7_acceptance.py`
  - **Issue**: Swarm chain binding failures
  - **Fix**: Define missing `original_chain` variable in test setup
  
- [ ] `CAPABILITY/TESTBENCH/test_phase8_router_receipts.py`
  - **Issue**: Receipt validation path mismatches
  - **Fix**: Update to Phase 8 receipt structure
  
- [x] `CAPABILITY/TESTBENCH/test_pipeline_chain.py` ✅
  - **Status**: Fixed by Antigravity
  - **Resolution**: Chain verification path updates
  
- [x] `CAPABILITY/TESTBENCH/test_pipeline_verify_cli.py` ✅
  - **Status**: Fixed by Antigravity
  - **Resolution**: CLI invocation path corrections
  
- [x] `CAPABILITY/TESTBENCH/test_pipelines.py` ✅
  - **Status**: Fixed by Antigravity
  - **Resolution**: Pipeline runtime path updates
  
- [x] `CAPABILITY/TESTBENCH/test_restore_runner.py` ✅
  - **Status**: Fixed by Antigravity
  - **Resolution**: Restore proof validation updates
  
- [x] `CAPABILITY/TESTBENCH/test_runtime_guard.py` ✅
  - **Status**: Fixed by Antigravity
  - **Resolution**: Updated `fs_guard.py` allowed paths for `LAW/CONTRACTS`
  
- [x] `CAPABILITY/TESTBENCH/test_schemas.py` ✅
  - **Status**: Fixed by Antigravity
  - **Resolution**: Schema paths updated to `LAW/SCHEMAS/`
  
- [x] `CAPABILITY/TESTBENCH/test_swarm_reuse.py` ✅
  - **Status**: Fixed by Antigravity (import errors resolved)
  - **Current Issue**: Execution elision logic mismatch (see Protocol 3)
  
- [x] `CAPABILITY/TESTBENCH/test_swarm_runtime.py` ✅
  - **Status**: Fixed by Antigravity
  - **Resolution**: Swarm runtime path updates
  
- [ ] `CAPABILITY/TESTBENCH/test_verifier_freeze.py`
  - **Issue**: Verifier freeze format validation
  - **Fix**: Update to match current bundle verification format
  
- [ ] `CAPABILITY/TESTBENCH/test_pipeline_dag.py`
  - **Issue**: DAG validation path issues
  - **Fix**: Update to new pipeline DAG structure
  
- [ ] `CAPABILITY/TESTBENCH/test_ledger_consistency.py` (Provisional)
  - **Issue**: Ledger schema validation
  - **Fix**: Update to match current ledger format
  
- [ ] `CAPABILITY/TESTBENCH/test_verifier_consistency.py` (Provisional)
  - **Issue**: Bundle verifier consistency checks
  - **Fix**: Update to SPECTRUM-02 format

### Cohort B: Antigravity (25%) - 14 Tests
**STATUS: COMPLETE** ✅

All 14 tests in this cohort were successfully fixed during the initial migration phase.

### Cohort C: User Ants (25%) - 14 Tests
**STATUS: DISPATCHED (TURBO SWARM: 4x qwen2.5:1.5b)**

- [x] `test_adversarial_pipeline_resume.py` ✅
- [x] `test_ags_phase6_capability_revokes.py` ✅ (partial - see Protocol 3)
- [ ] Remaining 12 files processing...

---

## PROTOCOL 2: Test Cleanup & Verification (2025-12-29)

### Status Summary
- **Date**: 2025-12-29
- **Total Tests**: 197
- **Passed**: 181 (92%)
- **Failed**: 16 (8%)
- **Test Run Time**: 28.77s

### Category 1: Missing Variable Definitions (5 tests)

- [ ] `test_verify_rejects_unpinned_even_if_pipeline_artifact_exists`
  - **Error**: `NameError: name 'pipeline_dir' is not defined`
  - **Fix**: Add `pipeline_dir = tmp_path / "pipeline"` before use
  - **Location**: Line ~45 in test function
  
- [ ] `test_verify_rejects_empty_pins`
  - **Error**: `NameError: name 'pipeline_dir' is not defined` AND `NameError: name 'pins_path' is not defined`
  - **Fix**: Add both variable definitions in test setup
  - **Location**: Test setup section
  
- [ ] `test_swarm_chain_binds_pipeline_proofs`
  - **Error**: `NameError: name 'original_chain' is not defined`
  - **Fix**: Define `original_chain = None` or load from expected location
  - **Location**: Assertion section
  
- [ ] `test_swarm_chain_fails_on_tampered_proof`
  - **Error**: `UnboundLocalError: local variable 'chain_path' referenced before assignment`
  - **Fix**: Initialize `chain_path` before conditional blocks
  - **Location**: Error handling section
  
- [ ] `test_crypto_dep_missing_fail_closed`
  - **Error**: `NameError: name 'patch' is not defined`
  - **Fix**: Add `from unittest.mock import patch` to imports
  - **Location**: Top of file

### Category 2: Missing Step Command (3 tests)

- [ ] `test_ags_route_deterministic_bytes`
  - **Error**: `ERROR: MISSING_STEP_COMMAND`
  - **Root Cause**: `_make_plan()` helper generates plan without `command` field in steps
  - **Fix**: Update `_make_plan()` to include `"command": ["echo", "test"]` in step spec
  - **Location**: Test helper function
  
- [ ] `test_ags_run_calls_verify_ok`
  - **Error**: `ERROR: MISSING_STEP_COMMAND`
  - **Root Cause**: Same as above
  - **Fix**: Same as above
  
- [ ] `test_pipeline_run_creates_state_when_missing`
  - **Error**: `ERROR: MISSING_STEP_COMMAND`
  - **Root Cause**: Same as above
  - **Fix**: Same as above

### Category 3: Assertion Mismatches (8 tests)

- [ ] `test_capability_versioning_in_place_upgrade_rejected`
  - **Expected**: `"capabilities version mismatch"`
  - **Actual**: `"CAPABILITY_HASH_MISMATCH"`
  - **Fix**: Update assertion to expect current error code
  - **Location**: Final assertion
  
- [ ] `test_fail_stderr`
  - **Error**: Unexpected assertion failure
  - **Fix**: Review stderr capture logic and update expectations
  
- [ ] `test_ags_plan_router_happy_path`
  - **Error**: Unexpected assertion failure
  - **Fix**: Update to match current router output format
  
- [ ] `test_ags_plan_over_output_fails_closed`
  - **Error**: Unexpected assertion failure
  - **Fix**: Update error message expectations
  
- [ ] `test_reject_invalid_hash_and_missing_hash`
  - **Error**: Validation assertion failure
  - **Fix**: Update hash validation error format expectations
  
- [ ] `test_invalid_hash_reject`
  - **Error**: Validation assertion failure
  - **Fix**: Update to match current hash rejection format
  
- [ ] `test_reject_invalid_hashes_and_expected_root_format`
  - **Error**: Validation assertion failure
  - **Fix**: Update Merkle root format expectations
  
- [ ] `test_capability_escalation_fails_closed`
  - **Expected**: `"REVOKED_CAPABILITY"`
  - **Actual**: `"ROUTER_STDERR_NOT_EMPTY"`
  - **Fix**: Update test to expect current error propagation behavior

### Category 4: Warnings (Non-critical)
- jsonschema.RefResolver deprecated warnings (not test failures)
- **Resolution**: Warnings suppressed in `ledger.py` and `ags.py` with `warnings.catch_warnings()`

---

## PROTOCOL 3: Current Active Failures (2025-12-30)

### Status Summary
- **Date**: 2025-12-30
- **Total Tests**: 134 collected
- **Passed**: 124 (92.5%)
- **Failed**: 10 (7.5%)
- **Test Run Time**: ~18s

### Detailed Failure Analysis

#### 1. `CAPABILITY/TESTBENCH/integration/test_demo_memoization_hash_reuse.py::test_phase2_demo_artifacts_are_falsifiable`
- **Priority**: HIGH
- **Error**: `AssertionError: assert False` (directory not found)
- **Root Cause**: Missing `CAPABILITY/TESTBENCH/integration/_demos/memoization_hash_reuse/` directory
- **Details**: 
  - Test expects pre-generated demo artifacts in `_demos/memoization_hash_reuse/{baseline,reuse}/`
  - Artifacts include: `PROOF.json`, `LEDGER.jsonl`, `DEREF_STATS.json`
  - These were likely excluded from git or lost during migration
- **Fix Options**:
  1. Restore `_demos/` from git history
  2. Make test self-contained by generating artifacts on-the-fly
  3. Skip test if `_demos/` doesn't exist (least preferred)
- **Recommended**: Option 2 - refactor test to be self-contained

#### 2. `CAPABILITY/TESTBENCH/integration/test_memoization.py::test_memoization_miss_then_hit_then_invalidate`
- **Priority**: MEDIUM
- **Error**: State persistence or invalidation logic mismatch
- **Root Cause**: Memoization cache invalidation not working as expected
- **Details**:
  - Test sequence: miss → hit → invalidate → miss again
  - Likely failing on the final "miss again" assertion
  - Cache may not be properly clearing on invalidation
- **Fix**: Review `memo_cache.py` invalidation logic and test expectations

#### 3. `CAPABILITY/TESTBENCH/phases/phase6_governance/test_ags_phase6_bridge.py::test_ags_cli_connectivity`
- **Priority**: HIGH
- **Error**: `ModuleNotFoundError: No module named 'CAPABILITY.PIPELINES.ags'`
- **Root Cause**: Test calling wrong module path
- **Details**:
  - Line 19: `[sys.executable, "-m", "CAPABILITY.PIPELINES.ags"]`
  - Should be: `[sys.executable, "-m", "CAPABILITY.TOOLS.ags"]`
  - AGS CLI moved from PIPELINES to TOOLS during 6-bucket migration
- **Fix**: Update line 19 in `test_ags_phase6_bridge.py`

#### 4. `CAPABILITY/TESTBENCH/phases/phase6_governance/test_ags_phase6_bridge.py::test_ags_preflight_verdict`
- **Priority**: HIGH
- **Error**: Same as #3
- **Root Cause**: Same as #3
- **Fix**: Same as #3

#### 5. `CAPABILITY/TESTBENCH/phases/phase6_governance/test_ags_phase6_bridge.py::test_ags_phase6_bridge`
- **Priority**: HIGH
- **Error**: Same as #3
- **Root Cause**: Same as #3
- **Fix**: Same as #3

#### 6. `CAPABILITY/TESTBENCH/phases/phase6_governance/test_ags_phase6_capability_revokes.py::test_revoked_capability_rejects_at_route`
- **Priority**: MEDIUM
- **Error**: Assertion mismatch on error message
- **Root Cause**: Error code format changed
- **Details**:
  - Test expects specific error message format for revoked capabilities
  - Current system may return different error code or format
  - Need to inspect actual vs expected error output
- **Fix**: Update assertion to match current `REVOKED_CAPABILITY` error format

#### 7. `CAPABILITY/TESTBENCH/phases/phase6_governance/test_ags_phase6_capability_revokes.py::test_verify_rejects_revoked_capability`
- **Priority**: MEDIUM
- **Error**: Same as #6
- **Root Cause**: Same as #6
- **Fix**: Same as #6

#### 8. `CAPABILITY/TESTBENCH/phases/phase7_swarm/test_phase7_acceptance.py::test_swarm_chain_binds_pipeline_proofs`
- **Priority**: HIGH
- **Error**: `NameError: name 'original_chain' is not defined` OR chain binding failure
- **Root Cause**: Missing variable definition in test
- **Details**:
  - Test attempts to verify chain binding between pipeline proofs
  - Variable `original_chain` used without being defined
  - May also have issues with chain path resolution
- **Fix**: 
  1. Add `original_chain` definition in test setup
  2. Verify chain path construction logic
  3. Ensure `CHAIN.json` is being generated correctly

#### 9. `CAPABILITY/TESTBENCH/phases/phase7_swarm/test_swarm_reuse.py::test_swarm_execution_elision`
- **Priority**: MEDIUM
- **Error**: Execution elision logic mismatch
- **Root Cause**: Swarm reuse detection not working as expected
- **Details**:
  - Test verifies that swarm can skip re-execution when inputs unchanged
  - Elision logic may have changed during refactoring
  - Check if memoization integration is working correctly
- **Fix**: Review swarm runtime elision logic and update test expectations

#### 10. `CAPABILITY/TESTBENCH/spectrum/test_spectrum02_resume.py::test_spectrum02_resume`
- **Priority**: HIGH
- **Error**: Resume state mismatch
- **Root Cause**: SPECTRUM-02 resume validation failing
- **Details**:
  - SPECTRUM-02 defines bundle resume semantics
  - Test verifies that interrupted runs can be resumed
  - State restoration may not be working correctly
  - Check `DOMAIN_ROOTS.json` and `OUTPUT_HASHES.json` handling
- **Fix**: 
  1. Review SPECTRUM-02 specification
  2. Verify bundle state persistence
  3. Update test to match current resume behavior

---

## Action Items by Priority

### 🔴 CRITICAL (Must Fix First)
1. **Fix AGS module path** (Failures #3, #4, #5)
   - Single line change in `test_ags_phase6_bridge.py`
   - Will immediately fix 3 failures
   
2. **Restore or refactor demo test** (Failure #1)
   - High-value integration test
   - Validates memoization end-to-end
   
3. **Fix chain binding** (Failure #8)
   - Core Phase 7 functionality
   - Blocks swarm acceptance

4. **Fix SPECTRUM-02 resume** (Failure #10)
   - Core temporal integrity feature
   - Critical for production use

### 🟡 MEDIUM (Fix After Critical)
5. **Fix revoke assertions** (Failures #6, #7)
   - Governance feature validation
   - Likely simple assertion updates
   
6. **Fix memoization invalidation** (Failure #2)
   - Cache correctness validation
   
7. **Fix swarm elision** (Failure #9)
   - Performance optimization validation

---

## Completion Criteria

- [ ] All 10 active failures resolved
- [ ] Full test suite at 100% pass rate (134/134)
- [ ] No regression in previously fixed tests
- [ ] All fixes comply with 6-bucket architecture
- [ ] SPECTRUM-01 through SPECTRUM-06 fully validated

---

**⚠️ REMINDER: This document MUST stay in `CAPABILITY/TESTBENCH/`. Moving it to repo root is FORBIDDEN.**
