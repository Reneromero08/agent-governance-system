<!-- CONTENT_HASH: 4fc8ab000c1b8855056d37edc5c8f16cc7014ab4aebc057187914efe047c773b -->

# SYSTEM FAILURE PROTOCOL 1 🚨

## Status
- **Date**: 2025-12-- *- **Total Failures**: 57 (Original), now significantly reduced.
- **Objective**: Distributed repair of 6-bucket pathing and architecture compliance errors.

## Assignments

### Cohort A: User Subagents (50%) - 29 Tests
*These automated agents handle the bulk of mechanical path updates.*

- [ ] `CAPABILITY/TESTBENCH/test_adversarial_pipeline_resume.py`
- [ ] `CAPABILITY/TESTBENCH/test_ags_phase6_bridge.py`
- [ ] `CAPABILITY/TESTBENCH/test_ags_phase6_capability_pins.py`
- [ ] `CAPABILITY/TESTBENCH/test_ags_phase6_capability_registry.py`
- [ ] `CAPABILITY/TESTBENCH/test_ags_phase6_capability_revokes.py`
- [ ] `CAPABILITY/TESTBENCH/test_ags_phase6_capability_versioning_semantics.py`
- [ ] `CAPABILITY/TESTBENCH/test_ags_phase6_mcp_adapter_e2e.py`
- [ ] `CAPABILITY/TESTBENCH/test_ags_phase6_registry_immutability.py`
- [ ] `CAPABILITY/TESTBENCH/test_ags_phase6_router_slot.py`
- [ ] `CAPABILITY/TESTBENCH/test_ags_phase8_model_binding.py`
- [x] `CAPABILITY/TESTBENCH/test_cortex_integration.py` (Fixed by Antigravity)
- [x] `CAPABILITY/TESTBENCH/test_governance_coverage.py` (Fixed by Antigravity)
- [x] `CAPABILITY/TESTBENCH/test_memoization.py` (Fixed by Antigravity)
- [x] `CAPABILITY/TESTBENCH/test_packing_hygiene.py` (Fixed by Antigravity)
- [ ] `CAPABILITY/TESTBENCH/test_phase7_acceptance.py`
- [ ] `CAPABILITY/TESTBENCH/test_phase8_router_receipts.py`
- [x] `CAPABILITY/TESTBENCH/test_pipeline_chain.py` (Fixed by Antigravity)
- [x] `CAPABILITY/TESTBENCH/test_pipeline_verify_cli.py` (Fixed by Antigravity)
- [x] `CAPABILITY/TESTBENCH/test_pipelines.py` (Fixed by Antigravity)
- [x] `CAPABILITY/TESTBENCH/test_restore_runner.py` (Fixed by Antigravity)
- [x] `CAPABILITY/TESTBENCH/test_runtime_guard.py` (Fixed by Antigravity)
- [x] `CAPABILITY/TESTBENCH/test_schemas.py` (Fixed by Antigravity)
- [x] `CAPABILITY/TESTBENCH/test_swarm_reuse.py` (Fixed by Antigravity)
- [x] `CAPABILITY/TESTBENCH/test_swarm_runtime.py` (Fixed by Antigravity)
- [ ] `CAPABILITY/TESTBENCH/test_verifier_freeze.py`
- [ ] `CAPABILITY/TESTBENCH/test_pipeline_dag.py` (In Progress)
- [ ] `CAPABILITY/TESTBENCH/test_ledger_consistency.py` (Provisional)
- [ ] `CAPABILITY/TESTBENCH/test_verifier_consistency.py` (Provisional)

### Cohort B: Antigravity (Me) (25%) - 14 Tests
*I handle tests requiring complex path logic or file system interaction fixes.*
**STATUS: COMPLETE**

1. [x] `CAPABILITY/TESTBENCH/test_runtime_guard.py::test_guarded_write_text_succeeds`
2. [x] `CAPABILITY/TESTBENCH/test_runtime_guard.py::test_guarded_mkdir_succeeds`
3. [x] `CAPABILITY/TESTBENCH/test_runtime_guard.py::test_guarded_write_bytes_succeeds`
   *   *(Issue: `RuntimeError: [WRITE_GUARD] WRITE_GUARD_PATH_NOT_ALLOWED` in `fs_guard.py`. Needs `LAW/CONTRACTS` update in guard logic.)*
4. [x] `CAPABILITY/TESTBENCH/test_schemas.py::test_all_schemas`
   *   *(Issue: `FileNotFoundError: jobspec.schema.json` at old path. Needs `LAW/SCHEMAS` update.)*
5. [x] `CAPABILITY/TESTBENCH/test_restore_runner.py::test_restore_rejects_when_verifier_strict_fails`
6. [x] `CAPABILITY/TESTBENCH/test_restore_runner.py::test_restore_rollback_failure_returns_restore_rollback_failed`
   *   *(Issue: `KeyError: 'RESTORE_VERIFIER_FAILED'` and `ModuleNotFoundError`. Needs path/import fix.)*
7. [x] `CAPABILITY/TESTBENCH/test_cortex_integration.py`
   *   *(Issue: `AssertionError: FILE_INDEX should have entries`.)*
8. [x] `CAPABILITY/TESTBENCH/test_governance_coverage.py`
   *   *(Issue: `AssertionError` on roadmap exists check.)*
9. [x] `CAPABILITY/TESTBENCH/test_memoization.py`
   *   *(Issue: `ModuleNotFoundError: No module named 'catalytic_runtime'`.)*
10. [x] `CAPABILITY/TESTBENCH/test_packing_hygiene.py` (3 failures)
    *   *(Issue: `Failed: DID NOT RAISE <class 'ValueError'>`.)*
11. [x] `CAPABILITY/TESTBENCH/test_pipeline_chain.py` (4 failures)
    *   *(Issue: Likely pathing in chain verification.)*
12. [x] `CAPABILITY/TESTBENCH/test_pipeline_verify_cli.py` (4 failures)
13. [x] `CAPABILITY/TESTBENCH/test_pipelines.py`
14. [x] `CAPABILITY/TESTBENCH/test_swarm_reuse.py`

### Cohort C: User Ants (25%) - 14 Tests
*Local mechanical ants handle remaining pattern matches.*
***STATUS: DISPATCHED (TURBO SWARM: 4x qwen2.5:1.5b)**
*   `test_adversarial_pipeline_resume.py`: Fixed ✅
*   `test_ags_phase6_capability_revokes.py`: Fixed ✅
*   Remaining 13 files: Processing in parallel...

- [ ] `CAPABILITY/TESTBENCH/test_ags_phase6_bridge.py` (Specific sub-tests)
- [ ] `CAPABILITY/TESTBENCH/test_ags_phase6_capability_pins.py` (Specific sub-tests)
- [ ] `CAPABILITY/TESTBENCH/test_ags_phase6_capability_revokes.py` (Specific sub-tests)
- [ ] `CAPABILITY/TESTBENCH/test_ags_phase6_mcp_adapter_e2e.py` (Specific sub-tests)
- [ ] `CAPABILITY/TESTBENCH/test_ags_phase6_registry_immutability.py` (Specific sub-tests)
- [ ] `CAPABILITY/TESTBENCH/test_ags_phase6_router_slot.py` (Specific sub-tests)
- [ ] `CAPABILITY/TESTBENCH/test_phase7_acceptance.py` (Specific sub-tests)
- [ ] `CAPABILITY/TESTBENCH/test_phase8_router_receipts.py` (Specific sub-tests)
[ ] `CAPABILITY/TESTBENCH/test_phase8_router_receipts.py` (Specific sub-tests)
[ ] `CAPABILITY/TESTBENCH/test_phase8_router_receipts.py` (Specific sub-tests)

---
**Protocol**:
1. User spawns Cohort A.
2. Antigravity executes Cohort B.
3. User Dispatcher assigns Cohort C.
