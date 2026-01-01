<!-- CONTENT_HASH: 63182a2c2362082c6a9968becb3ac210a802c948bcb61d5e4b3f872c1e3f965a -->

# SYSTEM FAILURE PROTOCOL FINAL (Consolidated) 🚨

## Overview
This document consolidates **SYSTEM FAILURE PROTOCOL 1** and **2** into a single source of truth for the project's current failure state and recovery efforts. It tracks the migration to the 6-bucket architecture and the stabilization of the test suite.

## Global Status
- **Consolidated Date**: 2025-12-30
- **Current Pass Rate**: 124 / 134 (92.5%)
- **Total Outstanding Failures**: 10
- **Objective**: Reach 100% test pass rate and verify full compliance with **SPECTRUM-01** through **SPECTRUM-06**.

---

## 1. Inherited Challenges (Legacy Logs)

### Category A: 6-Bucket Architecture Compliance
*Inherited from Protocol 1. Focuses on pathing and domain isolation.*
- [x] `fs_guard.py` path allowed list update for `LAW/CONTRACTS`.
- [x] `LAW/SCHEMAS` pathing for `jobspec.schema.json`.
- [x] Distributed repair of core primitives (`ledger.py`, `cas_store.py`).
- [x] Verification of `ags` tool functionality in `CAPABILITY/TOOLS`.

### Category B: Test Code Bugs & Setup Issues
*Inherited from Protocol 2. Focuses on test script errors.*
- [ ] **Missing Variable Definitions**: Fix tests where variables like `pipeline_dir`, `pins_path`, or `original_chain` are used without being defined.
- [ ] **Assertion Mismatches**: Update tests to expect current system error codes (e.g., `CAPABILITY_HASH_MISMATCH` instead of old generic messages).
- [x] **Import Error Cleanup**: Resolve `ImportError` or `ModuleNotFoundError` caused by renamed modules (e.g., `memoization` -> `memo_cache`).

---

## 2. Active Failures (New/Persistent) 🚨
*Identified during current session run on 134 collected tests.*

| Test File | Failure Reason | Priority |
| :--- | :--- | :--- |
| `CAPABILITY/TESTBENCH/integration/test_demo_memoization_hash_reuse.py` | `_demos` directory missing; `ImportError` (fixed) | High |
| `CAPABILITY/TESTBENCH/integration/test_memoization.py` | State persistence / invalidate logic mismatch | Medium |
| `CAPABILITY/TESTBENCH/phases/phase6_governance/test_ags_phase6_bridge.py` | Calling non-existent `CAPABILITY.PIPELINES.ags` | High |
| `CAPABILITY/TESTBENCH/phases/phase6_governance/test_ags_phase6_capability_revokes.py` | Expected message mismatch | Medium |
| `CAPABILITY/TESTBENCH/phases/phase7_swarm/test_phase7_acceptance.py` | Bind failure / missing `original_chain` | High |
| `CAPABILITY/TESTBENCH/phases/phase7_swarm/test_swarm_reuse.py` | Execution elision logic mismatch | Medium |
| `CAPABILITY/TESTBENCH/spectrum/test_spectrum02_resume.py` | Resume state mismatch | High |

---

## 3. Recommended Actions & Assignments

### Cohort A: Automated Refactoring (Swarm)
- [ ] Update all failing tests in `CAPABILITY/TESTBENCH/phases/` to use `CAPABILITY.TOOLS.ags` instead of `CAPABILITY.PIPELINES.ags`.
- [ ] Normalize test assertions for `REGISTRY_NONCANONICAL` and `CAPABILITY_HASH_MISMATCH`.

### Cohort B: Manual Intervention (Antigravity)
- [ ] Restore missing `_demos` artifacts or update `test_demo_memoization_hash_reuse.py` to be self-contained.
- [ ] Debug the `SwarmRuntime` chain binding issues in `test_phase7_acceptance.py`.
- [ ] Resolve the `RefResolver` / `referencing` library compatibility in `ledger.py` and `ags.py`.

### Cohort C: Cleanup
- [ ] Verify 6-bucket compliance for all newly created/modified files.
- [ ] Final purge of all leftover `.txt` and `.log` files in root.

---

**⚠️ You are bound by AGENTS.md Section 11 (The Law) - see root of repo.**
