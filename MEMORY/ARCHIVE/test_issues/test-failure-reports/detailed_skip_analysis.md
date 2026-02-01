# Detailed Test Skip/XFail Analysis

**Generated:** 2026-02-01
**Scope:** CAPABILITY/TESTBENCH/ 

## Summary

| Category | Count | Reason |
|----------|-------|--------|
| **Expected Failures (xfail)** | **2 tests** | Known limitations needing larger corpora |
| **Slow Tests (auto-skipped)** | **2 tests** | Stress tests requiring --run-slow flag |
| **Unconditionally Skipped** | **6 tests** | Features removed, replaced by cassette network |
| **Conditionally Skipped (skipif)** | **5 tests** | Platform or dependency requirements |

---

## Expected Failures (@pytest.mark.xfail) - 2 tests

These tests are expected to fail and do NOT block the push gate:

### 1. test_alpha_range
**File:** `CAPABILITY/TESTBENCH/cassette_network/cross_model/test_eigenstructure_alignment.py:158`

**Reason:** "Requires corpus size >> embedding dimension for valid alpha. Q21 validates with larger corpora."

**Details:**
- Tests that all models have alpha in range [0.4, 0.6]
- Alpha calculation requires corpus larger than embedding dimension
- Current test uses 20 samples in 384-d space = rank-deficient covariance
- Q21 validates alpha ~ 0.5 with proper corpora
- These tests document the calculation but may not achieve alpha ~ 0.5

### 2. test_transform_on_held_out_data
**File:** `CAPABILITY/TESTBENCH/cassette_network/cross_model/test_transform_discovery.py:336`

**Reason:** "15 training examples insufficient for stable cross-model transform. Needs larger corpus."

**Details:**
- Tests that transform learned on training data works on held-out data
- Uses Procrustes alignment on 15 training examples
- Overfits with small training sets
- Cross-model transforms require larger corpora for stability

---

## Slow Tests (Auto-Skipped by conftest.py) - 2 tests

These tests are marked with @pytest.mark.slow and automatically skipped during the push gate (unless --run-slow is passed):

**File:** `CAPABILITY/TESTBENCH/cassette_network/determinism/test_determinism.py:306-341`
**Class:** TestDeterminismAtScale

### 1. test_100_run_embedding_stability
**Why skipped:** Stress test running embedding 100 times to verify identical results
**Impact:** Would take significant time to run 100 embedding operations

### 2. test_100_run_retrieval_stability  
**Why skipped:** Stress test running retrieval 100 times to verify identical results
**Impact:** Would take significant time to run 100 retrieval operations

**Note:** The conftest.py in CAPABILITY/TESTBENCH/ automatically adds skip markers to any test with "slow" in its keywords:
```python
skip_slow = pytest.mark.skip(reason="Slow test - use --run-slow to run")
for item in items:
    if "slow" in item.keywords:
        item.add_marker(skip_slow)
```

---

## Unconditionally Skipped (@pytest.mark.skip) - 6 tests

These 6 tests are in 2 skipped classes - features that were removed and replaced by cassette network:

**File:** `CAPABILITY/TESTBENCH/skills/test_cortex_toolkit.py`

### Class 1: TestBuildOperation (3 tests skipped)
**Line:** 114
**Reason:** "build operation removed - cassette network handles semantic search"

Skipped tests in this class:
1. `test_build_can_be_invoked`
2. `test_build_with_missing_script_reports_error`
3. `test_build_expected_paths_verification` (implied by class structure)

**Background:** The build operation for cortex toolkit was removed because semantic search is now handled by the cassette network instead of being built from scratch.

### Class 2: TestVerifySystem1Operation (3 tests skipped)
**Line:** 212
**Reason:** "verify_system1 operation removed - cassette network handles semantic search"

Skipped tests in this class:
1. `test_verify_system1_in_operations_registry`
2. `test_verify_system1_handles_missing_db`
3. `test_verify_system1_reports_verification_results` (implied by class structure)

**Background:** The verify_system1 operation was removed because the system1 database is deprecated and FTS is now handled by the cassette network.

---

## Conditionally Skipped (@pytest.mark.skipif) - 5+ tests

These tests are skipped based on platform or available dependencies:

### 1. test_capture_rejects_symlinks
**File:** `CAPABILITY/TESTBENCH/catalytic/test_catalytic_runtime.py:82`
**Condition:** `sys.platform == "win32"`
**Reason:** "Symlinks require admin on Windows"

**Details:** Tests that symlinks in domain raise CatalyticError. Skipped on Windows because creating symlinks requires administrator privileges.

### 2-3. Tests for Deprecated Databases
**File:** `CAPABILITY/TESTBENCH/integration/test_stacked_symbol_resolution.py:35,44`

**Test 2:** `test_system1_symbol_resolution`
**Condition:** `not SYSTEM1_DB.exists()`
**Reason:** "system1.db deprecated - FTS via cassette network"

**Test 3:** `test_canon_index_symbol_resolution`  
**Condition:** `not CANON_INDEX_DB.exists()`
**Reason:** "canon_index.db deprecated - semantic via cassette network"

### 4-6. Tiktoken-Dependent Tests
**File:** `CAPABILITY/TESTBENCH/integration/test_semiotic_compression.py:307,345,374`

**Condition:** `not TIKTOKEN_AVAILABLE`
**Reason:** "tiktoken not installed"

Three tests for semiotic compression that require the tiktoken library for token counting. These are skipped if tiktoken is not installed.

---

## Impact on Push Gate

### What Actually Runs During Push

Out of 1,458 total tests:

| Category | Tests | Runs During Push? |
|----------|-------|-------------------|
| Normal tests | ~1,443 | YES |
| xfail tests | 2 | YES (expected to fail) |
| Slow tests | 2 | NO (skipped by conftest.py) |
| Unconditional skips | 6 | NO (features removed) |
| Conditional skips | ~5 | MAYBE (depends on platform/deps) |

### Effective Test Count

- **Minimum running:** ~1,443 tests (all except slow + unconditional skips)
- **Maximum running:** ~1,448 tests (if all skipif conditions are met)
- **Actually blocking:** 0 test failures (but 1 critic violation blocks everything)

### Critical Point

**The push gate is NOT blocked by test failures.** It's blocked by the critic check which detects a filesystem access pattern violation before any tests run.

---

## Recommendation

To see actual test results:
1. Fix the critic violation in `CAPABILITY/SKILLS/governance/adr-create/run.py` (remove Path.glob() usage)
2. Run: `python CAPABILITY/TOOLS/utilities/ci_local_gate.py --full`
3. This will reveal true pass/fail counts for the ~1,443 tests that actually run

## Content Hash
<!-- CONTENT_HASH: a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6a7b8c9d0e1f2 -->
