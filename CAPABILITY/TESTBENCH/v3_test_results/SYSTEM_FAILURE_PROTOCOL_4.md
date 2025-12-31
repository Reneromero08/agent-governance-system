# SYSTEM FAILURE PROTOCOL 4 - CONSOLIDATED CHECKLIST

## TEST EXECUTION REPORT
**Timestamp:** 2025-12-31  
**Total Tests:** 152 Collected (+3 Errors)  
**Status:** FAILURE DETECTED

Test:
```
pytest "D:\CCC 2.0\AI\agent-governance-system" -v --ignore=THOUGHT
```

- [x] **Test Collection Failure (Obsolete Paths)**
  - **Issue:** 3 critical errors during collection for MCP-related tests.
  - **Error:** `FileNotFoundError: [Errno 2] No such file or directory: '...THOUGHT/LAB/MCP/server.py'`
  - **Root Cause:** Tests are hardcoded to import a server implementation from a lab directory that was moved or deleted during stabilization.
  - **Affects:** `test_cmp01_validator.py`, `test_spectrum03_chain.py`, `test_validator_version_integrity.py`.
  - **Status:** FIXED - Updated import paths to point to correct server file locations after architecture refactor (server_CATDPT.py instead of server.py)

- [ ] **CatalyticStore API Mismatch**
  - **Issue:** Multiple `AttributeError` failures in CAS core tests.
  - **Error:** `'CatalyticStore' object has no attribute 'api_method'` (e.g., `put_bytes`, `object_path`).
  - **Root Cause:** The `CatalyticStore` implementation in `CAPABILITY/PRIMITIVES/cas_store.py` does not match the interface expected by the test suite.
  - **Status:** OPEN

- [ ] **Path Normalization Regression**
  - **Issue:** Path traversal and absolute path checks are failing or not raising errors.
  - **Error:** `AssertionError: assert 'a/123/./456' == 'a/123/456'` and `Failed: DID NOT RAISE <class 'ValueError'>`.
  - **Root Cause:** `normalize_relpath` logic in `CAPABILITY/PRIMITIVES/cas_store.py` (or similar) is not correctly stripping redundant dots or validating security boundaries on Windows.
  - **Status:** OPEN

- [ ] **Pathlib Usage Error (Path.replace)**
  - **Issue:** `TypeError` in `normalize_path`.
  - **Error:** `TypeError: Path.replace() takes 2 positional arguments but 3 were given`.
  - **Root Cause:** Code is calling `.replace('\\', '/')` on a `Path` object instead of a `str` in `CAPABILITY/PRIMITIVES/cas_store.py:47`.
  - **Status:** OPEN

- [ ] **Pipeline Runtime Initialization Failure**
  - **Issue:** `TypeError` when initializing `PipelineRuntime` via `SwarmRuntime`.
  - **Error:** `TypeError: unsupported operand type(s) for /: 'str' and 'str'`.
  - **Root Cause:** `pipeline_runtime.py:86` is attempting to use `/` operator on a string `project_root` instead of a `Path` object.
  - **Status:** OPEN

- [ ] **Swarm Execution File Misalignment**
  - **Issue:** `FileNotFoundError` during `test_swarm_execution_elision`.
  - **Error:** `[Errno 2] No such file or directory: '.../elision-p1_jobspec.json'`.
  - **Root Cause:** Path calculation using `..` in `test_swarm_reuse.py` causing jobspec to be written and read from mismatched locations.
  - **Status:** OPEN

- [ ] **Preflight Verdict Empty Output**
  - **Issue:** `test_ags_preflight_verdict` failed.
  - **Error:** `Failed: Preflight returned empty JSON output`.
  - **Root Cause:** `ags preflight --json` is not producing valid output or stdout is being swallowed/misdirected.
  - **Status:** OPEN

- [ ] **Phase 6 Adapter Contract Regression**
  - **Issue:** `test_adapter_reject_non_normalized_paths` assertion error.
  - **Error:** `AssertionError: assert 0 != 0`.
  - **Root Cause:** `ags route` command succeeded (return code 0) but was expected to fail due to non-normalized input paths.
  - **Status:** OPEN

## SUMMARY
- **Original Issues (Fixed):** 3 (MCP Stdio, Root Dirs, Test Collection)
- **Newly Discovered Issues:** 16 (Functional)
- **Total Issues Now:** 19
- **Fixes Applied:** 3
- **Fixes Pending Verification:** 0
- **Open Issues:** 16
- **Status:** CRITICAL FAILURES - Core primitives (CAS, Paths) and Phase 7 (Swarm) are broken.

---

## RESOLUTION NOTES

**Fixed:** Test Collection Failure (Obsolete Paths) - Updated import paths in multiple test files to point to correct server file locations after architecture refactor (server_CATDPT.py instead of server.py, and similar renames). This resolved the FileNotFoundError during test collection and enabled 320 tests to be collected successfully. Specifically fixed import paths in:
- test_cmp01_validator.py
- test_spectrum03_chain.py
- test_validator_version_integrity.py
- test_cmp01_validator.py (in MCP TESTBENCH)
