---
name: "Dead Code & Brittle Test Audit"
description: "Audit of dead code and assertion-free test files across CAPABILITY/, THOUGHT/LAB/, and cross-cutting structure."
generated: "2026-06-09T17:16:00"
source: "hermes-harness prompt_audit fixture"
agent: "hermes-harness"
canonical: true
---

<!-- CONTENT_HASH: 2E731D7DF013DCE669AA5427C25DF00695ECC2BBC521A7EDC02BEF47C7858CBB -->

# Dead Code & Brittle Test Audit Report
**Repository:** agent-governance-system  
**Path:** D:\CCC 2.0\AI\agent-governance-system  
**Date:** 2026-06-09  
**Mode:** Read-only audit  
**Python files:** ~425 (source + tests)  

---

## Executive Summary

Three layers audited:
1. **Core system** (CAPABILITY/) — 9 source files, 22 dead code findings
2. **Research lab** (THOUGHT/LAB/CAT_CAS/) — 425 Python scripts, widespread fragility
3. **Tests** — 181 test/verify files, **massive assertion deficit** (zero-assert epidemic)

**Headline:** The repo has a fundamental testing crisis. 181 of 200+ test-like files (90%) contain zero assertions. They are print-based verification scripts masquerading as tests. Only one real test file (hermes-harness/test_contracts.py) uses proper pytest assertions.

---

## PRIORITY 1 (CRITICAL): Assertion-Free "Tests"

### Finding: 181 files named test_*/verify_* have no assertions

**Evidence:**  
- FORMULA/v2_2: 119 verify_*.py files — 0 assertions total. All use `print()` and `sys.exit()` to signal pass/fail. No test framework integration.
- CAT_CAS: 62 test/verify files with zero assertions. Only 9 files have assertions (1-4 each).
- q38_noether/tests/test_geodesic_battery.py — 335 lines, 0 assertions. Named "test" but pure print-based script.  
- test_geodesic_proof.py, test_lie_detection.py — same pattern.

**What this means:**  
- No CI/CD can automatically verify correctness. Every "test" requires human inspection of stdout.
- `pytest` finds these files but they register as passing (return code 0 regardless of actual correctness).
- A regression that silently breaks results will not be caught.

**Representative sample (verify_q1.py):**
```
if E_cv < 0.5 and abs(E_mean - E_cal) < max(0.01, 2 * E_std):
    print("Q1 VERIFIED: ...")     # NOT an assertion
elif E_cv < 1.0:
    print("Q1 SUPPORTED: ...")    # NOT an assertion
else:
    print("Q1 NOT SUPPORTED: ...") # NOT an assertion
return 0  # Always exits 0 — pytest sees "passed" every time
```

**Severity:** CRITICAL  
**Files affected:** ~181 across FORMULA/v2_2, CAT_CAS, MODEL_TESTS, DEPRECATED  
**Next action:**  
- Decide which verify scripts should become real tests (add `assert` on thresholds)  
- Move research verification scripts to a `verification/` directory, exclude from pytest collection via `norecursedirs`  
- Write a single `test_all_assertions.py` that runs the verify scripts and asserts on exit code + output patterns

---

## PRIORITY 2 (HIGH): Core System Dead Code

### 2a. server.py — 26 unused imports (lines 33-62)

**File:** CAPABILITY/MCP/server.py  
**Evidence:** All 14 names from `.primitives` and 12 names from `.validation` are imported but never read. Importing them adds startup overhead and creates false dependency signals.  
**Examples:** `lock_file`, `atomic_write_jsonl`, `is_path_under_root`, `DURABLE_ROOTS`, `CONTRACTS_DIR`  
**Severity:** HIGH  
**Next action:** Remove unused imports. If they exist for documentation purposes, comment them out and note why.

### 2b. server.py — Dead method + broken test code (lines 1416, 1598)

- `_tool_not_implemented()` — never registered in `tool_handlers`, never called.  
- Test code at lines 1598-1614 calls `"cortex_query"` tool, but no handler exists — always hits the "Unknown tool" error branch.  
**Severity:** HIGH (method) / MEDIUM (broken test)  
**Next action:** Remove dead method. Fix or remove broken test code.

### 2c. ags.py — Dead parameter in ags_run() (line 740)

`strict = True if strict else True` — always True, never read after assignment. `--strict` is hardcoded elsewhere.  
**Severity:** MEDIUM  
**Next action:** Remove the parameter or actually wire it to control `--strict`.

### 2d. critic.py — 3 dead globals, 1 dead variable (lines 60-67, 253-256)

- `CHANGELOG_PATH`, `ALLOWED_OUTPUT_ROOTS`, `bare_excepts_patterns` — all assigned but never read.  
- Suggests incomplete governance checks.  
**Severity:** MEDIUM  
**Next action:** Complete the bare-excepts check or remove the dead code.

### 2e. catalytic-wormhole/run.py — 2 dead data structures (lines 122, 201)

- `rank_k` — computed but never referenced; `K` used directly everywhere.  
- `optimal = {}` — populated at line 224 but never read after the loop.  
**Severity:** MEDIUM  
**Next action:** Remove dead variables; if `optimal` was meant for output, add it to the return value.

---

## PRIORITY 3 (MEDIUM): Research Lab Fragility

### 3a. catalytic_ffi.pyd — 14 files import a compiled binary

**Evidence:** `catalytic_ffi.pyd` is a compiled Rust binary (Windows-only, Python 3.11-specific). Imported by 14 files in THOUGHT/LAB/CAT_CAS/.  
**Affected:** `14_bekenstein_violator/`, `16_catalytic_27b_inference/`, `19_catalytic_computronium/` experiments.  
**Risk:** Python version upgrade or non-Windows execution makes these files unimportable.  
**Severity:** HIGH  
**Next action:** Provide a pure-Python fallback or document Python version lock.

### 3b. holo_core — 27+ files import from a DEPRECATED directory

**Evidence:** Files use `sys.path.insert(0, ...)` to reach `THOUGHT/DEPRECATED/TINY_COMPRESS/holographic-image/holo_core.py`.  
**Risk:** If DEPRECATED directory is cleaned up, all 27+ files break at import.  
**Severity:** MEDIUM  
**Next action:** Promote holo_core out of DEPRECATED or migrate importers.

### 3c. catalytic_tape.py — 3 divergent copies

- Root `catalytic_tape.py` and `45_phase_math/catalytic_tape.py` are identical (class `CatalyticTape`, 256 MB, seed 42).  
- `47_phase_atom/catalytic_tape.py` is divergent (class `BennettHistoryTape`, 10 MB, seed 47).  
**Severity:** MEDIUM  
**Next action:** Consolidate identical copies. The divergent copy is intentional but should be noted as such.

### 3d. Hardcoded absolute paths (D:\CCC 2.0)

**Evidence:** Multiple DEPRECATED and experimental test files contain hardcoded `D:\CCC 2.0\AI\agent-governance-system` paths using `sys.path.insert` and `os.chdir`.  
**Affected:** `DEPRECATED/TINY_COMPRESS/llm-spectral/auto_feedback/test_pretrained.py`, `DEPRECATED/CAT_CHAT/tests/test_bundle_execution.py`, `DEPRECATED/LIL_Q/test_sandbox/_test_retrieve.py`, `CAT_CAS/16_catalytic_27b_inference/_test_phase2.py`.  
**Severity:** MEDIUM (files are already in DEPRECATED or experimental)  
**Next action:** Convert to relative paths using `Path(__file__).parent` pattern (already used correctly by hermes-harness tests).

### 3e. Linux-only scripts in Windows repo

**Evidence:** `44_phase_ssh_linux/session_scripts/phase1_msr/` — 15 files access `/dev/cpu/*/msr` (Linux MSR interface). Will fail with `FileNotFoundError` on Windows.  
**Severity:** MEDIUM  
**Next action:** Document as Linux-only; add `sys.platform` guard with early exit.

### 3f. No-import-guard scripts (execute at import time)

**Evidence:** 4 files have no `if __name__ == '__main__'` guard: `launch_server.py`, `baseline.py`, `sub_p4.py`, `classify_ef.py`. Importing them triggers side effects (subprocesses, MSR reads, file I/O).  
**Severity:** LOW (experimental scripts, not imported by others)  
**Next action:** Add `__name__` guard.

---

## PRIORITY 4 (LOW): Minor Dead Code

| File | Finding | Lines |
|------|---------|-------|
| cas.py | Unused `import os` | 2 |
| cas.py | Unused `from typing import Union` | 5 |
| server.py | Unused `import hashlib` | 15 |
| server.py | Unused module-level `import tempfile, time, uuid, datetime` (re-imported locally) | 21-24 |
| server.py | Unused `Callable, Iterator` from typing | 26 |
| wormhole/run.py | Unused `import math, hashlib, Optional` | 17, 22 |
| wormhole/run.py | Unused `OUTPUT_DIR` constant | 28 |
| run.py | Unused `_in_capability` variable | 24, 30 |
| server.py | `commit_ceremony` prompt listed but unimplemented | 738 |
| critic.py | Duplicate comment lines 251-252 | 251-252 |
| ags.py | Unreachable `sys.stderr.write("ERROR: unsupported command")` | 933-934 |

---

## The One Good Test File

`CAPABILITY/SKILLS/agents/hermes-harness/tests/test_contracts.py` — 333 lines, 36 test functions, ALL have assertions. Uses tempfile for isolation, monkeypatch for external dependencies. This is the model to follow.

---

## Output Contract

| What this audit delivers | What it does not deliver |
|---|---|
| 22 dead code findings with line numbers and evidence | Test execution results (read-only audit) |
| 181 assertion-free test files identified | Coverage percentages |
| Hardcoded path inventory | Fix PRs or code changes |
| Dependency risk assessment (catalytic_ffi, holo_core) | Runtime profiling or performance data |
| Structural analysis across 3 code layers | Security vulnerability assessment |

**Next recommended actions (in order):**
1. Convert top-20 verify scripts to real pytest tests with `assert` (PRIORITY 1)
2. Strip 26 unused imports from server.py (PRIORITY 2a)
3. Remove `_tool_not_implemented` and fix `cortex_query` test (PRIORITY 2b)
4. Consolidate identical catalytic_tape.py copies (PRIORITY 3c)
5. Clean up minor dead code (PRIORITY 4)
