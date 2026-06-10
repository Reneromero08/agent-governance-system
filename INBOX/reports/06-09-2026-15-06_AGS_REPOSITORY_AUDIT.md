---
name: "AGS Repository Audit: Dead Code & Brittle Tests"
description: "Comprehensive read-only audit of CAPABILITY/, THOUGHT/LAB/CAT_CAS/, and cross-cutting structure with 19 critical/high/medium findings."
generated: "2026-06-09T15:06:00"
source: "hermes-harness prompt_worker fixture"
agent: "hermes-harness"
canonical: true
---

<!-- CONTENT_HASH: 879DA70C626FE5B12C64356876FC0E1ACA8CEE7462B7D4C26071BDBBFE939ECA -->

# AGS Repository Audit: Dead Code & Brittle Tests

**Target:** `D:\CCC 2.0\AI\agent-governance-system`
**Mode:** Read-only
**Date:** 2026-06-09
**Scope:** All Python source (.py), test files, structural integrity
**Total files audited:** ~58,775 (5,701 .py, 17,384 .json, 18,014 .md)

Three parallel audits covered CAPABILITY/, THOUGHT/LAB/CAT_CAS/, and cross-cutting structure. Findings are merged and deduplicated below.

---

## CRITICAL (fix immediately)

### 1. Bare `except:` clauses still present (2 instances)
**File:** `CAPABILITY/SKILLS/agents/catalytic-wormhole/run.py`
**Lines:** 138, 182
**Pattern:** `except: pass` — silently swallows `ValueError` and `IndexError` when parsing safetensors key strings (`int(parts[i+1])`). These also swallow `KeyboardInterrupt` and `SystemExit`.
**Fix:** Replace with `except (ValueError, IndexError): pass`.
**Severity rationale:** CHANGELOG from >2 years ago claimed "67+ bare except: clauses narrowed to specific exception types." These 2 were missed and remain silent-fail risk in a core skill.

### 2. Orphaned test files — hundreds of tests never collected by pytest
**Cause:** `pytest.ini` sets `testpaths = CAPABILITY/TESTBENCH` — a single narrow entry.
**Impacted:** 1,300+ test files outside that path, including:
- `CAPABILITY/SKILLS/agents/hermes-harness/tests/test_contracts.py` (34 tests, full fixture suite)
- `CAPABILITY/PRIMITIVES/tests/test_alignment_key.py` (7 tests)
- `CAPABILITY/PRIMITIVES/tests/test_vector_communication.py` (5 tests)
- `CAPABILITY/SKILLS/inbox/inbox-report-writer/test_inbox_hash.py`
- Various skill tests under `CAPABILITY/SKILLS/*/tests/`

All are structured pytest modules with proper imports and fixtures. They simply aren't reachable.
**Fix:** Either expand testpaths to include these directories, or add them to a test runner script.

### 3. Test imports from DEPRECATED path (will break if pruned)
**File:** `CAPABILITY/TESTBENCH/core/test_cmp01_validator.py` (line 21)
**Pattern:** Uses `importlib.util.spec_from_file_location` to load from:
```python
SERVER_PATH = REPO_ROOT / "THOUGHT" / "DEPRECATED" / "MCP_EXPERIMENTAL" / "server_CATDPT.py"
```
**Risk:** If the DEPRECATED directory is ever removed, the entire 682-line test file breaks silently. The import can't be traced by static analysis.

---

## HIGH (fix in next iteration)

### 4. Hardcoded absolute paths — machine-specific, non-portable
**30+ occurrences across 10+ files**, all pointing to Rene's machine:
```
d:\CCC 2.0\AI\agent-governance-system\...
D:\CCC 2.0\AI\agent-governance-system\...
```
**Specific locations:**
- `CAPABILITY/TOOLS/utilities/verify_f3.py` line 8: `ROOT = Path("d:/CCC 2.0/AI/agent-governance-system")`
- `CAPABILITY/TESTBENCH/platform_compat/test_wsl_compat.py`: hardcodes `D:\\CCC 2.0\\AI\\repo` in assertion strings
- `THOUGHT/LAB/CAT_CAS/25_lattice_holography/*.py` (6+ files): hardcoded MODEL_DIR and HOLO_PATH
- `THOUGHT/LAB/CAT_CAS/34_zeta_eigenbasis/tests/test_random_harmonic_sieve.py` and 9 others: absolute repo paths

**Fix:** Replace with `Path(__file__).resolve().parents[N]` anchoring or `REPO_ROOT` discovery.

### 5. Dead code files — never imported, never called
| File | Status |
|------|--------|
| `CAPABILITY/TOOLS/critic.py` | Defines `check_search_protocol()` + `CONCEPTUAL_INDICATORS`. Zero imports found across entire repo. The live critic is `CAPABILITY/TOOLS/governance/critic.py`. |
| `CAPABILITY/TOOLS/utilities/terminal_hunter.py` | Standalone utility, never imported. Also has bug: `PROJECT_ROOT = Path(__file__).resolve().parents[1]` resolves to CAPABILITY/TOOLS/, not repo root. References nonexistent `CATALYTIC-DPT/LAB/ARCHIVE` directory. |
| `CAPABILITY/TOOLS/utilities/verify_f3.py` | Standalone F3 CAS verification, never imported. Contains hardcoded absolute path. |

### 6. Duplicate `catalytic_tape.py` — naming collision
Three files with same name but different content:
- `THOUGHT/LAB/CAT_CAS/catalytic_tape.py` — class `CatalyticTape` (identical to 45_phase_math copy)
- `THOUGHT/LAB/CAT_CAS/45_phase_math/catalytic_tape.py` — class `CatalyticTape` (identical to root copy)
- `THOUGHT/LAB/CAT_CAS/47_phase_atom/catalytic_tape.py` — class `BennettHistoryTape` (different content, naming collision)

The root copy is unreferenced dead weight. Import order ambiguity means `from catalytic_tape import ...` could silently resolve to the wrong copy depending on sys.path.

### 7. Tests without assertions — pseudo-tests that always pass
**Confirmed across all three workers:**

**In CAPABILITY/:**
- `CAPABILITY/PRIMITIVES/tests/test_vector_communication.py::test_anchor_set_comparison()` — prints results, no `assert` statement. Always prints "PASSED."
- `CAPABILITY/PRIMITIVES/tests/test_vector_communication.py::test_cross_model_communication()` — prints accuracy, line 122 has no `assert`.
- `CAPABILITY/PRIMITIVES/tests/test_alignment_key.py::test_aligned_pair_with_mock()` — prints statuses, no correctness assertion.

**In THOUGHT/LAB/CAT_CAS/:** 34 of 35 `*test*.py` files contain zero `assert` statements. They are "run and print" scripts masquerading as tests:
- `07_quantum_simulator/catalytic_shor_test.py` — 317 lines, prints, no assertions
- `12_structured_tape_acceleration/verify_integrity.py` — 263 lines, prints, no assertions
- `46_phase_bio/*/tests/test_protein*.py` (6 files) — print gap metrics, no assertions

**Only real tests in CAT_CAS:** `47_phase_atom/tests/test_bennett_history_tape.py` (17 assertions) and `ORACLE/visualizer/tests/smoke.py` (~90 test functions with assertions).

---

## MEDIUM (address this cycle)

### 8. Duplicated core implementations
**SHA-256 hashing (6 implementations):**
| File | Function |
|------|----------|
| `CAPABILITY/PRIMITIVES/cas_store.py` | `sha256_file(path)` |
| `CAPABILITY/PRIMITIVES/canonical_json.py` | `sha256_hex(data)` |
| `CAPABILITY/MCP/primitives.py` | `compute_hash(file_path)` |
| `CAPABILITY/TOOLS/catalytic/provenance.py` | `hash_file(filepath)` |
| `CAPABILITY/SKILLS/agents/ant-worker/scripts/run.py` | `compute_hash(file_path)` |
| `CAPABILITY/SKILLS/utilities/doc-merge-batch-skill/...` | `sha256_bytes(data)` |

At least 4 are byte-identical (`hashlib.sha256(data).hexdigest()`). Consolidate into `CAPABILITY/PRIMITIVES/`.

**Path normalization (5 implementations):**
| File | Function |
|------|----------|
| `CAPABILITY/PRIMITIVES/cas_store.py` | `normalize_path(rel)` |
| `CAPABILITY/PRIMITIVES/wsl_compat.py` | `normalize_path_for_platform(path)` |
| `CAPABILITY/PRIMITIVES/repo_digest.py` | `normalize_path(path, repo_root)` |
| `CAPABILITY/PRIMITIVES/path_utils.py` | `validate_path_in_root(path, root)` |

Different normalization rules, no shared base.

### 9. Hardware-dependent tests with no skip markers
- `THOUGHT/LAB/CAT_CAS/16_catalytic_27b_inference/_test_cavity.py` — requires CUDA (`device_map='cuda'`), no try/except or `@pytest.mark.skipif`
- `THOUGHT/LAB/CAT_CAS/16_catalytic_27b_inference/_test_cavity_full.py` — same, also `.to('cuda')` call
- `THOUGHT/LAB/CAT_CAS/44_phase_ssh_linux/session_scripts/phase2_kuramoto/kuramoto_test.py` — requires `/dev/cpu/*/msr` (Linux AMD CPU MSR registers), no skip
- `THOUGHT/LAB/CAT_CAS/44_phase_ssh_linux/session_scripts/phase1_msr/*.py` — import `msr` modules requiring root and specific hardware

### 10. Missing `__init__.py` in production packages
**26 directories** containing .py files lack `__init__.py`:
- `CAPABILITY/PIPELINES/` — production package, should be importable
- `CAPABILITY/CAS/` — production package
- `CAPABILITY/TESTBENCH/` and 17 subdirectories (core/, integration/, pipeline/, etc.)
- `NAVIGATION/CORTEX/`

While namespace packages work in Python 3, explicit `__init__.py` prevents import ambiguity.

### 11. CAS test duplication
- `CAPABILITY/TESTBENCH/cas/test_cas.py::test_identical_input_same_hash` and `CAPABILITY/TESTBENCH/cas/test_cas_dedup.py::test_cas_dedup_same_hash` test the exact same property (roundtrip + identical-input-same-hash).
- `CAPABILITY/TESTBENCH/artifacts/test_artifact_store.py` and `test_artifact_dedup.py` similarly overlap on dedup testing.

### 12. Unused imports in core files
- `CAPABILITY/TOOLS/ags.py` line 4: `import shutil` — never used
- `CAPABILITY/TOOLS/prune_memory.py` line 14: `from typing import Tuple` — never used

### 13. `.pytest_cache/` at repo root
Not gitignored. Should be added to `.gitignore` or cleaned.

### 14. CHANGELOG references to 50+ deleted/renamed files
References such as `LAW/CANON/INBOX_POLICY.md`, `LAW/CANON/SYSTEM_BUCKETS.md`, `CAPABILITY/TOOLS/governance/inbox_normalize.py` point to files that no longer exist. These are stale documentation, not functional bugs, but they mislead anyone reading the changelog for context.

---

## LOW (address when convenient)

### 15. 20 files named `experiment.py`
Across `THOUGHT/LAB/CAT_CAS/` experiments 01 through 44. Not true byte-identical duplicates, but the generic naming makes dependency tracing impossible without reading every file. Code search for `import experiment` or `from experiment import` is ambiguous.

### 16. 12 experiment directories with zero documentation
No README.md, no SKILL.md, no docstrings explaining purpose:
`09_borrowing_os_memory`, `15_hdd_native_inference`, `25_lattice_holography` (11 files, no docs), `25_wigners_friend`, `25a_lattice_holography`, `26_hawking_quantum`, `26_optical_3sat`, `27_landauer_limit`, `28_stealth_crypto`, `29_graph_reachability`, `30_boundary_stress`, `31_graph_isomorphism`

### 17. Python version mismatch in stale `.pyc` files
`44_phase_ssh_linux/session_scripts/phase5_8/__pycache__/` contains `.pyc` files compiled with Python 3.8 while the current venv uses 3.11. Not harmful (Python ignores version-mismatched .pyc), but clutter.

### 18. Project root resolution bug
`CAPABILITY/TOOLS/utilities/terminal_hunter.py`: `PROJECT_ROOT = Path(__file__).resolve().parents[1]` resolves to `CAPABILITY/TOOLS/` (1 level up from utilities/), not the actual repo root (3 levels up).

### 19. Redundant bug in `ags.py`
`ags.py` line 21 has a commented-out `sys.path.insert(0, str(REPO_ROOT / "CATALYTIC-DPT"))` — references a directory structure that CHANGELOG confirms was migrated away from.

---

## REPO COMPOSITION (for context)

| Type | Count | % |
|------|-------|---|
| .md | 18,014 | 30.7% |
| .json | 17,384 | 29.6% |
| .py | 5,701 | 9.7% |
| .c/.h | 3,401 | 5.8% |
| Other (.txt, .bin, .o, .db, .pdf, .csv, .js) | 14,275 | 24.3% |

The repo is heavily documentation-and-data driven. The C/C surface is concentrated in `THOUGHT/LAB/CAT_CAS/44_phase_ssh_linux/` (coreboot dump) and `THOUGHT/DEPRECATED/EIGEN_ALIGNMENT/qgt_lib/`.

---

## OUTPUT CONTRACT

This audit was performed read-only via three parallel subagent workers:
1. CAPABILITY/ deep inspection (20 API calls, all .py source files under core governance)
2. THOUGHT/LAB/CAT_CAS/ deep inspection (23 API calls, all .py source files under experimental lab)
3. Cross-cutting structural scan (19 API calls, full-tree duplicate/corruption/orphan/testpath analysis)

Findings were deduplicated across workers. All reported locations are verified against actual file reads. The 50+ CHANGELOG dead references and 1300+ orphaned test file count come from automated `search_files` and `terminal` scans. The bare `except:` count in CAPABILITY/ was confirmed by grep for `except:\s*(#|pass)`.

No files were modified. No tests were executed. No network calls were made.
