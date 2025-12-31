# Bug Report: SKILLS and TOOLS Folders

**Generated:** 2025-12-27
**Scope:** `SKILLS/` and `TOOLS/` directories
**Total Issues Found:** 22

---

## Critical Bugs (4)

### 1. Function Signature Mismatch
- **File:** `SKILLS/mcp-smoke/run.py:139`
- **Issue:** `find_entrypoint()` is defined with 3 parameters including `project_root`, but the parameter is never used in the function body. Compare to `SKILLS/mcp-extension-verify/run.py:50` which correctly defines it with 2 parameters.
- **Impact:** Unused parameter causes confusion; inconsistent API between similar skills.

### 2. Logic Error - Strict Mode Always True
- **File:** `TOOLS/ags.py:552`
- **Issue:** `strict = True if strict else True`
- **Impact:** The `strict` parameter can never be set to False, making the parameter useless.
- **Fix:** Change to `strict = bool(strict)` or remove the line.

### 3. Missing Import Path
- **File:** `TOOLS/critic.py:24`
- **Issue:** `import schema_validator` without proper module path.
- **Impact:** RuntimeError `ImportError` when critic.py tries to call `schema_validator.validate_file()` on line 141.
- **Fix:** Use `from TOOLS import schema_validator` or ensure TOOLS is in sys.path.

### 4. Type Error on None Conversion
- **File:** `TOOLS/cortex.py:183-184`
- **Issue:**
  ```python
  start_line = int(record.get("start_line"))
  end_line = int(record.get("end_line"))
  ```
- **Impact:** `TypeError: int() argument must be a string, a bytes-like object or a number, not 'NoneType'` when record is missing these keys.
- **Fix:** Add defaults: `int(record.get("start_line", 0))`

---

## Significant Bugs (7)

### 5. Bare Except Clause - Silent Failure
- **File:** `SKILLS/swarm-directive/run.py:124`
- **Issue:**
  ```python
  try:
      with open(output_file, 'w') as f:
          json.dump(error_output, f, indent=2)
  except:
      pass
  ```
- **Impact:** Catches and silently suppresses ALL exceptions including `KeyboardInterrupt`, `SystemExit`. Makes debugging impossible.
- **Fix:** Use `except (IOError, OSError) as e:` with logging.

### 6. Bare Except in Token Parsing
- **File:** `SKILLS/pack-validate/run.py:124`
- **Issue:**
  ```python
  try:
      stats["tokens"] = int(line.split(":")[1].strip().replace(",", ""))
  except:
      pass
  ```
- **Impact:** Silent failure on malformed token lines. Users won't know why stats are missing.
- **Fix:** Use `except (IndexError, ValueError):` with warning.

### 7. Incomplete Error Message
- **File:** `SKILLS/pack-validate/run.py:148`
- **Issue:** `"errors": [f"Pack not found: "]` - message ends with colon, no path shown.
- **Impact:** Unhelpful error message for users.
- **Fix:** Include the actual path variable in the message.

### 8. Missing File Existence Check
- **File:** `SKILLS/canon-migration/run.py:91`
- **Issue:** Reads `PACK_INFO.json` without checking if it still exists between detection and application.
- **Impact:** `FileNotFoundError` if file is deleted between checks.
- **Fix:** Add `if pack_info_path.exists():` guard.

### 9. Incomplete Verification Logic
- **File:** `TOOLS/provenance.py:298-349`
- **Issue:** The `verify_provenance()` function has unfinished logic with bare `pass` statements and fragile regex matching.
- **Impact:** Verification may incorrectly report valid provenance as invalid.

### 10. Case-Insensitive Regex on Case-Sensitive Tokens
- **File:** `TOOLS/compress.py:150`
- **Issue:** `re.finditer(pattern, result, re.IGNORECASE)` applied to compression tokens like `@C0`, `@I0`.
- **Impact:** Incorrect compression results; may match unintended strings.
- **Fix:** Remove `re.IGNORECASE` or apply selectively.

### 11. Missing FileNotFoundError Guard
- **File:** `TOOLS/check_canon_governance.py:74`
- **Issue:** Reads CHANGELOG.md without checking existence.
- **Impact:** Crash if CHANGELOG.md is missing.
- **Fix:** Add existence check before read.

---

## Type Annotation Issues (3)

### 12. Lowercase `callable` Instead of `Callable`
- **File:** `SKILLS/canon-migration/run.py:24`
- **Issue:** `MIGRATIONS: Dict[str, callable] = {}`
- **Fix:** `MIGRATIONS: Dict[str, Callable[..., Tuple[Dict[str, Any], List[str]]]] = {}`

### 13. Lowercase `any` Instead of `Any`
- **File:** `TOOLS/catalytic_runtime.py:95`
- **Issue:** `def diff(self, other: "CatalyticSnapshot") -> Dict[str, any]:`
- **Fix:** Change `any` to `Any`

### 14. Lowercase `any` Instead of `Any` (Duplicate Pattern)
- **File:** `TOOLS/catalytic_validator.py:95`
- **Issue:** Same as above.

---

## Moderate Bugs (5)

### 15. No Error Check on Git Stash
- **File:** `TOOLS/emergency.py:137-138`
- **Issue:** `subprocess.run(["git", "stash", ...])` without checking return code.
- **Impact:** If stashing fails, rollback proceeds with uncommitted changes.
- **Fix:** Check `result.returncode` before continuing.

### 16. Connection Not Closed on Error
- **File:** `TOOLS/research_cache.py:60`
- **Issue:** Database connection not closed if exception occurs between open and commit.
- **Impact:** Resource leak.
- **Fix:** Use `with` context manager or try-finally.

### 17. Path Resolution Without Guard
- **File:** `TOOLS/tokenizer_harness.py:86`
- **Issue:** `path.relative_to(PROJECT_ROOT)` without try-catch.
- **Impact:** `ValueError` if file is outside project root.
- **Fix:** Wrap in try-except or check path first.

### 18. JSON Parsing Without Exception Handling
- **File:** `TOOLS/provenance.py:235`
- **Issue:** `json.loads(content)` without handling `JSONDecodeError`.
- **Impact:** Unhandled exception on malformed JSON.
- **Fix:** Add try-except for `json.JSONDecodeError`.

### 19. Redundant Import
- **File:** `TOOLS/codebook_build.py:382`
- **Issue:** `import sys` inside `main()` when already imported at module level (line 22).
- **Impact:** Code quality issue, not a runtime bug.

---

## Indentation Inconsistencies (2 files)

### 20. Extra Space Indentation
- **File:** `SKILLS/llm-packer-smoke/run.py`
- **Lines:** 105-106, 248-249, 255-262, 264-266, 268-269
- **Issue:** One extra space of indentation compared to expected level.

### 21. Extra Space Indentation
- **File:** `SKILLS/pack-validate/run.py`
- **Lines:** 50-52, 79-84, 92, 94
- **Issue:** Same inconsistent indentation pattern.

---

## Minor Issues (1)

### 22. Empty Regex Pattern Possible
- **File:** `TOOLS/lint_tokens.py:68`
- **Issue:** If `terms` list is empty, creates invalid regex `\b()\b`.
- **Fix:** Add guard: `if not terms: return []`

---

## Summary by Severity

| Severity | Count |
|----------|-------|
| Critical | 4 |
| Significant | 7 |
| Type Annotations | 3 |
| Moderate | 5 |
| Indentation | 2 |
| Minor | 1 |
| **Total** | **22** |

---

## Recommended Priority

1. Fix critical bugs first (ags.py, critic.py, cortex.py)
2. Replace bare except clauses with specific exceptions
3. Add missing file existence checks
4. Fix type annotations for better IDE support
5. Clean up indentation for consistency
