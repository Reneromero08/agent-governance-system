# Cortex-Toolkit Operations Investigation Report

**Date:** 2026-01-25
**Investigator:** Claude Opus 4.5
**Status:** INVESTIGATION COMPLETE - NO ACTION REQUIRED

---

## Executive Summary

The `op_build` and `op_verify_system1` functions exist in the cortex-toolkit `run.py` but are intentionally excluded from the `OPERATIONS` registry. This is **by design** following a deliberate architectural decision during the Cassette Network migration (Phase 2.4, January 2026).

**Verdict:** The current state is CORRECT. The dead code should eventually be cleaned up, but there is no bug or governance violation.

---

## Investigation Findings

### 1. The Functions Exist But Are Not Registered

**Location:** `D:\CCC 2.0\AI\agent-governance-system\CAPABILITY\SKILLS\cortex-toolkit\run.py`

The file contains:
- `op_build()` function (lines 100-166) - Rebuilds CORTEX index
- `op_verify_system1()` function (lines 269-327) - Verifies system1.db integrity

But the `OPERATIONS` registry (line 569-573) only includes:
```python
OPERATIONS = {
    "verify_cas": op_verify_cas,
    "summarize": op_summarize,
    "smoke_test": op_smoke_test,
}
```

### 2. Root Cause: Deliberate Deprecation in Phase 2.4

**Commit:** `8450ab40d46e70d9e2d9a25b33b36b47cd83fd7b`
**Date:** 2026-01-11
**Author:** Raul R Romero with Claude Sonnet 4.5

The commit message explicitly states:
```
feat(cassette-network): Phase 2.4 cleanup - deprecate system1.db infrastructure

Completed comprehensive cleanup of deprecated CORTEX code after successful
migration to cassette network. All 1,242 AGS tests pass.

REMOVED (17,688 lines):
- NAVIGATION/CORTEX/db/ - system1_builder, adr_index.db, canon_index.db,
  skill_index.db, codebase_full.db, system2.db, schemas
...

UPDATED (473 lines):
...
- cortex-toolkit skill - Updated for cassette network
```

The diff shows the operations were intentionally removed from the registry:
```diff
-OPERATIONS = {
-    "build": op_build,
-    "verify_cas": op_verify_cas,
-    "verify_system1": op_verify_system1,
-    "summarize": op_summarize,
-    "smoke_test": op_smoke_test,
-}
+OPERATIONS = {
+    "verify_cas": op_verify_cas,
+    "summarize": op_summarize,
+    "smoke_test": op_smoke_test,
+}
```

### 3. Why the Code Was Left Behind

The function bodies were intentionally left in place but the operations were removed from the registry. This pattern suggests:

1. **Non-destructive deprecation** - The code remains for reference but is no longer callable
2. **Migration safety** - If issues arose with cassette network, the code could be quickly re-enabled
3. **Incomplete cleanup** - The function bodies should ideally have been removed entirely

### 4. The Cassette Network Migration

According to `NAVIGATION/CORTEX/network/CHANGELOG.md` (Phase 2.4, version 3.8.2):

```markdown
### Phase 2.4: Cleanup and Deprecation - COMPLETE

**Goal:** Remove deprecated system1.db infrastructure now that cassette network is operational

#### Removed
- **NAVIGATION/CORTEX/db/** - Deprecated database files
  - `system1_builder.py`, `cortex.build.py`
  - `adr_index.db`, `canon_index.db`, `skill_index.db`, `codebase_full.db`
  - `system1.db`, `system2.db`
...
```

The cassette network (9 cassettes in `NAVIGATION/CORTEX/cassettes/`) replaced the monolithic `system1.db`:
- `canon.db`, `governance.db`, `capability.db`, `navigation.db`
- `direction.db`, `thought.db`, `memory.db`, `inbox.db`, `resident.db`

### 5. ADR Status

**No formal ADR exists** for this specific deprecation. The decision is documented in:
- CHANGELOG.md (version 3.8.2, lines 693-753)
- CASSETTE_NETWORK_ROADMAP.md (Phase 2.4 section)
- The commit message itself

This represents a **governance gap** - significant deprecations should have formal ADRs.

### 6. Test Coverage

The test file `CAPABILITY/TESTBENCH/skills/test_cortex_toolkit.py` handles this correctly:

```python
class TestOperationDispatch:
    def test_operations_registry_contains_all_operations(self):
        # Note: build and verify_system1 removed - cassette network handles semantic search
        expected_ops = ["verify_cas", "summarize", "smoke_test"]
        for op in expected_ops:
            assert op in cortex_toolkit.OPERATIONS

@pytest.mark.skip(reason="build operation removed - cassette network handles semantic search")
class TestBuildOperation:
    ...

@pytest.mark.skip(reason="verify_system1 operation removed - cassette network handles semantic search")
class TestVerifySystem1Operation:
    ...
```

### 7. Deprecated Skills Archive

The original standalone skills were properly archived:
- `MEMORY/ARCHIVE/skills-deprecated/cortex-build/SKILL.md`
- `MEMORY/ARCHIVE/skills-deprecated/system1-verify/SKILL.md`

Both contain deprecation notices pointing to cortex-toolkit.

---

## Timeline of Events

| Date | Event |
|------|-------|
| 2026-01-07 | Skills consolidated: 18 skills merged into 4 toolkits (commit `1ac43491`) |
| 2026-01-07 | cortex-toolkit created with all 5 operations: build, verify_cas, verify_system1, summarize, smoke_test |
| 2026-01-11 | Phase 2.4 cleanup: build and verify_system1 removed from registry (commit `8450ab40`) |
| 2026-01-11 | system1.db and related infrastructure deleted (17,688 lines removed) |
| 2026-01-18 | Cassette network declared complete (all success metrics passing) |
| 2026-01-25 | Investigation performed |

---

## Assessment

### Is This a Bug?

**NO.** The operations were intentionally disabled as part of a deliberate migration.

### Should the Dead Code Be Removed?

**YES, eventually.** The function bodies for `op_build` and `op_verify_system1` serve no purpose now that:
1. `system1.db` no longer exists
2. The build script (`NAVIGATION/CORTEX/db/cortex.build.py`) was deleted
3. The cassette network is the canonical semantic search infrastructure

However, this is a **low-priority cleanup** task, not a bug fix.

### Should There Be an ADR?

**YES.** A formal ADR documenting the system1.db deprecation would improve governance. This could be:
- ADR: "Deprecate system1.db in favor of Cassette Network"
- Status: Accepted
- Decision: system1.db replaced by 9 bucket-aligned cassettes
- Context: Cassette network provides better partitioning, receipts, determinism

---

## Recommendations

### Option A: Do Nothing (RECOMMENDED)

The current state is stable:
- Tests pass (1,242/1,242 AGS tests)
- The dead code causes no harm
- The OPERATIONS registry correctly reflects available functionality
- Documentation exists in CHANGELOG and SKILL.md files

### Option B: Clean Up Dead Code (OPTIONAL)

If cleanup is desired:
1. Remove `op_build()` function (lines 79-166)
2. Remove `op_verify_system1()` function (lines 241-327)
3. Remove unused imports and constants (`DEFAULT_BUILD_SCRIPT`, `DEFAULT_SECTION_INDEX`, `DB_PATH`)
4. Update docstring to remove references to removed operations

This would reduce the file by ~130 lines.

### Option C: Create Formal ADR (GOOD GOVERNANCE)

Create `THOUGHT/LEDGER/ADR/2026/XXX-deprecate-system1-db.md` documenting:
- The decision to deprecate system1.db
- Rationale (cassette network benefits)
- Migration path
- What was removed
- Impact on skills

---

## Conclusion

The `op_build` and `op_verify_system1` functions are **intentionally disabled orphan code** from the Phase 2.4 cassette network migration. This is not a bug - it's the expected state after a major architectural transition.

The proper governance action is:
1. **Acknowledge** the current state is correct
2. **Optionally** clean up the dead code in a future maintenance pass
3. **Consider** creating a formal ADR for historical documentation

No urgent action is required.

---

## References

- **Commit:** `8450ab40` - Phase 2.4 cleanup
- **CHANGELOG:** `D:\CCC 2.0\AI\agent-governance-system\CHANGELOG.md` (version 3.8.2)
- **Network CHANGELOG:** `D:\CCC 2.0\AI\agent-governance-system\NAVIGATION\CORTEX\network\CHANGELOG.md`
- **Roadmap:** `D:\CCC 2.0\AI\agent-governance-system\MEMORY\ARCHIVE\cassette-network-research\CASSETTE_NETWORK_ROADMAP.md`
- **Skill File:** `D:\CCC 2.0\AI\agent-governance-system\CAPABILITY\SKILLS\cortex-toolkit\run.py`
- **Test File:** `D:\CCC 2.0\AI\agent-governance-system\CAPABILITY\TESTBENCH\skills\test_cortex_toolkit.py`
- **Deprecated Skills:** `D:\CCC 2.0\AI\agent-governance-system\MEMORY\ARCHIVE\skills-deprecated/`
