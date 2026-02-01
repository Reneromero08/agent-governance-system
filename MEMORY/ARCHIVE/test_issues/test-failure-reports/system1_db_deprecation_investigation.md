<!-- CONTENT_HASH: 96aa4dc4 -->

# System1.db and Canon_index.db Deprecation Investigation

**Date:** 2026-01-25
**Author:** Claude Opus 4.5 (Investigation Agent)
**Status:** Investigation Complete

---

## Executive Summary

This investigation reveals that **system1.db and canon_index.db are in a state of incomplete deprecation**. The databases were deprecated as part of the cassette network migration (Phase 1-6, completed 2026-01-11 to 2026-01-18), but the deprecation process was not fully followed according to the canonical DEPRECATION.md policy. Tests still reference these databases via skipif conditions, and codebook_lookup.py still defines paths to them for stacked resolution features (L1+L2 FTS, L1+L3 semantic).

**Key Finding:** The proper deprecation ceremony was NOT followed. There is no ADR specifically for deprecating system1.db, canon_index.db, or the codebook stacked resolution functionality. The migration happened de facto but without formal governance artifacts.

---

## 1. Deprecation Status and History

### 1.1 Timeline

| Date | Event | Evidence |
|------|-------|----------|
| 2025-12-28 | ADR-027 Dual-DB Architecture accepted | LAW/CONTEXT/decisions/ADR-027-dual-db-architecture.md |
| 2025-12-28 | Cassette Network research begins | MEMORY/ARCHIVE/cassette-network-research/ |
| 2026-01-11 | Phase 1 Cassette Partitioning complete | NAVIGATION/CORTEX/network/CHANGELOG.md |
| 2026-01-11 | Phase 2.4 Cleanup and Deprecation | 30MB of deprecated databases deleted |
| 2026-01-16 | Phase 6 Production Hardening | Cassette network validated, system1.db marked deprecated |
| 2026-01-18 | Session Cache (L4) complete | All phases complete |

### 1.2 What Was Actually Deprecated

According to NAVIGATION/CORTEX/network/CHANGELOG.md [3.8.2]:

**Removed:**
- CAPABILITY/TOOLS/cortex/* - Dead code referencing deprecated `_generated/` folder
- NAVIGATION/CORTEX/_generated/* - Deprecated compressed database artifacts
- NAVIGATION/CORTEX/db/* - Deprecated database files including:
  - system1_builder.py, cortex.build.py, build_swarm_db.py
  - adr_index.db, canon_index.db, skill_index.db, codebase_full.db
  - instructions.db, swarm_instructions.db, system2.db
  - reset_system1.py, system2_ledger.py
- Migration scripts (migrate_to_cassettes.py, structure_aware_migration.py)
- Research/demo code

**Updated to use cassette network:**
- CAPABILITY/MCP/semantic_adapter.py
- NAVIGATION/CORTEX/semantic/query.py
- NAVIGATION/CORTEX/semantic/semantic_search.py

### 1.3 Deprecation Notices Present

Several files contain deprecation comments:

1. **CAPABILITY/MCP/semantic_adapter.py** (line 8):
   ```
   Note: system1.db is deprecated. All semantic search is now handled
   by the cassette network (NAVIGATION/CORTEX/cassettes/).
   ```

2. **NAVIGATION/CORTEX/semantic/semantic_search.py** (line 5):
   ```
   DEPRECATED: system1.db is deprecated. Use the cassette network for search
   ```

3. **NAVIGATION/CORTEX/semantic/query.py** (line 7):
   ```
   Note: system1.db and cortex.db are deprecated. All queries now route
   through the cassette network (NAVIGATION/CORTEX/cassettes/).
   ```

4. **NAVIGATION/CORTEX/network/cassettes.json** (line 150):
   ```json
   {
     "id": "system1",
     "name": "Legacy System1 (deprecated)",
     "db_path": "NAVIGATION/CORTEX/db/system1.db",
     "enabled": false
   }
   ```

5. **NAVIGATION/CORTEX/README.md** (line 75):
   ```
   1. **v1.0** - Monolithic `system1.db` (deprecated, removed)
   2. **v2.0** - `_generated/cortex.db` with section indexes (deprecated, removed)
   3. **v3.0** - Cassette network with bucket-aligned partitions
   ```

---

## 2. Governance Compliance Analysis

### 2.1 Deprecation Policy Requirements (LAW/CANON/GOVERNANCE/DEPRECATION.md)

The policy requires:

| Requirement | Status | Evidence |
|------------|--------|----------|
| Create deprecation ADR | **NOT DONE** | No ADR-xxx-deprecate-system1.md exists |
| Mark item as deprecated with notice | PARTIAL | Comments in code, but not formal notices |
| Specify deprecation window | **NOT DONE** | No version/date window specified |
| Specify replacement | PARTIAL | Cassette network mentioned informally |
| Create migration artifacts | DONE | Migration scripts created (now deleted) |
| Update CHANGELOG | DONE | Changelog entries present |

### 2.2 Item Type Classification

According to the policy, different items have different deprecation windows:

| Item Type | Window Required | system1.db Classification |
|-----------|-----------------|---------------------------|
| Canon rules | 2 major versions OR 90 days | N/A |
| Token grammar | 1 major version OR 30 days | N/A |
| Skills | 1 minor version OR 14 days | N/A |
| **Cortex schema** | **1 major version OR 30 days** | **APPLICABLE** |

**Analysis:** system1.db and canon_index.db are part of the Cortex schema. According to policy, they require a 1 major version OR 30 days deprecation window where BOTH old and new paths MUST work.

### 2.3 Compliance Issues

1. **No ADR for deprecation**: The policy explicitly requires `CONTEXT/decisions/ADR-xxx-deprecate-*.md` but none exists for system1.db or canon_index.db.

2. **No formal deprecation window**: The databases were removed without a documented deprecation window.

3. **Both paths not maintained**: The old paths (system1.db) do not work because the files were deleted. The policy states "During the deprecation window, both old and new paths MUST work."

4. **Tests left in limbo**: Tests that use `@pytest.mark.skipif(not SYSTEM1_DB.exists())` indicate the old path was expected to potentially work, but this was never resolved.

---

## 3. Current State of Affected Tests

### 3.1 Tests Referencing system1.db

**File:** `CAPABILITY/TESTBENCH/integration/test_phase_5_2_3_stacked_resolution.py`

```python
@pytest.mark.skipif(not SYSTEM1_DB.exists(), reason="system1.db not available")
class TestL1L2FTSResolution:
    def test_fts_stacked_resolution(self):
        result = stacked_lookup("\u6cd5", query="verification", limit=5)
        assert result["found"] is True
        assert result["resolution"] == "L1+L2"
```

**Issue:** This test is perpetually skipped because system1.db no longer exists. The test is testing L1+L2 stacked resolution (FTS search within a symbol domain), which was the codebook_lookup feature that relied on system1.db.

### 3.2 Tests Referencing canon_index.db

**Same file:**

```python
@pytest.mark.skipif(not CANON_INDEX_DB.exists(), reason="canon_index.db not available")
class TestL1L3SemanticResolution:
    def test_semantic_stacked_resolution(self):
        result = stacked_lookup("\u6cd5", semantic="verification protocols", limit=5)
        assert result["found"] is True
        assert result["resolution"] == "L1+L3"
```

**Issue:** Same problem - canon_index.db does not exist, so this test is perpetually skipped.

### 3.3 codebook_lookup.py Still References These

**File:** `CAPABILITY/TOOLS/codebook_lookup.py` (lines 446-449)

```python
SYSTEM1_DB = PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "db" / "system1.db"
CANON_INDEX_DB = PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "db" / "canon_index.db"
ADR_INDEX_DB = PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "db" / "adr_index.db"
SKILL_INDEX_DB = PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "db" / "skill_index.db"
```

These paths define where the stacked resolution looks for FTS and semantic search databases. The functions `_fts_search_within_paths()` and `_semantic_search_within_paths()` check if these files exist before attempting queries.

---

## 4. What The Tests SHOULD Be Testing Now

### 4.1 The Feature Being Tested

The stacked resolution feature (Phase 5.2.3) allows:
- **L1**: Pure symbol resolution (e.g., "\u6cd5" -> LAW/CANON)
- **L1+L2**: Symbol + FTS query (e.g., "\u6cd5" + "verification" -> FTS search within LAW/CANON)
- **L1+L3**: Symbol + semantic query (e.g., "\u6cd5" + semantic="verification protocols" -> vector search within LAW/CANON)

### 4.2 Options for the Tests

**Option A: Migrate to Cassette Network**

The tests should use the cassette network's FTS5 and semantic search capabilities:
- FTS: Query `canon.db` cassette directly using FTS5
- Semantic: Query `canon.db` cassette using geometric/vector search

**Option B: Rebuild the Index Databases**

The `canon_index.py` primitive can rebuild `canon_index.db`:
```bash
python CAPABILITY/PRIMITIVES/canon_index.py --embed
```

This would create `canon_index.db` at `NAVIGATION/CORTEX/db/canon_index.db`.

**Option C: Mark Tests as Xfail (Deprecated Feature)**

If the stacked resolution feature (L1+L2, L1+L3) is being deprecated along with the databases, the tests should be marked as `@pytest.mark.xfail(reason="Stacked resolution deprecated - see ADR-xxx")`.

**Option D: Remove Tests Entirely**

If the feature is fully deprecated and no longer needed, remove the tests after creating a proper deprecation ADR.

---

## 5. The Real Fix

### 5.1 Recommended Actions

#### Short-term (Immediate)

1. **Create a deprecation ADR** for system1.db, canon_index.db, and the stacked resolution feature:
   - `LAW/CONTEXT/decisions/ADR-XXX-deprecate-stacked-resolution-databases.md`
   - Document what was deprecated, why, and the replacement (cassette network)
   - Acknowledge the deprecation already happened and formalize it retroactively

2. **Update tests** based on decision:
   - If stacked resolution is deprecated: Mark tests with `@pytest.mark.xfail` referencing the ADR
   - If stacked resolution should migrate: Update to use cassette network

#### Medium-term

3. **Clean up codebook_lookup.py**:
   - Either remove the dead paths (SYSTEM1_DB, CANON_INDEX_DB, etc.)
   - Or implement stacked resolution using cassette network instead

4. **Update cassettes.json**:
   - Remove the `legacy_cassettes` section entirely, or
   - Add formal deprecation metadata (version deprecated, removal target)

#### Long-term

5. **Follow proper ceremony for future deprecations**:
   - Always create ADR before deprecating
   - Maintain both paths during deprecation window
   - Only remove after window expires

### 5.2 What Would It Take to Rebuild canon_index.db?

If the decision is to keep stacked resolution working:

1. **Run the canon_index.py primitive:**
   ```bash
   python CAPABILITY/PRIMITIVES/canon_index.py --embed
   ```

2. **This will:**
   - Inventory all files in LAW/CANON
   - Generate embeddings for each file
   - Store in `NAVIGATION/CORTEX/db/canon_index.db`
   - Emit a receipt for the operation

3. **Dependency:** Requires the EmbeddingEngine from `NAVIGATION/CORTEX/semantic/embeddings.py`

### 5.3 Can codebook_lookup Use Cassette Network Instead?

**Yes.** The cassette network already provides:

1. **FTS search** via `query.py`:
   ```python
   from NAVIGATION.CORTEX.semantic.query import CortexQuery
   cq = CortexQuery()
   results = cq.search_fts("verification", cassettes=["canon"])
   ```

2. **Semantic search** via cassette geometric queries:
   ```python
   network = GeometricCassetteNetwork()
   results = network.query("verification protocols", cassettes=["canon"])
   ```

**Migration path:**
- Replace `_fts_search_within_paths()` with cassette FTS query
- Replace `_semantic_search_within_paths()` with cassette geometric query
- Update path filtering to use cassette filtering instead

---

## 6. Conclusion

### 6.1 Root Cause

The root cause is **incomplete deprecation governance**. The migration from system1.db to cassette network was done correctly from a technical standpoint (the cassette network works, system1.db was removed), but the governance ceremony was not followed:

1. No deprecation ADR was created
2. No deprecation window was observed
3. Tests were left in a "skipif exists" limbo
4. Code (codebook_lookup.py) still references the deprecated paths

### 6.2 Status Summary

| Component | Status | Proper Deprecation? |
|-----------|--------|---------------------|
| system1.db | Deleted | **NO** - No ADR, no window |
| canon_index.db | Never built | **NO** - Left as dead code |
| Stacked resolution tests | Skipped forever | **NO** - Not migrated or formally deprecated |
| codebook_lookup.py | Has dead paths | **NO** - Not cleaned up |
| cassettes.json legacy section | Exists but disabled | Partial - marked deprecated but not removed |

### 6.3 Recommended Priority

**Priority 1 (Do Now):**
- Create ADR-XXX documenting the retroactive deprecation
- Update tests to use cassette network OR mark as xfail

**Priority 2 (Next Sprint):**
- Clean up codebook_lookup.py dead paths
- Migrate stacked resolution to cassette network if needed

**Priority 3 (Debt Cleanup):**
- Remove legacy_cassettes from cassettes.json
- Archive deprecated skills referencing system1.db

---

## References

| Document | Location |
|----------|----------|
| ADR-027 Dual-DB Architecture | LAW/CONTEXT/decisions/ADR-027-dual-db-architecture.md |
| Cassette Network Spec | LAW/CANON/SEMANTIC/CASSETTE_NETWORK_SPEC.md |
| Deprecation Policy | LAW/CANON/GOVERNANCE/DEPRECATION.md |
| Cassette Network Roadmap | MEMORY/ARCHIVE/cassette-network-research/CASSETTE_NETWORK_ROADMAP.md |
| CORTEX Network CHANGELOG | NAVIGATION/CORTEX/network/CHANGELOG.md |
| codebook_lookup.py | CAPABILITY/TOOLS/codebook_lookup.py |
| Stacked Resolution Tests | CAPABILITY/TESTBENCH/integration/test_phase_5_2_3_stacked_resolution.py |
| canon_index.py | CAPABILITY/PRIMITIVES/canon_index.py |

---

**Report Hash:** To be computed on commit
