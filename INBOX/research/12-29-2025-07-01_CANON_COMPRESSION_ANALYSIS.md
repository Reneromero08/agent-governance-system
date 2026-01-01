---
title: "CANON_COMPRESSION_ANALYSIS"
section: "research"
author: "System"
priority: "Medium"
created: "2025-12-29 07:01"
modified: "2025-12-29 07:01"
status: "Active"
summary: "Legacy research document migrated to canon format"
tags: ["research", "legacy"]
---
<!-- CONTENT_HASH: 54de23f5e90683ea28cf206180dc10e22b6ecb9bb46a7c991bd8261c133beeeb -->

# CANON Folder Compression Analysis

## Executive Summary

The CANON folder contains 17 files totaling 73K. The primary bloat source is `CHANGELOG.md` (17K = 23% of CANON). Most other files are appropriately sized and serve distinct purposes. One clear optimization exists; other merges offer marginal benefit.

---

## File Inventory

| File | Size | Lines | Category | Status |
|------|------|-------|----------|--------|
| CHANGELOG.md | 17K | 387 | History | **BLOAT** |
| CATALYTIC_COMPUTING.md | 5.8K | 129 | Processes | OK |
| CODEBOOK.md | 5.8K | 137 | Meta (generated) | OK |
| CRISIS.md | 5.6K | 196 | Processes | OK |
| CONTRACT.md | 5K | 61 | Constitution | OK |
| MIGRATION.md | 4.5K | 172 | Processes | Merge candidate |
| STEWARDSHIP.md | 4.4K | 173 | Processes | OK |
| ARBITRATION.md | 3.9K | 90 | Processes | OK |
| DEPRECATION.md | 3.5K | 122 | Processes | Merge candidate |
| GLOSSARY.md | 3.2K | 26 | Meta | OK |
| GENESIS.md | 2.8K | 68 | Meta | OK |
| INVARIANTS.md | 2.8K | 33 | Constitution | OK |
| SECURITY.md | 2.5K | 71 | Constitution | OK |
| INDEX.md | 2K | 27 | Meta | OK |
| AGREEMENT.md | 1.9K | 32 | Constitution | OK |
| GENESIS_COMPACT.md | 1.3K | 65 | Meta | OK |
| VERSIONING.md | 1.2K | 29 | Constitution | OK |

**Total:** 73K across 17 files

---

## Category Breakdown

### Constitution (Core Authority) - 18K
- AGREEMENT.md - Liability separation (Human = Sovereign, Agent = Instrument)
- CONTRACT.md - Non-negotiable rules and authority gradient
- INVARIANTS.md - Locked decisions requiring ceremony to change
- SECURITY.md - Trust boundaries (read/write access)
- VERSIONING.md - Version policy and compatibility

**Verdict:** No consolidation. Each is a distinct authority layer.

### Processes (Ceremonies) - 22K
- ARBITRATION.md - Conflict resolution between rules
- CRISIS.md - Emergency procedures (5 levels)
- DEPRECATION.md - End-of-life workflow
- MIGRATION.md - Breaking change execution
- STEWARDSHIP.md - Human escalation paths
- CATALYTIC_COMPUTING.md - Memory model for scratch operations

**Verdict:** DEPRECATION + MIGRATION could merge. Others are distinct.

### Meta (Navigation & Reference) - 16K
- INDEX.md - Master navigation
- GENESIS.md - Bootstrap prompt (full)
- GENESIS_COMPACT.md - Bootstrap prompt (compressed)
- GLOSSARY.md - Term definitions (manual)
- CODEBOOK.md - Compressed IDs (auto-generated)

**Verdict:** No consolidation. Different authoring modes and audiences.

### History - 17K
- CHANGELOG.md - Full change history since v0.1.0

**Verdict:** Archive older entries to CONTEXT.

---

## Consolidation Recommendations

### Priority 1: CHANGELOG Trim (Save ~15K)

**Problem:** CHANGELOG.md is 17K (387 lines) - the largest file in CANON by far. It contains full history back to v0.1.0, but agents only need recent changes for operational context.

**Solution:**
1. Keep in `CANON/CHANGELOG.md`:
   - `[Unreleased]` section
   - Last 3 released versions (2.8.0, 2.6.0, 2.5.x)
   - Footer: "Full history: `CONTEXT/archive/CHANGELOG_HISTORY.md`"

2. Create `CONTEXT/archive/CHANGELOG_HISTORY.md`:
   - Move all entries from v2.5.3 and earlier
   - Add header explaining this is the historical archive

**Impact:**
- CANON/CHANGELOG.md: 17K → ~4K
- Total CANON: 73K → ~60K (18% reduction)
- Token savings: ~3K tokens in packs

**Risk:** Low. History preserved, just relocated.

---

### Priority 2: DEPRECATION + MIGRATION Merge (Save ~1K)

**Problem:** These files describe sequential parts of the same lifecycle:
1. Deprecate item (DEPRECATION.md)
2. Wait for deprecation window
3. Execute migration (MIGRATION.md)
4. Remove item

**Solution:**
1. Create `CANON/LIFECYCLE.md` containing:
   - Deprecation policy and windows
   - Migration ceremony
   - Removal procedure

2. Delete DEPRECATION.md and MIGRATION.md

3. Update INDEX.md to point to LIFECYCLE.md

**Impact:**
- Two files (8K) → One file (~7K)
- Savings: ~1K (header/boilerplate deduplication)
- Clearer mental model (one file for "how to retire things")

**Risk:** Medium. Requires ADR and INDEX.md update.

---

### No Action Required

#### GENESIS.md + GENESIS_COMPACT.md
**Why keep separate:**
- GENESIS.md (2.8K) - Full bootstrap for human reading and new agents
- GENESIS_COMPACT.md (1.3K) - Token-efficient for constrained contexts
- GENESIS_COMPACT explicitly states: "companion to GENESIS.md, not a replacement"
- Different audiences, different use cases

#### CRISIS.md + STEWARDSHIP.md
**Why keep separate:**
- CRISIS.md (5.6K) - Emergency **procedures** (what to do)
- STEWARDSHIP.md (4.4K) - Human **roles** (who does it)
- CRISIS references STEWARDSHIP for escalation paths
- Clear separation of concerns

#### GLOSSARY.md + CODEBOOK.md
**Why keep separate:**
- GLOSSARY.md (3.2K) - Manual definitions, stable API
- CODEBOOK.md (5.8K) - Auto-generated by `TOOLS/codebook_build.py`
- Different authoring modes (human vs generated)
- CODEBOOK is regenerated; merging would break the build

#### Constitution Files
**Why keep separate:**
- AGREEMENT.md - Highest authority (liability)
- CONTRACT.md - Operational rules
- INVARIANTS.md - Locked decisions
- SECURITY.md - Trust boundaries
- VERSIONING.md - Compatibility policy
- Authority gradient requires these to be distinct layers

---

## Token Impact Summary

| State | CANON Size | Estimated Tokens |
|-------|------------|------------------|
| Current | 73K | ~18K |
| After CHANGELOG trim | ~60K | ~15K |
| After LIFECYCLE merge | ~59K | ~15K |

**Primary optimization:** CHANGELOG trim saves ~3K tokens
**Secondary optimization:** LIFECYCLE merge saves ~250 tokens

---

## Implementation Checklist

### CHANGELOG Trim
- [ ] Create `CONTEXT/archive/CHANGELOG_HISTORY.md`
- [ ] Move entries v2.5.3 and earlier to archive
- [ ] Add footer to CANON/CHANGELOG.md pointing to archive
- [ ] Update CANON/INDEX.md if needed
- [ ] No ADR required (historical reorganization, not rule change)

### LIFECYCLE Merge (Optional)
- [ ] Create ADR for DEPRECATION + MIGRATION consolidation
- [ ] Draft `CANON/LIFECYCLE.md` combining both ceremonies
- [ ] Update CANON/INDEX.md
- [ ] Update any references in CONTRACT.md, AGENTS.md
- [ ] Delete DEPRECATION.md and MIGRATION.md
- [ ] Regenerate CODEBOOK.md

---

## Files NOT to Consolidate

| Pair | Reason |
|------|--------|
| AGREEMENT + CONTRACT | Different authority levels |
| GENESIS + GENESIS_COMPACT | Different audiences (human vs token-constrained) |
| CRISIS + STEWARDSHIP | Different concerns (procedures vs roles) |
| GLOSSARY + CODEBOOK | Different authoring (manual vs generated) |
| ARBITRATION + anything | Unique conflict resolution logic |
| CATALYTIC_COMPUTING + anything | Specialized memory model theory |

---

## Risk Assessment

| Action | Risk | Mitigation |
|--------|------|------------|
| CHANGELOG trim | Low | History preserved in CONTEXT |
| LIFECYCLE merge | Medium | Requires ADR, careful reference updates |
| No consolidation | None | Status quo maintained |

---

## Conclusion

The CANON folder is well-organized with appropriate separation of concerns. The only significant bloat is CHANGELOG.md (23% of folder). Trimming it to recent versions and archiving history would reduce CANON by ~18% with minimal risk.

The optional DEPRECATION + MIGRATION merge offers marginal benefit (~1K) but requires more ceremony. All other files should remain separate.

**Recommended action:** Trim CHANGELOG only. Other consolidations are not worth the ceremony overhead.

---

*Analysis Date: December 2024*
