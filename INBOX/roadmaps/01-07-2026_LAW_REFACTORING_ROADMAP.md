---
uuid: 7b3e1f8a-4d92-5c6e-a1b0-9f2c8d4e6a3b
title: "LAW Refactoring — Implementation Roadmap"
section: roadmap
bucket: law/governance
author: Claude Opus 4.5
priority: Medium
created: 2026-01-07
modified: 2026-01-07
status: IN_PROGRESS
summary: Comprehensive refactoring roadmap for the LAW directory. Includes CONTRACT.md fixes, _runs cleanup strategy with artifact preservation, canon modernization, CANON bucket reorganization, and tooling improvements.
tags:
- law
- refactoring
- governance
- cleanup
- runs-directory
- canon-buckets
---
<!-- CONTENT_HASH: UPDATE_PENDING -->

# LAW Refactoring — Implementation Roadmap

**Date:** 2026-01-07
**Status:** IN PROGRESS (Phases 1-4, 6 complete; Phase 5 manual pending; Phase 7 in progress)
**Scope:** LAW directory structure, CONTRACT.md fixes, _runs cleanup, CANON bucket reorganization, tooling improvements

---

## Executive Summary

The LAW directory serves as the Supreme Authority bucket of the Agent Governance System. This roadmap addresses:
1. **Critical bug** in CONTRACT.md (duplicate rule numbering) ✓
2. **Careful cleanup** of `_runs/` directory with artifact preservation (PENDING - manual)
3. **Structural improvements** to canon organization ✓
4. **Tooling enhancements** for governance validation ✓
5. **Canon file compression** to meet INV-009 limits ✓
6. **CANON bucket reorganization** for better info architecture (IN PROGRESS)

**Risk Level:** Medium — `_runs/` contains irreplaceable phase receipts and reports that must be preserved.

---

## Task Checklist

### Phase 1: CONTRACT.md Numbering Fix (P0) ✓ COMPLETE
- [x] Read current CONTRACT.md
- [x] Renumber rules sequentially (1-13)
- [x] Update content hash
- [ ] Update cross-references in other canon files (if any)

### Phase 2: Canon Structural Improvements (P2) ✓ COMPLETE
- [x] Create `LAW/CANON/canon.json` with categorized files
- [x] Create `LAW/CONTEXT/archive/` directory
- [ ] Move any superseded ADRs to archive (none identified yet)

### Phase 3: Runner Modernization (P3) ✓ COMPLETE
- [x] Add `FixtureResult` dataclass
- [x] Add `--json` flag for JSON output
- [x] Add `--filter PATTERN` option
- [ ] Add `--parallel` flag (deferred — not critical)
- [x] Preserve backward compatibility

### Phase 4: CI Validation Improvements (P3) ✓ COMPLETE
- [x] Create `canon-validators` fixture directory
- [x] Implement duplicate rule number check
- [x] Implement line count warning (250) and error (300)
- [x] Implement authority gradient consistency check
- [ ] Integrate into CI workflow (fixture exists, needs CI hook)

### Phase 5: _runs Directory Cleanup (Manual — PENDING)
- [ ] Generate inventory of all files
- [ ] Archive RECEIPTS/, REPORTS/, LOGS/ to MEMORY/ARCHIVE/
- [ ] Clean disposable directories (pytest_tmp, _tmp, _cache, test-*)
- [ ] Manual review of _demos/, _pipelines/, feedback/, etc.
- [ ] Add .gitignore rules
- [ ] Create ADR-036 for hygiene policy

### Phase 6: Canon File Compression (P2) ✓ COMPLETE
- [x] Compress DOCUMENT_POLICY.md (402 → 203 lines)
- [x] Compress STEWARDSHIP.md (374 → 160 lines)
- [x] Engineering culture kept implicit in STEWARDSHIP.md
- [x] Do NOT touch CATALYTIC/* files (per user request)

### Phase 7: CANON Bucket Reorganization (P2) — IN PROGRESS
- [ ] Create bucket subdirectories (CONSTITUTION, GOVERNANCE, POLICY, META, capabilities)
- [ ] Move files to appropriate buckets
- [ ] Update canon.json to reflect new structure
- [ ] Update cross-references in moved files
- [ ] Verify no broken links

---

## Phase 1: CONTRACT.md Numbering Fix (P0 — Critical) ✓ COMPLETE

### 1.1 Problem Statement

~~[LAW/CANON/CONTRACT.md](LAW/CANON/CONTRACT.md) has duplicate rule numbers:~~
- ~~Lines 37, 44, 45: Three rules numbered "4."~~
- ~~Lines 57, 58: Two rules numbered "8."~~

**FIXED:** Rules renumbered 1-13. Content hash updated.

### 1.2 Implementation

1. ~~Read current CONTRACT.md~~
2. ~~Renumber rules sequentially (1-13)~~
3. ~~Update any cross-references in other canon files~~
4. ~~Add fixture to validate sequential numbering in CI~~

### 1.3 Verification

```bash
# Check no duplicate numbered rules
grep -E "^[0-9]+\." LAW/CANON/CONTRACT.md | sort | uniq -d
# Should return empty ✓
```

---

## Phase 2: Canon Structural Improvements (P2) ✓ COMPLETE

### 2.1 Machine-Readable Canon Index ✓

Created `LAW/CANON/canon.json` with 29 files categorized into:
- foundations, constitution, governance, processes, meta, protocols, catalytic

### 2.2 CONTEXT Archive ✓

Created `LAW/CONTEXT/archive/` directory.

**Why this exists:** Standard governance pattern for superseded ADRs. When an ADR is replaced by a newer decision, move it here rather than deleting (preserves audit history per INV-010).

---

## Phase 3: Runner Modernization (P3) ✓ COMPLETE

### 3.1 Current State (Before)

[LAW/CONTRACTS/runner.py](LAW/CONTRACTS/runner.py) was 213 lines, functional but basic:
- Returned simple int failure count
- No parallel execution
- No structured output
- No filtering

### 3.2 Enhancements Added ✓

| Feature | Status |
|---------|--------|
| `FixtureResult` dataclass | ✓ Implemented |
| `--json` flag | ✓ Implemented |
| `--filter PATTERN` | ✓ Implemented |
| `--parallel` | Deferred (not critical) |
| Backward compatibility | ✓ Verified |

### 3.3 Implementation

```python
@dataclass
class FixtureResult:
    name: str
    status: Literal["passed", "failed", "skipped"]
    duration_ms: int
    error: Optional[str] = None

def run_fixtures(filter_pattern: str = None, quiet: bool = False) -> List[FixtureResult]:
    ...
```

---

## Phase 4: CI Validation Improvements (P3) ✓ COMPLETE

### 4.1 Canon Validators Created ✓

Location: `LAW/CONTRACTS/fixtures/governance/canon-validators/`

**Checks implemented:**
- Duplicate rule numbers within sections
- Line count: warn at 250, error at 300 (per INV-009)
- Authority gradient consistency with canon.json

### 4.2 Findings

**6 files exceed 300-line limit (INV-009 violation):**

| File | Lines | Action |
|------|-------|--------|
| `CATALYTIC/CMP-01_CATALYTIC_MUTATION_PROTOCOL.md` | 351 | Skip (per user) |
| `CATALYTIC/SPECTRUM-04_IDENTITY_SIGNING.md` | 738 | Skip (per user) |
| `CATALYTIC/SPECTRUM-05_VERIFICATION_LAW.md` | 507 | Skip (per user) |
| `CATALYTIC/SPECTRUM-06_RESTORE_RUNNER.md` | 606 | Skip (per user) |
| `DOCUMENT_POLICY.md` | 402 | ✓ Compressed in Phase 6 |
| `STEWARDSHIP.md` | 374 | ✓ Compressed in Phase 6 |

---

## Phase 5: _runs Directory Cleanup (Manual — Last Phase)

> **This phase is intentionally last.** All other phases must complete first, allowing time for manual review and artifact sorting.

### 5.1 Artifact Classification

The `_runs/` directory contains mixed content. **MANUAL REVIEW REQUIRED** before any deletion.

#### 5.1.1 PROTECTED (Never Delete Without Ceremony)

| Directory/File | Purpose | Why Protected |
|----------------|---------|---------------|
| `RECEIPTS/` | Phase task receipts | Proof of task completion by phase |
| `REPORTS/` | Phase task reports | Implementation documentation |
| `LOGS/` | Phase execution logs | Audit trail |
| `fixtures/` | Contract runner output | Test evidence |
| `mcp_logs/` | MCP execution logs | Runtime audit |
| `mcp_ledger/` | MCP state ledger | System state |
| `override_logs/` | MASTER_OVERRIDE audit | Security audit trail |
| `catalytic/` | Catalytic computing artifacts | System proofs |
| `determinism/` | Determinism test results | INV-005 evidence |
| `p2_determinism/` | Phase 2 determinism proofs | Phase-specific evidence |
| `pipeline-*` | Pipeline execution runs | Execution artifacts |
| `commit_queue/` | Commit queue state | Active system state |
| `inbox_weekly_*` | Weekly inbox digests | Historical summaries |
| `*.json` root files | System state snapshots | `RESTORE_PROOF.json`, `PURITY_SCAN.json`, etc. |

#### 5.1.2 DISPOSABLE (Safe to Clean)

| Pattern | Purpose | Why Safe |
|---------|---------|----------|
| `pytest_tmp/` | pytest temporary files | Regenerable, per INV-014 |
| `_tmp/` | Scratch space | Explicitly disposable per INV-014 |
| `_cache/` | Cache data | Regenerable |
| `test-adapt-*` | Ephemeral test runs | UUID-tagged, no ceremony |
| `test-copy-*` | Ephemeral test runs | UUID-tagged, no ceremony |
| `test-ledger-*` | Ephemeral test runs | UUID-tagged, no ceremony |
| `test-missing-*` | Ephemeral test runs | UUID-tagged, no ceremony |
| `ant-worker-copy-*` | Worker copies | UUID-tagged, temporary |

#### 5.1.3 REQUIRES MANUAL REVIEW

| Directory | Current State | Action Required |
|-----------|---------------|-----------------|
| `_demos/` | Demo artifacts | Review if demos still needed |
| `_pipelines/` | Pipeline configs | Check if active |
| `_test_pipeline_toolkit/` | Test toolkit artifacts | Assess if needed |
| `canonical-doc-enforcer/` | Doc enforcer output | Check if reports needed |
| `doc_update/` | Doc update artifacts | Review for preservation |
| `feedback/` | Feedback records | May contain valuable data |
| `intent/` | Intent records | May be important context |
| `message_board/` | Message board state | Check if active |
| `terminals/` | Terminal session logs | Review for audit value |
| `test_smoke/` | Smoke test results | May contain key evidence |

### 5.2 Cleanup Strategy

**DO NOT AUTOMATE DELETION.** Follow this manual process:

#### Step 1: Inventory (Before Any Changes)

```bash
# Generate inventory of all files
find LAW/CONTRACTS/_runs -type f > _runs_inventory.txt
wc -l _runs_inventory.txt  # Record count

# Generate size report
du -sh LAW/CONTRACTS/_runs/*
```

#### Step 2: Archive Protected Content

```bash
# Create dated archive of protected directories
mkdir -p MEMORY/ARCHIVE/runs_backup_2026-01-07
cp -r LAW/CONTRACTS/_runs/RECEIPTS MEMORY/ARCHIVE/runs_backup_2026-01-07/
cp -r LAW/CONTRACTS/_runs/REPORTS MEMORY/ARCHIVE/runs_backup_2026-01-07/
cp -r LAW/CONTRACTS/_runs/LOGS MEMORY/ARCHIVE/runs_backup_2026-01-07/
```

#### Step 3: Clean Disposable Safely

```bash
# Remove pytest temp (deep nesting, safe)
rm -rf LAW/CONTRACTS/_runs/pytest_tmp/

# Remove _tmp (explicitly disposable)
rm -rf LAW/CONTRACTS/_runs/_tmp/

# Remove _cache (regenerable)
rm -rf LAW/CONTRACTS/_runs/_cache/

# Remove ephemeral test runs (UUID-tagged)
rm -rf LAW/CONTRACTS/_runs/test-adapt-*
rm -rf LAW/CONTRACTS/_runs/test-copy-*
rm -rf LAW/CONTRACTS/_runs/test-ledger-*
rm -rf LAW/CONTRACTS/_runs/test-missing-*
rm -rf LAW/CONTRACTS/_runs/ant-worker-copy-*
```

#### Step 4: Manual Review Directories

For each directory in section 5.1.3:
1. List contents: `ls -laR <directory>`
2. Check modification dates
3. If older than 30 days and not referenced → move to archive
4. If actively used → keep in place
5. Document decision in ADR

#### Step 5: Add .gitignore Rules

Create `LAW/CONTRACTS/_runs/.gitignore`:

```gitignore
# Ephemeral test artifacts (safe to regenerate)
pytest_tmp/
test-adapt-*
test-copy-*
test-ledger-*
test-missing-*
ant-worker-copy-*

# Explicitly disposable per INV-014
_tmp/

# Regenerable cache
_cache/
```

### 5.3 Verification After Cleanup

```bash
# Verify protected directories intact
test -d LAW/CONTRACTS/_runs/RECEIPTS && echo "RECEIPTS OK"
test -d LAW/CONTRACTS/_runs/REPORTS && echo "REPORTS OK"
test -d LAW/CONTRACTS/_runs/LOGS && echo "LOGS OK"

# Run fixtures to ensure nothing broken
python LAW/CONTRACTS/runner.py
```

### 5.4 Effort

Medium-High (2-4 hours manual review)

---

## Phase 6: Canon File Compression ✓ COMPLETE

### 6.1 DOCUMENT_POLICY.md ✓
- **Before:** 402 lines
- **After:** 203 lines
- **Method:** Merged duplicate examples, trimmed Python code, condensed migration guide

### 6.2 STEWARDSHIP.md ✓
- **Before:** 374 lines
- **After:** 160 lines
- **Method:** Compressed engineering culture section (kept implicit per user request)

### 6.3 NOT TOUCHED (per user request)
- `CATALYTIC/*.md` files — left as-is

---

## Phase 7: CANON Bucket Reorganization (P2) — IN PROGRESS

### 7.1 Problem Statement

Current `LAW/CANON/` is a flat directory with 25+ files. This makes it difficult to:
- Find related documents quickly
- Understand the authority hierarchy
- Maintain logical groupings

### 7.2 Proposed Structure

```
LAW/CANON/
├── CONSTITUTION/           # Foundational law (highest authority)
│   ├── AGREEMENT.md        # Supreme agreement
│   ├── CONTRACT.md         # Non-negotiable rules
│   ├── FORMULA.md          # The living formula
│   ├── INVARIANTS.md       # System invariants
│   └── INTEGRITY.md        # Integrity rules
│
├── GOVERNANCE/             # How the system operates
│   ├── VERSIONING.md
│   ├── DEPRECATION.md
│   ├── MIGRATION.md
│   ├── ARBITRATION.md
│   ├── CRISIS.md
│   ├── STEWARDSHIP.md
│   └── VERIFICATION_PROTOCOL_CANON.md
│
├── POLICY/                 # Policies and standards
│   ├── DOCUMENT_POLICY.md
│   ├── IMPLEMENTATION_REPORTS.md
│   ├── SECURITY.md
│   └── AGENT_SEARCH_PROTOCOL.md
│
├── META/                   # About the system itself
│   ├── GENESIS.md
│   ├── GENESIS_COMPACT.md
│   ├── SYSTEM_BUCKETS.md
│   ├── GLOSSARY.md
│   ├── CODEBOOK.md
│   └── INDEX.md
│
├── CATALYTIC/              # Already exists - catalytic computing
│   └── (existing files - DO NOT TOUCH)
│
├── SEMANTIC/               # Already exists - semantic layer
│   └── TOKEN_RECEIPT_SPEC.md
│
├── capabilities/           # JSON configs (lowercase for data files)
│   ├── CAPABILITIES.json
│   ├── CAPABILITY_PINS.json
│   ├── CAPABILITY_REVOKES.json
│   └── swarm_config.json
│
└── canon.json              # Index at root (updated to reflect structure)
```

### 7.3 Authority Hierarchy

The bucket organization reflects the authority gradient:

1. **CONSTITUTION/** — Supreme authority, rarely changes
2. **GOVERNANCE/** — How decisions are made and executed
3. **POLICY/** — Operational standards and requirements
4. **META/** — Documentation about the system itself
5. **CATALYTIC/** — Cryptographic computing protocols
6. **SEMANTIC/** — Token and symbol definitions
7. **capabilities/** — Runtime configuration

### 7.4 Migration Plan

1. Create subdirectories
2. Move files to appropriate buckets
3. Update `canon.json` with new paths
4. Search for and update cross-references
5. Verify no broken links

### 7.5 Effort

Medium (1-2 hours)

---

## Implementation Order

| Phase | Priority | Status | Effort |
|-------|----------|--------|--------|
| 1. CONTRACT.md Fix | P0 | ✓ COMPLETE | Done |
| 2. Canon Structure | P2 | ✓ COMPLETE | Done |
| 3. Runner Modernization | P3 | ✓ COMPLETE | Done |
| 4. CI Validators | P3 | ✓ COMPLETE | Done |
| 5. _runs Cleanup | Manual | PENDING | 2-4 hours |
| 6. Canon Compression | P2 | ✓ COMPLETE | Done |
| 7. CANON Bucket Reorg | P2 | IN PROGRESS | 1-2 hours |

**Sequence:** 1 ✓ → 2 ✓ → 3 ✓ → 4 ✓ → 6 ✓ → 7 → 5

Phase 5 (_runs cleanup) is intentionally last because:
- It requires careful manual review of each directory
- All other refactoring should complete first
- Protected artifacts must be inventoried and archived before any cleanup
- You (the human) will manually sort receipts, reports, and logs from key phases

---

## Appendix A: Current _runs Directory State (2026-01-07)

### Protected Directories with Contents

```
RECEIPTS/
├── packer-pruned-second-pass.json
├── phase-2/
│   ├── task-2.4.1c.2_pipeline_mcp_firewall.json
│   └── task-2.4.1C.2_runtime_write_surface_enforcement.json
└── phase-7/
    ├── task-7.4.1_pruned_pack_variant.json
    ├── task-7.4.2_pruned_pack_polish.json
    ├── task-7.4.3_pruned_atomic_replace_hardening.json
    ├── task-7.4.4_pruned_second_pass_runner_fix.json
    └── task-7.4.5_pruned_always_on.json

REPORTS/
├── phase-2/
│   ├── task-2.4.1c.2_pipeline_mcp_firewall.md
│   └── task-2.4.1C.2_runtime_write_surface_enforcement.md
└── phase-7/
    └── (multiple dated task reports)

LOGS/
└── phase-2/
    └── (phase execution logs)
```

### Key Root Files

| File | Size | Purpose |
|------|------|---------|
| `RESTORE_PROOF.json` | 19KB | Catalytic restore proof |
| `PRE_DIGEST.json` | 13KB | Pre-execution digest |
| `POST_DIGEST.json` | 14KB | Post-execution digest |
| `PURITY_SCAN.json` | 13KB | Purity verification |
| `INBOX_DRY_RUN.json` | 20KB | Inbox dry run results |
| `INBOX_EXECUTION.json` | 3KB | Inbox execution results |

---

## Appendix B: ADR Required for This Work

Before implementing Phase 5 (cleanup), create:

**ADR-036: _runs Directory Hygiene Policy**

Context:
- `_runs/` accumulates ephemeral test artifacts
- Protected artifacts mixed with disposable ones
- No clear retention policy

Decision:
- Define protected vs disposable artifact classes
- Add .gitignore for ephemeral patterns
- Document retention periods for each class

Consequences:
- Cleaner repository
- Preserved audit trail
- Clear policy for future cleanup

---

## Changelog

| Date | Change |
|------|--------|
| 2026-01-07 | Initial roadmap created |
| 2026-01-07 | Reorganized: Phase 5 (_runs cleanup) moved to last |
| 2026-01-07 | Phases 1-4 COMPLETE; Added Phase 6 (canon compression) |
| 2026-01-07 | Phase 6 COMPLETE; Added Phase 7 (CANON bucket reorganization) |
