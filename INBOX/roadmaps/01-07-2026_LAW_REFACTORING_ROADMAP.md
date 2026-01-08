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
summary: Comprehensive refactoring roadmap for the LAW directory. Includes CONTRACT.md fixes, _runs cleanup strategy with artifact preservation, canon modernization, and tooling improvements.
tags:
- law
- refactoring
- governance
- cleanup
- runs-directory
---
<!-- CONTENT_HASH: b99a5d992d49316819750b0f5835de01c5256e917b4e9b035755d087f04daf13 -->

# LAW Refactoring — Implementation Roadmap

**Date:** 2026-01-07
**Status:** IN PROGRESS (Phases 1-4, 6 complete; Phase 5 manual pending)
**Scope:** LAW directory structure, CONTRACT.md fixes, _runs cleanup, tooling improvements

---

## Task Checklist

### Phase 1: CONTRACT.md Numbering Fix (P0)
- [x] Read current CONTRACT.md
- [x] Renumber rules sequentially (1-13)
- [x] Update content hash
- [ ] Update cross-references in other canon files (if any)

### Phase 2: Canon Structural Improvements (P2)
- [x] Create `LAW/CANON/canon.json` with categorized files
- [x] Create `LAW/CONTEXT/archive/` directory
- [ ] Move any superseded ADRs to archive (none identified yet)

### Phase 3: Runner Modernization (P3)
- [x] Add `FixtureResult` dataclass
- [x] Add `--json` flag for JSON output
- [x] Add `--filter PATTERN` option
- [ ] Add `--parallel` flag (deferred — not critical)
- [x] Preserve backward compatibility

### Phase 4: CI Validation Improvements (P3)
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

### Phase 6: Canon File Compression (P2)
- [x] Compress DOCUMENT_POLICY.md (402 → 203 lines) ✓
- [x] Compress STEWARDSHIP.md (374 → 160 lines) ✓
- [x] Engineering culture kept implicit in STEWARDSHIP.md
- [x] Do NOT touch CATALYTIC/* files (per user request)

---

## Phase 1: CONTRACT.md Numbering Fix (P0 — Critical) ✓ COMPLETE

### 1.1 Problem Statement

~~[LAW/CANON/CONTRACT.md](LAW/CANON/CONTRACT.md) has duplicate rule numbers~~

**FIXED:** Rules renumbered 1-13. Content hash updated.

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

### 3.1 Enhancements Added

| Feature | Status |
|---------|--------|
| `FixtureResult` dataclass | ✓ Implemented |
| `--json` flag | ✓ Implemented |
| `--filter PATTERN` | ✓ Implemented |
| `--parallel` | Deferred |
| Backward compatibility | ✓ Verified |

---

## Phase 4: CI Validation Improvements (P3) ✓ COMPLETE

### 4.1 Canon Validators Created

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
| `DOCUMENT_POLICY.md` | 402 | **Compress in Phase 6** |
| `STEWARDSHIP.md` | 374 | **Compress in Phase 6** |

---

## Phase 5: _runs Directory Cleanup (Manual — Last Phase)

> **This phase is intentionally last.** All other phases must complete first, allowing time for manual review and artifact sorting.

*(Content unchanged — see sections 5.1-5.4 below)*

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

### 5.2-5.4 *(Unchanged — see previous version)*

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

## Implementation Order

| Phase | Priority | Status | Effort |
|-------|----------|--------|--------|
| 1. CONTRACT.md Fix | P0 | ✓ COMPLETE | Done |
| 2. Canon Structure | P2 | ✓ COMPLETE | Done |
| 3. Runner Modernization | P3 | ✓ COMPLETE | Done |
| 4. CI Validators | P3 | ✓ COMPLETE | Done |
| 5. _runs Cleanup | Manual | PENDING | 2-4 hours |
| 6. Canon Compression | P2 | ✓ COMPLETE | Done |

**Sequence:** 1 ✓ → 2 ✓ → 3 ✓ → 4 ✓ → 6 ✓ → 5

---

## Appendix A: Current _runs Directory State (2026-01-07)

*(Unchanged)*

---

## Appendix B: ADR Required for This Work

Before implementing Phase 5 (cleanup), create:

**ADR-036: _runs Directory Hygiene Policy**

---

## Changelog

| Date | Change |
|------|--------|
| 2026-01-07 | Initial roadmap created |
| 2026-01-07 | Reorganized: Phase 5 (_runs cleanup) moved to last |
| 2026-01-07 | Phases 1-4 COMPLETE; Added Phase 6 (canon compression) |
