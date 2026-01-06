---
uuid: 00000000-0000-0000-0000-000000000000
title: 01-06-2026-12-58 Inbox Normalization Report V1 1
section: report
bucket: INBOX/reports
author: System
priority: Medium
created: 2026-01-06 13:09
modified: 2026-01-06 13:09
status: Active
summary: Document summary
tags: []
hashtags: []
---
<!-- CONTENT_HASH: ccf71d0b732e9ac13ab3a9938a02e0ada7f98a70ec2a9060c0364e27d9ca74fc -->
# INBOX Normalization Report v1.1

**Date:** 2026-01-05  
**Operation Version:** 1.1.0  
**Version Hash:** `a7f3c9e2d8b1a456`  
**Based on:** v1.0.0 (01-05-2026-20-29)


---

## What Changed from v1.0 Report

This v1.1 report corrects three ambiguities identified in v1.0:

| Issue | v1.0 Ambiguity | v1.1 Correction |
|-------|----------------|-----------------|
| **ISO Week vs Month** | "Week-01" in "2025-12" folder not explained | Added explicit ISO 8601 week documentation |
| **Timestamp Authority** | `"prefer_filename_over_mtime"` suggested mtime fallback | Changed to explicit `filename_only` with `fail_closed: true` |
| **Digest Semantics** | "Pre and post digest match" - misleading | Separated into `content_integrity` (PASS) and `tree_digest` (changes expected) |

---

## Schema Clarification

### Folder Structure: `YYYY-MM/Week-XX`

**This is NOT calendar month grouping.** The schema uses:
- **Primary folder**: Calendar month derived from timestamp date (`YYYY-MM`)
- **Secondary folder**: ISO 8601 week number (`Week-XX`)

### ISO 8601 Week Number Semantics

> ⚠️ **IMPORTANT**: ISO 8601 week numbers do NOT align perfectly with calendar months.

| Week Number | Can Start In | Can End In |
|-------------|--------------|------------|
| Week-01 | December 29-31 | January 1-5 |
| Week-52/53 | December 28-31 | January 1-4 |

**Example from this normalization:**
- `2025-12-29` (Monday) is in **ISO Week-01** of 2026
- This file appears at `2025-12/Week-01/` (month = December, week = Week-01 of 2026)
- This is **correct behavior** per ISO 8601 standard

### Why This Design?

1. **Temporal ordering**: Files are organized by when they were created
2. **ISO week standard**: Provides consistent week numbering across years
3. **Month context**: Calendar month helps locate files temporally within a year
4. **Predictable**: Same timestamp always produces same folder path

---

## Timestamp Authority Policy

### Source Priority (Fail-Closed)

| Priority | Source | Fallback | Behavior |
|----------|--------|----------|----------|
| 1 | Filename embedded timestamp | **NONE** | Files without parseable timestamps are **excluded** |
| 2 | N/A | N/A | No mtime fallback - fail closed |

### Supported Patterns

| Pattern | Regex | Example |
|---------|-------|---------|
| `MM-DD-YYYY-HH-MM_*` | `(\d{2})-(\d{2})-(\d{4})-(\d{2})-(\d{2})` | `12-28-2025-12-00_REPORT.md` |
| `TASK-YYYY-MM-DD_*` | `TASK-(\d{4})-(\d{2})-(\d{2})` | `TASK-2025-12-30-001.json` |

### Fallback Behavior

- **Filesystem mtime**: NOT used as fallback
- **Excluded files**: Remain in place, logged in receipt
- **No silent classification**: If timestamp cannot be parsed, file is excluded with reason

---

## Digest Semantics Clarification

### ⚠️ Misunderstanding Prevented

**The v1.0 statement "Pre and post digest match" was misleading.**

**Why?** The **tree digest** (path-based file listing) necessarily changes after file moves. Only the **content integrity** (file hashes) should remain unchanged.

### Two Separate Verifications

| Verification | What It Checks | Expected Result |
|--------------|----------------|-----------------|
| **Content Integrity** | SHA256 hash of file contents unchanged | ✅ PASS (content identical) |
| **Tree Digest** | Files exist at new paths after moves | ✅ PASS (paths updated) |

### Receipts Structure

```
PRE_DIGEST.json     → Tree state BEFORE moves (75 files at old paths)
POST_DIGEST.json    → Tree state AFTER moves (75 files at new paths)
INBOX_EXECUTION.json → Contains:
  - content_integrity.verdict = "PASS"
  - tree_digest.verdict = "PASS"
  - Note: tree_digest CHANGES (expected, not a bug)
```

---

## Summary

### Files Moved

| Metric | Count |
|--------|-------|
| Total files classified | 70 |
| Files successfully moved | 70 |
| Files excluded (no timestamp) | 5 |
| Conflicts resolved | 1 |

### Final Structure

```
INBOX/
├── 2025-12/
│   ├── Week-01/     (27 files - dates Dec 29-31, 2025)
│   └── Week-52/     (40 files - dates Dec 23-28, 2025)
├── 2026-01/
│   └── Week-01/     (3 files - dates Jan 1-2, 2026)
├── agents/Local Models/ (unchanged)
├── INBOX.md         (unchanged)
├── LEDGER.yaml      (unchanged)
└── inbox_normalize.py (unchanged)
```

### Governance Compliance

| Check | Status | Notes |
|-------|--------|-------|
| Determinism | ✅ PASS | Same timestamps → same paths |
| Reversibility | ✅ PASS | RESTORE_PROOF.json generated |
| No Content Modification | ✅ PASS | Content hashes verified |
| Content Integrity | ✅ PASS | All 70 file hashes match |
| Tree State | ✅ PASS | Files at new paths |
| No Data Loss | ✅ PASS | 75 files accounted for |
| Purity Scan | ✅ PASS | No temp artifacts |
| Fail-Closed | ✅ PASS | 5 files excluded (no parseable timestamp) |

---

## Receipt Files

All receipts in `LAW/CONTRACTS/_runs/`:

| Receipt | Description |
|---------|-------------|
| `INBOX_DRY_RUN.json` | Classification plan with explicit schema |
| `INBOX_EXECUTION.json` | Execution results with digest semantics clarified |
| `PRE_DIGEST.json` | Tree state before execution |
| `POST_DIGEST.json` | Tree state after execution |
| `PURITY_SCAN.json` | Temp file scan |
| `RESTORE_PROOF.json` | Reverse moves for rollback |

---

## Rollback Instructions

Execute reverse moves from `RESTORE_PROOF.json`:

```bash
# Example
mv "INBOX/2025-12/Week-01/report.md" "INBOX/reports/12-28-2025-12-00_report.md"
```

---

## References

- ISO 8601 Week Date: https://en.wikipedia.org/wiki/ISO_week_date
- INBOX Policy: `LAW/CANON/INBOX_POLICY.md`
