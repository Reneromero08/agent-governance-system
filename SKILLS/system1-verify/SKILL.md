# System 1 Verify Skill

## Status
Proposed

**Version:** 1.0.0

## Purpose
Ensures system1.db is in sync with repository state and verifies system1 indexer is working correctly.

## Description
This skill runs verification checks against the System 1 database to detect:
1. Missing indexed files
2. Stale (orphaned) database entries
3. Hash mismatches between indexed files and actual CANON content

## Usage
Run from repository root:
```bash
ags run --skill system1-verify
```

## Fixtures
### fixture_01_index_sync.json
Tests that core CANON files are present in system1.db.

Expected:
- Success: true
- warnings: []
- missing_files: []
- stale_files: []
- hash_mismatches: []

### fixture_02_no_orphans.json
Tests that there are no orphaned entries in system1.db (entries without corresponding CANON files).

Expected:
- Success: true
- warnings: []
- orphaned_count: 0

### fixture_03_hashes_match.json
Tests that content hashes in system1.db match the actual CANON file content on disk.

Expected:
- Success: true
- warnings: []
- mismatches: []

## Dependencies
- system1.db (CORTEX/system1.db)
- CANON directory files
