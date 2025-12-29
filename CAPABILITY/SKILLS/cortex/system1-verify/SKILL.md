# System 1 Verify Skill

**Version:** 1.0.0

**Status:** Experimental

**Required_Canon_Version:** >=2.0.0

**Date:** 2025-12-28
**Confidence:** Medium
**Impact:** Low
**Tags:** [verification, cortex, system1]

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

## Dependencies
- system1.db (CORTEX/system1.db)
- CANON directory files
