---
uuid: 7d071ad1-4a0c-4a24-965a-2dbaa11f8fe7
title: 01-06-2026-12-17 Canonical Document Enforcement Implementation
section: report
bucket: INBOX/reports
author: System
priority: Medium
created: 2026-01-06 12:19
modified: 2026-01-06 13:09
status: Active
summary: Document summary
tags: []
---
<!-- CONTENT_HASH: b98ceda55d8081da668039f8afe6476e946c90346572d9b348dd7ac037cb3b35 -->
# Canonical Document Enforcement - Implementation Summary

**Date:** 2026-01-06
**Status:** COMPLETE
**Scope:** Repo-wide canonical format enforcement for ALL markdown documentation

---

## What Was Built

### 1. Canonical Document Enforcer Skill
**Location:** `CAPABILITY/SKILLS/governance/canonical-doc-enforcer/`

**Files Created:**
- `SKILL.md` - Skill manifest and documentation
- `run.py` - Implementation (validate, fix, report modes)

**Capabilities:**
- **Validate:** Scan repository for non-canonical documents
- **Fix:** Automatically rename and add metadata to files
- **Report:** Generate compliance reports

### 2. Updated Canon Policy
**Location:** `LAW/CANON/DOCUMENT_POLICY.md` (renamed from INBOX_POLICY.md)

**Changes:**
- Expanded scope from INBOX-only to **repo-wide**
- Applies to: INBOX/, MEMORY/ARCHIVE/, LAW/CONTEXT/, NAVIGATION/, all docs
- Maintains exemptions for: LAW/CANON/, SKILL.md files, test fixtures, etc.

### 3. Canonical Format Requirements

**Filename:**
```
MM-DD-YYYY-HH-MM_DESCRIPTIVE_TITLE.md
```

**YAML Frontmatter (Required):**
```yaml
---
uuid: "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
title: "Human Readable Title"
section: report|research|roadmap|guide|archive
bucket: "category/subcategory"
author: "System|Antigravity|Human Name"
priority: High|Medium|Low
created: "YYYY-MM-DD HH:MM"
modified: "YYYY-MM-DD HH:MM"
status: "Draft|Active|Complete|Archived"
summary: "One-line summary"
tags: [tag1, tag2, tag3]
hashtags: ["#category", "#topic"]
---
<!-- CONTENT_HASH: <sha256> -->
```

---

## Usage

### Validate Repository
```bash
python CAPABILITY/SKILLS/governance/canonical-doc-enforcer/run.py --mode validate
```

### Fix Non-Canonical Files
```bash
# Dry run (show what would happen)
python CAPABILITY/SKILLS/governance/canonical-doc-enforcer/run.py --mode fix --dry-run

# Actually fix
python CAPABILITY/SKILLS/governance/canonical-doc-enforcer/run.py --mode fix
```

### Fix Specific File
```bash
python CAPABILITY/SKILLS/governance/canonical-doc-enforcer/run.py --mode fix --file "path/to/file.md"
```

### Generate Compliance Report
```bash
python CAPABILITY/SKILLS/governance/canonical-doc-enforcer/run.py --mode report --output INBOX/reports/
```

---

## What Was Demonstrated

### Test Case: AGS_ROADMAP_3.3.18.md

**Before:**
- Filename: `AGS_ROADMAP_3.3.18.md` (non-canonical)
- No YAML frontmatter
- No content hash
- Located in: `MEMORY/ARCHIVE/roadmaps/`

**After Fix:**
- Filename: `01-05-2026-12-45_AGS_ROADMAP_3_3_18.md` ✅
- YAML frontmatter added with all required fields ✅
- Content hash computed and inserted ✅
- All violations resolved ✅

**Validation Output:**
```
[VIOLATION] MEMORY\ARCHIVE\roadmaps\01-05-2026-12-45_AGS_ROADMAP_3_3_18.md
  - Missing required field: uuid
  - Missing required field: section
  - Missing required field: bucket
  - Missing required field: author
  - Missing required field: priority
  - Missing required field: created
  - Missing required field: modified
  - Missing required field: status
  - Missing required field: summary
  - Missing required field: tags
  - Missing required field: hashtags
  - Missing content hash
```

**Fix Output:**
```
[FIXED] MEMORY\ARCHIVE\roadmaps\01-05-2026-12-45_AGS_ROADMAP_3_3_18.md
  - Update YAML frontmatter
  - Update content hash
```

---

## Integration Points

### Pre-commit Hook (Recommended)
Add to `.githooks/pre-commit`:
```bash
# Validate canonical document format
python CAPABILITY/SKILLS/governance/canonical-doc-enforcer/run.py --mode validate --staged-only
if [ $? -ne 0 ]; then
    echo "ERROR: Non-canonical documents detected. Run fix mode to resolve."
    exit 1
fi
```

### CI/CD (Recommended)
Add to `.github/workflows/governance.yml`:
```yaml
- name: Validate Canonical Documents
  run: python CAPABILITY/SKILLS/governance/canonical-doc-enforcer/run.py --mode validate
```

---

## Exemptions

The following paths are **EXEMPT** from canonical enforcement:
- `LAW/CANON/*.md` - Canon files (different governance)
- `CAPABILITY/SKILLS/*/SKILL.md` - Skill manifests
- `CAPABILITY/TESTBENCH/**/*.md` - Test fixtures
- `LAW/CONTRACTS/fixtures/**/*.md` - Test data
- `README.md`, `AGENTS.md`, `CHANGELOG.md` - Root documentation
- `.github/**/*.md` - GitHub metadata
- `BUILD/**/*.md` - User workspace

---

## Exit Codes

- `0` - Success (all documents canonical)
- `1` - Violations found (validate mode)
- `2` - Fix operation failed
- `3` - Invalid arguments

---

## Next Steps

1. **Run repo-wide validation:**
   ```bash
   python CAPABILITY/SKILLS/governance/canonical-doc-enforcer/run.py --mode validate
   ```

2. **Fix all violations:**
   ```bash
   python CAPABILITY/SKILLS/governance/canonical-doc-enforcer/run.py --mode fix
   ```

3. **Update pre-commit hook** to enforce on new commits

4. **Update CHANGELOG.md** with this implementation

5. **Update roadmap** to mark canonical enforcement as complete

---

## Receipts

All fix operations emit receipts to:
```
LAW/CONTRACTS/_runs/canonical-doc-enforcer/fix_receipt.json
```

---

**Implementation Status:** ✅ COMPLETE
**Coverage:** Repo-wide (all markdown documentation)
**Enforcement:** Skill-based (validate, fix, report modes)
**Policy:** LAW/CANON/DOCUMENT_POLICY.md
