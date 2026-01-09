---
name: canonical-doc-enforcer
description: "Enforce canonical filename and metadata standards for ALL markdown documentation across the repository."
version: "0.1.0"
status: "Active"
required_canon_version: ">=0.1.0"
---

# Canonical Document Enforcer Skill

**Purpose:** Enforce canonical filename and metadata standards for ALL markdown documentation across the repository.

## Scope

This skill validates and fixes:
- **All `.md` files** in the repository (except exempted paths)
- Filename format: `MM-DD-YYYY-HH-MM_DESCRIPTIVE_TITLE.md`
- YAML frontmatter with required fields
- Content hash placement and validity
- Timestamp consistency

## Exemptions

The following paths are **EXEMPT** from canonical enforcement:
- `LAW/CANON/*.md` - Canon files (different governance)
- `CAPABILITY/SKILLS/*/SKILL.md` - Skill manifests
- `CAPABILITY/TESTBENCH/**/*.md` - Test fixtures
- `LAW/CONTRACTS/fixtures/**/*.md` - Test data
- `README.md`, `AGENTS.md`, `CHANGELOG.md` - Root documentation
- `.github/**/*.md` - GitHub metadata
- `BUILD/**/*.md` - User workspace

## Operations

### 1. Validate
Scans repository for non-canonical documents and reports violations.

**Usage:**
```bash
python run.py --mode validate
```

**Output:**
- List of files with violations
- Specific violation types per file
- Suggested fixes

### 2. Fix
Automatically renames and updates files to canonical format.

**Usage:**
```bash
python run.py --mode fix [--dry-run]
```

**Actions:**
- Renames files to `MM-DD-YYYY-HH-MM_TITLE.md`
- Adds/updates YAML frontmatter
- Computes and inserts content hash
- Preserves file content

### 3. Report
Generates compliance report for the repository.

**Usage:**
```bash
python run.py --mode report --output INBOX/reports/
```

## Required Fields

All canonical documents MUST have:

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

## Validation Rules

1. **Filename Pattern:** `^\d{2}-\d{2}-\d{4}-\d{2}-\d{2}_.+\.md$`
2. **Title Format:** ALL_CAPS_WITH_UNDERSCORES
3. **YAML Presence:** Lines 1-N start with `---`
4. **Hash Placement:** Immediately after closing `---`
5. **Hash Validity:** SHA256 matches content (excluding YAML and hash line)
6. **Timestamp Match:** Filename timestamp == YAML `created` field
7. **UUID Format:** RFC 4122 UUID v4

## Exit Codes

- `0` - All documents canonical
- `1` - Violations found (validate mode)
- `2` - Fix operation failed
- `3` - Invalid arguments

## Integration

### Pre-commit Hook
```bash
# In .githooks/pre-commit
python CAPABILITY/SKILLS/governance/canonical-doc-enforcer/run.py --mode validate --staged-only
```

### CI/CD
```bash
# In .github/workflows/governance.yml
- name: Validate Canonical Docs
  run: python CAPABILITY/SKILLS/governance/canonical-doc-enforcer/run.py --mode validate
```

## Examples

### Validate Repository
```bash
python run.py --mode validate
# Output:
# [VIOLATION] MEMORY/ARCHIVE/roadmaps/AGS_ROADMAP_3.3.18.md
#   - Invalid filename (missing timestamp prefix)
#   - Missing YAML frontmatter
#   - Missing content hash
```

### Fix Non-Canonical File
```bash
python run.py --mode fix --file "MEMORY/ARCHIVE/roadmaps/AGS_ROADMAP_3.3.18.md"
# Output:
# [RENAMED] AGS_ROADMAP_3.3.18.md -> 01-05-2026-12-45_AGS_ROADMAP_3_3_18.md
# [UPDATED] Added YAML frontmatter
# [UPDATED] Added content hash
```

### Dry Run
```bash
python run.py --mode fix --dry-run
# Output:
# [DRY-RUN] Would rename: AGS_ROADMAP_3.3.18.md -> 01-05-2026-12-45_AGS_ROADMAP_3_3_18.md
# [DRY-RUN] Would add YAML frontmatter
# [DRY-RUN] Would add content hash
```

## Receipts

All operations emit receipts to `LAW/CONTRACTS/_runs/canonical-doc-enforcer/`:
- `validate_receipt.json` - Validation results
- `fix_receipt.json` - Fix operation results
- `report_receipt.json` - Compliance report data

## Related

- `LAW/CANON/DOCUMENT_POLICY.md` - Canonical document policy
- `CAPABILITY/TOOLS/utilities/rename_canon.py` - Legacy renaming tool
- `CAPABILITY/SKILLS/inbox/inbox-report-writer/` - INBOX-specific tooling
