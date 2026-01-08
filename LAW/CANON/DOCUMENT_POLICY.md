<!-- CONTENT_HASH: ab0eb3ac2a2c3c604ffb6518f5cf65aa2da7f9d743fcd676a52d05cfab0d9400 -->

# Document Policy (Canonical Format)

**Purpose:** ALL markdown documentation across the repository must follow canonical filename and metadata standards for consistency, discoverability, and integrity.

---

## Policy Statement

**ALL `.md` files** in the repository (except exempted paths) MUST follow canonical format:
- Filename: `MM-DD-YYYY-HH-MM_DESCRIPTIVE_TITLE.md`
- YAML frontmatter with required fields
- Content hash for integrity verification

**Applies to:** `INBOX/`, `MEMORY/ARCHIVE/`, `LAW/CONTRACTS/_runs/REPORTS/`, document artifacts in `_runs/`

## Required Content Types (MUST be in INBOX/)

1. **Implementation Reports** — completion reports, session reports, testing results
2. **Research Documents** — architecture research, external research, experimental results
3. **Roadmaps and Planning** — roadmaps, planning docs, feature proposals
4. **Decisions and Context** — ADRs, meeting notes, policy proposals
5. **Other Human-Readable Docs** — guides, tutorials, status reports

## INBOX Structure

```
INBOX/
├── reports/      # Implementation reports
├── research/     # Research findings
├── roadmaps/     # Planning documents
├── decisions/    # ADRs and discussions
├── summaries/    # Session summaries
└── ARCHIVE/      # Processed items
```

## Document Requirements

### 1. Filename Format (MANDATORY)

**Format:** `MM-DD-YYYY-HH-MM_DESCRIPTIVE_TITLE.md`

- Timestamp: system time at creation
- Title: ALL CAPS with underscores (no spaces)
- Example: `01-07-2026-14-30_CASSETTE_NETWORK_IMPLEMENTATION.md`

### 2. Document Header (MANDATORY)

```yaml
---
uuid: "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"  # Agent session UUID
title: "Descriptive Title (Human Readable)"
section: report|research|roadmap|guide
bucket: "primary_category/subcategory"
author: "System|Antigravity|Human Name"
priority: High|Medium|Low
created: "YYYY-MM-DD HH:MM"
modified: "YYYY-MM-DD HH:MM"
status: "Draft|Ready for Review|Archived|Complete"
summary: "One-line summary of document purpose"
tags: [tag1, tag2, tag3]
---
<!-- CONTENT_HASH: <sha256_of_content_after_yaml> -->
```

**Rules:**
- YAML block MUST be first
- Content hash MUST be immediately after YAML
- Hash computed on content AFTER the hash line
- All fields REQUIRED
- **uuid**: Agent session ID (use `00000000-0000-0000-0000-000000000000` for legacy)

### 3. Cortex References

Use `@C:{hash_short}` references instead of duplicating content:
```markdown
This implements @C:ab5e61a8 (Cassette Protocol) per @C:d3f2b8a7 (ADR-030).
```

## Example Document

**Filename:** `12-28-2025-14-30_CASSETTE_NETWORK_IMPLEMENTATION.md`

```markdown
---
uuid: "a7b3c5d9-e8f2-41b4-c8e5-d6a7b3e9f8a4"
title: "Cassette Network Implementation Report"
section: report
bucket: "implementation/cassette_network"
author: "System"
priority: High
created: "2025-12-28 14:30"
modified: "2025-12-28 14:30"
status: "Complete"
summary: "Implementation report for Cassette Network Phase 1"
tags: [cassette, network, implementation]
---
<!-- CONTENT_HASH: a7b3c5d9e8f2a1b4c8e5d6a7b3e9f8a4c5d -->

# Cassette Network Implementation Report

## Executive Summary
This report documents the implementation of @C:ab5e61a8 (Cassette Protocol)...
```

## Governance Enforcement

The pre-commit hook verifies:
1. Filename matches `MM-DD-YYYY-HH-MM_TITLE.md` pattern
2. Valid YAML frontmatter with ALL required fields
3. UUID is RFC 4122 compliant
4. Content hash exists and is valid
5. Timestamp consistency (filename ↔ YAML `created`)
6. @Symbol references resolve (if present)

**Violations block commit** with specific error messages.

## Exceptions (EXEMPT from policy)

| Path | Reason |
|------|--------|
| `LAW/CANON/*` | Source of truth |
| `NAVIGATION/CORTEX/_generated/*` | System outputs |
| `LAW/CONTRACTS/_runs/*` | System outputs |
| `CAPABILITY/TOOLS/*.py` | Implementation files |
| `CAPABILITY/SKILLS/*/SKILL.md` | Skill manifests |
| `LAW/CONTEXT/decisions/*` | Append-first storage |
| `LAW/CONTEXT/preferences/*` | Append-first storage |
| `BUILD/*` | User workspace |
| `INBOX/prompts/*` | Governed by prompt policy |

## Agent Usage

```python
from datetime import datetime
import hashlib
from pathlib import Path

now = datetime.now()
timestamp = now.strftime("%m-%d-%Y-%H-%M")
yaml_ts = now.strftime("%Y-%m-%d %H:%M")
doc_uuid = get_session_id()  # or "00000000-0000-0000-0000-000000000000"

yaml_header = f"""---
uuid: "{doc_uuid}"
title: "Report Title"
section: "report"
bucket: "implementation/feature"
author: "System"
priority: "High"
created: "{yaml_ts}"
modified: "{yaml_ts}"
status: "Complete"
summary: "One-line summary"
tags: [tag1, tag2]
---"""

content_body = "# Report Title\n\n## Summary\n..."
content_hash = hashlib.sha256(content_body.encode()).hexdigest()
final = f"{yaml_header}\n<!-- CONTENT_HASH: {content_hash} -->\n{content_body}"
Path(f"INBOX/reports/{timestamp}_REPORT_TITLE.md").write_text(final)
```

## Human Usage

```bash
# Find all documents
ls INBOX/reports/ INBOX/research/ INBOX/roadmaps/

# Verify integrity
grep "CONTENT_HASH:" INBOX/reports/*.md

# Resolve cortex references
python CAPABILITY/TOOLS/cortex_query.py resolve @C:ab5e61a8
```

## Migration & Maintenance

**Migrate existing docs:**
1. Move to appropriate `INBOX/` subdirectory
2. Rename to `MM-DD-YYYY-HH-MM_TITLE.md` format
3. Add YAML frontmatter
4. Add content hash

**Maintenance:**
- Weekly: Archive processed items, verify hashes
- Monthly: Archive docs >6 months old, verify @Symbol refs

---

## Rationale

| Feature | Why |
|---------|-----|
| **INBOX/** | Single location for human-facing docs; reduces cognitive load |
| **Content hashes** | Integrity verification; detect tampering |
| **@Symbol refs** | Token efficiency; single source of truth |
| **Timestamps** | Chronological sorting; no collisions |

---

**Canon Version:** 2.16.0
