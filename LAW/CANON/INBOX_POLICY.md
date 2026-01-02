<!-- CONTENT_HASH: 007a53ac9558cefec29b758d74d77fc428cdf08a7aa97e15fc37cfaf05d2527b -->

# INBOX Policy

**Purpose:** All documentation, research, reports, and roadmaps intended for human review ("god mode") must be stored in a centralized INBOX folder for consistent discoverability.

---

## Policy Statement

All non-system-generated documents that require human review or attention must be placed in the repository root `INBOX/` directory.

## Required Content Types

All of the following MUST be stored in `INBOX/`:

1. **Implementation Reports**
   - All implementation completion reports
   - Session reports and documentation
   - Testing results and validation reports

2. **Research Documents**
   - Architecture research and findings
   - External research (arXiv, academic papers)
   - Experimental results and analysis

3. **Roadmaps and Planning**
   - Roadmap documents (draft and active)
   - Planning documents
   - Feature proposals and designs

4. **Decisions and Context**
   - ADRs (Architecture Decision Records)
   - Meeting notes and discussion summaries
   - Policy proposals and reviews

5. **Other Human-Readable Documentation**
   - User-facing guides and tutorials
   - Status reports and summaries
   - Any document requiring human attention

## INBOX Structure

```
INBOX/
├── reports/              # Implementation reports
├── research/             # Research findings and analysis
├── roadmaps/            # Planning and roadmap documents
├── decisions/            # ADRs and policy discussions
├── summaries/            # Session and status summaries
└── ARCHIVE/              # Processed items (keep for history)
```

## Document Requirements

All documents in `INBOX/` MUST follow these strict formatting rules:

### 1. Filename Format (MANDATORY)
**Format:** `MM-DD-YYYY-HH-MM_DESCRIPTIVE_TITLE.md`

**Rules:**
- Timestamp uses system time at document creation
- Title must be ALL CAPS with underscores (no spaces)
- Title should be descriptive and human-readable
- Examples:
  - `01-01-2026-11-37_SYSTEM_POTENTIAL_REPORT.md`
  - `12-28-2025-14-22_CASSETTE_NETWORK_IMPLEMENTATION.md`
  - `12-29-2025-09-15_SEMANTIC_CORE_PHASE_ONE_COMPLETE.md`

**Rationale:**
- Chronological sorting by filename
- Instant timestamp visibility
- No filename collisions
- Easy grep/search by date range

### 2. Document Header (MANDATORY)
**Format:** YAML frontmatter followed by content hash

```yaml
---
uuid: "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
title: "Descriptive Title (Human Readable)"
section: report|research|roadmap|guide
bucket: "primary_category/subcategory"
author: "System|Antigravity|Human Name"
priority: High|Medium|Low
created: "YYYY-MM-DD HH:MM"
modified: "YYYY-MM-DD HH:MM"
status: "Draft|Ready for Review|Archived|Complete"
summary: "One-line summary of document purpose and content"
tags: [tag1, tag2, tag3]
hashtags: ["#category", "#topic", "#status"]
---
<!-- CONTENT_HASH: <sha256_of_content_after_yaml> -->
```

**Rules:**
- YAML block MUST be first (lines 1-N)
- Content hash MUST be immediately after YAML (line N+1)
- Hash is computed on content AFTER the hash line (not including YAML or hash line itself)
- All fields are REQUIRED (no optional fields)
- Timestamps use `YYYY-MM-DD HH:MM` format

**Field Specifications:**
- **uuid**: RFC 4122 compliant UUID v4 (generated once, never changes)
- **bucket**: Hierarchical category path (e.g., "implementation/phase1", "research/architecture")
- **tags**: Machine-readable tags (lowercase, underscores)
- **hashtags**: Human-readable hashtags with # prefix (for cross-referencing and discovery)

### 3. Cortex References
- When applicable, use @Symbol references instead of full content
- Format: `@C:{hash_short}` referencing cortex entries
- Reduces token usage and keeps INBOX lightweight

## Examples

### Implementation Report

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
summary: "Implementation report for Cassette Network Phase 1 with receipt chains and trust policies"
tags: [cassette, network, implementation]
hashtags: ["#cassette", "#phase1", "#complete"]
---
<!-- CONTENT_HASH: a7b3c5d9e8f2a1b4c8e5d6a7b3e9f8a4c5d -->

# Cassette Network Implementation Report

## Executive Summary
...
```

### Research Document

**Filename:** `12-28-2025-09-15_CASSETTE_ARCHITECTURE_RESEARCH.md`

```markdown
---
uuid: "8f2d3b4e-1a9c-5d6e-7f8a-2b1c9d4e5f6a"
title: "Cassette Network Architecture Research"
section: research
bucket: "research/architecture"
author: "Antigravity"
priority: Medium
created: "2025-12-28 09:15"
modified: "2025-12-28 09:15"
status: "Draft"
summary: "Research findings on distributed cassette architecture and semantic indexing strategies"
tags: [cassette, architecture, research]
hashtags: ["#research", "#cassette", "#architecture"]
---
<!-- CONTENT_HASH: 8f2d3b4e1a9c5d6e7f8a2b1c9d4e5f6a8b7c3d2e -->

# Cassette Network Architecture Research

## Overview
...

## Required Context: @Cortex

When INBOX documents reference canon or indexed content, they MUST use @Symbol references from cortex:

```markdown
This implementation aligns with @C:ab5e61a8 (Cassette Protocol) and
extends @C:ce89a30e (Network Hub) as defined in @C:d3f2b8a7 (ADR-030).
```

### Finding Cortex References

Use `CAPABILITY/TOOLS/cortex_query.py` (or equivalent) to resolve @Symbols:

```bash
python CAPABILITY/TOOLS/cortex_query.py resolve @C:ab5e61a8
```

## Governance Enforcement

The pre-commit hook (`.githooks/pre-commit` or `CAPABILITY/SKILLS/governance/canon-governance-check/scripts/pre-commit`) will verify:

1. ✅ **Filename Format:** All INBOX files match `MM-DD-YYYY-HH-MM_TITLE.md` pattern
2. ✅ **YAML Frontmatter:** All INBOX documents contain valid YAML with ALL required fields
3. ✅ **UUID Validity:** UUID field is RFC 4122 compliant UUID v4
4. ✅ **Bucket Format:** Bucket follows hierarchical path format (category/subcategory)
5. ✅ **Hashtags Format:** Hashtags are properly formatted with # prefix
6. ✅ **Content Hash:** Hash line exists immediately after YAML frontmatter
7. ✅ **Hash Validity:** Content hash matches actual content (excluding YAML and hash line)
8. ✅ **Timestamp Consistency:** Filename timestamp matches YAML `created` field
9. ✅ **INBOX Structure:** Documents are in correct subdirectories (reports/, research/, roadmaps/)
10. ✅ **@Symbol References:** Cortex references are valid when present

## Exceptions

The following are EXEMPT from INBOX policy:

1. **Canon documents** (`LAW/CANON/*`) - These ARE the source of truth
2. **Generated artifacts** (`NAVIGATION/CORTEX/_generated/*`, `LAW/CONTRACTS/_runs/*`) - System outputs
3. **Code implementations** (`CAPABILITY/TOOLS/*.py`, `CAPABILITY/SKILLS/*/run.py`) - Implementation files
4. **Test fixtures** (`LAW/CONTRACTS/fixtures/*`, `CAPABILITY/TESTBENCH/*`) - Test data
5. **Skill manifests** (`CAPABILITY/SKILLS/*/SKILL.md`) - These stay with their skills
6. **Context records** (`LAW/CONTEXT/decisions/*`, `LAW/CONTEXT/preferences/`) - Append-first storage
7. **Build outputs** (BUILD/*) - User workspace outputs
8. **INBOX.md** - The index file itself

## Enforcement

### Pre-commit Hook

When committing changes, the governance check will:

1. Scan for new `.md` files in `INBOX/`
2. Validate filename matches `MM-DD-YYYY-HH-MM_*.md` pattern
3. Parse YAML frontmatter and verify all required fields exist
4. Verify content hash exists and is valid
5. Check timestamp consistency between filename and YAML
6. Verify @Symbol references are valid (if present)
7. Report violations:
   - ERROR: Invalid filename format (must be MM-DD-YYYY-HH-MM_TITLE.md)
   - ERROR: Missing or invalid YAML frontmatter
   - ERROR: Missing content hash after YAML
   - ERROR: Content hash mismatch
   - ERROR: Timestamp mismatch between filename and YAML
   - ERROR: Invalid @Symbol reference

### Violation Handling

If violations are found:
- **Block commit** with clear error message
- Suggest correct format and required fields
- Example: `ERROR: INBOX/reports/my-report.md has invalid filename. Must be: MM-DD-YYYY-HH-MM_TITLE.md`

## Rationale

### Why INBOX?

- **Discoverability:** Single location for all human-facing documentation
- **UX Consistency:** Always know where to find reports, research, decisions
- **Reduced Cognitive Load:** Don't hunt for documents across entire repo
- **Governance:** Central location for monitoring and cleanup

### Why Content Hashes?

- **Integrity:** Detect if documents are modified after signing
- **Verification:** Confirm report hasn't been tampered with
- **Traceability:** Track document versions and changes

### Why @Symbol References?

- **Token Efficiency:** Cortex already indexed; don't duplicate content
- **Maintainability:** Updates to canon automatically reflected in INBOX
- **Consistency:** Single source of truth (CORTEX) with lightweight references

## Usage Examples

### For Agents

When creating implementation reports:

```python
from datetime import datetime
import hashlib
import uuid

# 1. Get current timestamp
now = datetime.now()
timestamp = now.strftime("%m-%d-%Y-%H-%M")
yaml_timestamp = now.strftime("%Y-%m-%d %H:%M")

# 2. Generate UUID (once per document, never changes)
doc_uuid = str(uuid.uuid4())

# 3. Create filename
title = "CASSETTE_NETWORK_IMPLEMENTATION"
filename = f"{timestamp}_{title}.md"
report_path = f"INBOX/reports/{filename}"

# 4. Define metadata
bucket = "implementation/cassette_network"
hashtags = ["#cassette", "#phase1", "#complete"]
tags = ["cassette", "network", "implementation"]

# 5. Build YAML frontmatter
yaml_header = f"""---
uuid: "{doc_uuid}"
title: "Cassette Network Implementation Report"
section: "report"
bucket: "{bucket}"
author: "System"
priority: "High"
created: "{yaml_timestamp}"
modified: "{yaml_timestamp}"
status: "Complete"
summary: "Implementation report for Cassette Network Phase 1 with receipt chains and trust policies"
tags: [{', '.join(tags)}]
hashtags: [{', '.join(f'"{h}"' for h in hashtags)}]
---"""

# 6. Build content body (use @Symbol references for canon content)
content_body = """
# Cassette Network Implementation Report

## Executive Summary
This report documents the implementation of @C:ab5e61a8 (Cassette Protocol)...

## Implementation Details
...
"""

# 7. Compute content hash (hash the body only, not YAML or hash line)
content_hash = hashlib.sha256(content_body.encode('utf-8')).hexdigest()

# 8. Assemble final document
final_content = f"{yaml_header}\n<!-- CONTENT_HASH: {content_hash} -->\n{content_body}"

# 9. Save
Path(report_path).write_text(final_content)
```

### For Humans

When reviewing repository:

```bash
# All documents for review are in one place:
ls INBOX/reports/
ls INBOX/research/
ls INBOX/roadmaps/

# Verify integrity
grep "CONTENT_HASH:" INBOX/reports/*.md

# Resolve cortex references
python CAPABILITY/TOOLS/cortex_query.py resolve @C:ab5e61a8
```

## Migration Guide

### Existing Documents

If you have human-readable documents scattered across the repo:

1. **Identify candidates:**
   - Session reports
   - Implementation reports
   - Research documents
   - Roadmaps
   - Status summaries

2. **Move to INBOX:**
   ```bash
   mv SESSION_REPORTS/*.md INBOX/reports/
   mv ROADMAP-*.md INBOX/roadmaps/
   mv research-*.md INBOX/research/
   ```

3. **Add content hashes:**
   ```bash
   # For each file, add to top:
   # !sha256sum path/to/file.md >> INBOX/reports/file.md
   echo "<!-- CONTENT_HASH: $(sha256sum path/to/file.md | cut -d' ' -f1) -->" >> INBOX/reports/file.md
   ```

4. **Replace full content with @Symbols:**
   ```bash
   # If duplicating canon content, use cortex reference instead
   python TOOLS/cortex.py resolve @C:{hash}
   ```

## Cleanup and Maintenance

### Regular INBOX Maintenance

**Weekly:**
- Archive processed items to `INBOX/ARCHIVE/`
- Remove duplicates
- Verify all hashes are valid

**Monthly:**
- Check for outdated reports (archive if >6 months)
- Review ARCHIVE/ and remove if unnecessary
- Ensure @Symbol references still resolve

---

**Canon Version:** 2.16.0
**Required Canon Version:** >=2.16.0
