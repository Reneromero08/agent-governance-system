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

All documents in `INBOX/` MUST include:

1. **Content Hash**
   - SHA-256 hash of document content
   - Purpose: Detect modifications and verify integrity
   - Format: `<!-- CONTENT_HASH: <sha256> -->` in Markdown files
   - Location: First line or metadata section

2. **Document Metadata**
   - Title
   - Date created
   - Author/agent identity
   - Status (Draft, Ready for Review, Archived)
   - Related resources (links to canon, roadmaps, etc.)

3. **Cortex References**
   - When applicable, use @Symbol references instead of full content
   - Format: `@C:{hash_short}` referencing cortex entries
   - Reduces token usage and keeps INBOX lightweight

## Examples

### Implementation Report

```markdown
# Cassette Network Implementation Report

<!-- CONTENT_HASH: a7b3c5d9e8f2a1b4c8e5d6a7b3e9f8a4c5d -->
**Date:** 2025-12-28
**Status:** COMPLETE
**Agent:** opencode@agent-governance-system | 2025-12-28

...
```

### Research Document

```markdown
# Cassette Network Architecture Research

<!-- CONTENT_HASH: 8f2d3b4e1a9c5d6e7f8a2b1c9d4e5f6a8b7c3d2e -->
**Date:** 2025-12-28
**Researcher:** opencode

...

## Required Context: @Cortex

When INBOX documents reference canon or indexed content, they MUST use @Symbol references from cortex:

```markdown
This implementation aligns with @C:ab5e61a8 (Cassette Protocol) and
extends @C:ce89a30e (Network Hub) as defined in @C:d3f2b8a7 (ADR-030).
```

### Finding Cortex References

Use `TOOLS/cortex.py` to resolve @Symbols:

```bash
python TOOLS/cortex.py resolve @C:ab5e61a8
```

## Governance Enforcement

The pre-commit hook (`SKILLS/canon-governance-check/scripts/pre-commit`) will verify:

1. ✅ All human-readable documents are in `INBOX/` (not scattered across repo)
2. ✅ All INBOX documents contain content hash
3. ✅ Documents use @Symbol references when appropriate (not duplicating canon content)
4. ✅ INBOX structure is maintained (reports/, research/, roadmaps/, etc.)

## Exceptions

The following are EXEMPT from INBOX policy:

1. **Canon documents** (CANON/*) - These ARE the source of truth
2. **Generated artifacts** (CORTEX/_generated/*, CONTRACTS/_runs/*) - System outputs
3. **Code implementations** (TOOLS/*.py, SKILLS/*/run.py) - Implementation files
4. **Test fixtures** (CONTRACTS/fixtures/*, CATALYTIC-DPT/TESTBENCH/*) - Test data
5. **Skill manifests** (SKILLS/*/SKILL.md) - These stay with their skills
6. **Context records** (CONTEXT/decisions/*, CONTEXT/preferences/*) - Append-first storage
7. **Build outputs** (BUILD/*) - User workspace outputs

## Enforcement

### Pre-commit Hook

When committing changes, the governance check will:

1. Scan for new `.md` files outside allowed locations
2. Verify files in `INBOX/` have content hashes
3. Verify @Symbol references are valid (if present)
4. Report violations:
   - ERROR: Human-readable document outside INBOX
   - ERROR: INBOX document missing content hash
   - ERROR: Invalid @Symbol reference

### Violation Handling

If violations are found:
- **Block commit** with clear error message
- Suggest moving file to `INBOX/` and adding hash
- Example: `ERROR: INBOX/required-file.md found outside INBOX/, move to INBOX/reports/ and add <!-- CONTENT_HASH: ... -->`

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
# 1. Write report to INBOX/reports/
report_path = "INBOX/reports/cassette-network-implementation-report.md"

# 2. Include content hash
content_hash = compute_sha256(report_content)
report_content = f"<!-- CONTENT_HASH: {content_hash} -->\n\n{report_body}"

# 3. Use @Symbol references for canon content
# Instead of duplicating CONTRACT.md text, use: @C:d3f2b8a7

# 4. Save
Path(report_path).write_text(report_content)
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
python TOOLS/cortex.py resolve @C:ab5e61a8
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
