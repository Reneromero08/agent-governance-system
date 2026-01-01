<!-- CONTENT_HASH: fc8fd84e8d667eeecccc3831a6fa722d52557bfe69411868bb694ff33b8ff335 -->

# Catalytic Chat Contract

**Version:** 1.0
**Status:** Phase 0 Complete
**Roadmap Phase:** Phase 0 — Freeze scope and interfaces

## Purpose

Defines the immutable contract for Catalytic Chat substrate. All implementations must honor these schemas, constraints, and error policies without deviation.

## Core Objects

### Section

Canonical content unit extracted from source files.

```json
{
  "section_id": "sha256_hash",
  "file_path": "string",
  "heading_path": ["heading1", "heading2"],
  "line_start": 0,
  "line_end": 42,
  "content_hash": "sha256_hash"
}
```

**Constraints:**
- `section_id`: SHA-256 of `file_path:line_start:line_end:content_hash`
- `file_path`: Absolute path relative to repo root
- `heading_path`: Array of heading names from root to this section (empty for file-level)
- `line_start` ≤ `line_end`
- `content_hash`: SHA-256 of the exact content (normalized line endings)

**Fail-closed rules:**
- Missing file → FAIL
- Invalid line range → FAIL
- Content hash mismatch → FAIL

---

### Symbol

Compact reference to a Section or file.

```json
{
  "symbol_id": "@NAMESPACE/NAME",
  "target_type": "SECTION|FILE|HEADING",
  "target_ref": "section_id_or_file_path",
  "default_slice_policy": "lines[a:b]"
}
```

**Constraints:**
- `symbol_id`: Must start with `@`
- `target_type`: One of `SECTION`, `FILE`, `HEADING`
- `target_ref`: Valid `section_id` or `file_path`
- `default_slice_policy`: Valid slice specification

**Fail-closed rules:**
- Invalid symbol format → FAIL
- Unknown `target_ref` → FAIL
- Unresolvable target → FAIL

**Namespace conventions:**
- `@CANON/...` → Canon documents
- `@CONTRACTS/...` → Contract specifications
- `@TOOLS/...` → Tool documentation
- `@SKILLS/...` → Skill documentation

---

### Message

Model output requesting work with explicit resource references.

```json
{
  "intent": "string",
  "refs": ["@Symbol1", "@Symbol2"],
  "ops": [
    {
      "type": "READ|WRITE|EXECUTE",
      "target": "@Symbol",
      "params": {}
    }
  ],
  "budgets": {
    "max_symbols": 10,
    "max_sections": 5,
    "max_bytes_expanded": 10000,
    "max_expands_per_step": 3
  },
  "required_outputs": ["output1", "output2"]
}
```

**Constraints:**
- `intent`: Non-empty string
- `refs`: Array of valid symbol_ids
- `ops`: Array of operation objects
- `budgets`: All integer values ≥ 0
- `required_outputs`: Array of output identifiers

**Fail-closed rules:**
- Empty `intent` → FAIL
- Invalid symbol in `refs` → FAIL
- Missing `budgets` field → FAIL
- Budget breach during execution → FAIL

---

### Expansion

Bounded content retrieval for a Symbol or Section.

```json
{
  "run_id": "uuid",
  "symbol_or_section_id": "@Symbol or section_id",
  "slice": "lines[0:100]",
  "content_hash": "sha256_hash",
  "payload_ref": "path_or_hash"
}
```

**Constraints:**
- `run_id`: UUID for the execution run
- `symbol_or_section_id`: Valid symbol_id or section_id
- `slice`: Valid slice specification
- `content_hash`: SHA-256 of expanded content
- `payload_ref`: Reference to stored content

**Slice forms (canonical):**
- `lines[a:b]` - Line range (0-indexed, inclusive start, exclusive end)
- `chars[a:b]` - Character range
- `head(n)` - First n lines
- `tail(n)` - Last n lines

**Fail-closed rules:**
- Invalid slice syntax → FAIL
- Slice exceeds bounds → FAIL
- Content hash mismatch → FAIL
- Requested `slice=ALL` → FAIL (unbounded expansion forbidden)

---

### Receipt

Immutable record of execution step.

```json
{
  "run_id": "uuid",
  "step_id": "step-uuid",
  "expanded": [
    {
      "symbol_or_section_id": "@Symbol",
      "slice": "lines[0:100]",
      "content_hash": "hash"
    }
  ],
  "actions": [
    {
      "type": "READ|WRITE|EXECUTE",
      "target": "@Symbol",
      "status": "SUCCESS|FAILURE",
      "result": {}
    }
  ],
  "outputs": {
    "output1": "value"
  },
  "status": "SUCCESS|FAILURE|PARTIAL"
}
```

**Constraints:**
- `run_id`: UUID matching parent run
- `step_id`: UUID for this step
- `expanded`: Array of Expansion records
- `actions`: Array of action results
- `outputs`: Map of output identifier to value
- `status`: One of `SUCCESS`, `FAILURE`, `PARTIAL`

**Fail-closed rules:**
- Missing `run_id` or `step_id` → FAIL
- Incomplete `expanded` records → FAIL
- Missing `required_outputs` from Message → FAIL
- Receipt must be append-only → IMMUTABLE after creation

---

## Budget Definitions

Maximum resources per message step:

```json
{
  "max_symbols": 10,
  "max_sections": 5,
  "max_bytes_expanded": 10000,
  "max_expands_per_step": 3
}
```

**Enforcement:**
- Count symbols resolved (de-duplicate)
- Count unique sections expanded
- Sum bytes of all expanded content
- Count expansion operations
- Any budget breach → FAIL immediately

---

## Error Policy

**Fail-closed approach:** All errors are hard failures. No graceful degradation.

**Error categories:**

1. **Invalid input**
   - Malformed JSON → FAIL
   - Missing required fields → FAIL
   - Invalid data types → FAIL

2. **Resolution errors**
   - Unknown symbol → FAIL
   - Unknown section → FAIL
   - Invalid slice → FAIL

3. **Budget violations**
   - Exceeds max_symbols → FAIL
   - Exceeds max_sections → FAIL
   - Exceeds max_bytes_expanded → FAIL
   - Exceeds max_expands_per_step → FAIL

4. **Execution errors**
   - Expansion failed → FAIL
   - Action failed → FAIL
   - Missing required output → FAIL

**Error handling:**
- All errors are logged with full context
- Receipts record failure status and reason
- No silent failures
- No partial success for critical steps

---

## Canonical Sources

Folders and file types considered canonical for Section extraction:

```
LAW/CANON/*.md
LAW/CONTEXT/**/*.md
SKILLS/**/*.md
TOOLS/**/*.md
CONTRACTS/**/*.md
CORTEX/db/**/*.sql
```

**Future extensions:**
- Code files (`.py`, `.js`, `.ts`)
- Contract fixtures (`.json`)
- Test files (as needed)

---

## Determinism Requirements

**Deterministic addressing:**
- Same inputs → same section_ids across runs
- Hash-based identifiers required (SHA-256)

**Deterministic indexing:**
- Two consecutive builds on unchanged corpus → identical SECTION_INDEX
- Content ordering stable
- Hash computation reproducible

---

## Exit Criteria (Phase 0)

- [x] CONTRACT.md exists and is referenced by roadmap
- [x] A dummy end-to-end walkthrough can be expressed using only contract objects (no prose)

---

## References

- Roadmap: `CAT_CHAT_ROADMAP_V1.md`
- Phase 0 checklist: Phase 0 — Freeze scope and interfaces
- Related: CANON/CATALYTIC_COMPUTING.md
