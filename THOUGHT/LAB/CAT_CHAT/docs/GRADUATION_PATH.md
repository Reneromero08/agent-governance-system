# CAT_CHAT Graduation Path (Phase B.4)

**Purpose:** Document the future path for CAT_CHAT graduation from LAB to main system.

**Status:** FUTURE REFERENCE - CAT_CHAT stays in LAB for now.

---

## Current State: LAB Isolation

CAT_CHAT currently lives in the sandbox:

```
THOUGHT/LAB/CAT_CHAT/
    _generated/
        cat_chat.db     # All CAT_CHAT state (sessions, events, symbols, etc.)
    catalytic_chat/     # Python package
    tests/
    docs/
```

**All state is local.** CAT_CHAT reads from main cassettes but does not contribute to them.

---

## What Must Stay Local

These components are instance-specific and should **never** graduate to main cassettes:

| Component | Table(s) | Why Local |
|-----------|----------|-----------|
| Sessions | `sessions` | Per-instance state |
| Session Events | `session_events` | Hash-chained, instance-specific |
| Working Set | `session_working_set` | Runtime context |
| Pointer Set | `session_pointer_set` | Runtime references |
| Expansion Cache | `expansion_cache` | Runtime cache, expires |
| Cassette Jobs | `cassette_jobs` | Execution state |
| Cassette Steps | `cassette_steps` | FSM state |
| Cassette Receipts | `cassette_receipts` | Execution proofs |
| Job Budgets | `cassette_job_budgets` | Runtime tracking |

---

## What May Graduate (Future)

When CAT_CHAT is mature and stable:

### Option A: Session Data to resident.db

If CAT_CHAT sessions should persist across instances:

| Local Table | Target | Migration |
|-------------|--------|-----------|
| `sessions` | `resident.db:sessions` | Merge with existing |
| `session_events` | `resident.db:session_events` | Append-only |

This would require:
1. Schema alignment with resident.db session tables
2. Agent ID linking (sessions belong to agents)
3. Merkle root compatibility

### Option B: New CAT_CHAT Cassette

If CAT_CHAT should become a searchable knowledge source:

```json
{
  "id": "cat_chat",
  "name": "CAT Chat Knowledge Base",
  "db_path": "NAVIGATION/CORTEX/cassettes/cat_chat.db",
  "capabilities": ["fts", "semantic_search"],
  "type": "generic",
  "mutability": "append_only"
}
```

This would require:
1. Creating `chunks` and `files` tables from `sections`
2. Creating `chunks_fts` FTS5 virtual table
3. Validating with `GenericCassette.validate_schema()`

### Option C: Keep in LAB Permanently

If CAT_CHAT should remain experimental:
- No migration needed
- State stays in `_generated/`
- Can be reset/cleared without affecting main system

---

## Schema Mapping (Future Reference)

If graduation proceeds, here's the schema mapping:

### Current CAT_CHAT Tables

**Index Layer:**
- `sections` (section_id, file_path, heading_path, content_hash, ...)
- `section_index_meta` (key, value, updated_at)
- `symbols` (symbol_id, target_type, target_ref, default_slice, ...)

**Session Layer:**
- `sessions` (session_id, corpus_snapshot_id, chain_head, ...)
- `session_events` (event_id, session_id, event_type, content_hash, prev_hash, chain_hash, ...)
- `session_working_set` (session_id, item_id, added_at)
- `session_pointer_set` (session_id, item_id, added_at)

**Cassette Layer:**
- `cassette_meta` (key, value)
- `cassette_messages` (message_id, run_id, source, payload_json, ...)
- `cassette_jobs` (job_id, message_id, intent, ordinal, ...)
- `cassette_steps` (step_id, job_id, status, lease_owner, ...)
- `cassette_receipts` (receipt_id, step_id, outcome, receipt_json, ...)
- `cassette_job_budgets` (job_id, bytes_consumed, symbols_consumed)

### Target Cassette Schema

```sql
-- Main cassette schema (chunks + files + FTS)
CREATE TABLE files (
    file_id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE
);

CREATE TABLE chunks (
    chunk_id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL,
    chunk_hash TEXT NOT NULL,
    content TEXT NOT NULL,
    FOREIGN KEY(file_id) REFERENCES files(file_id)
);

CREATE VIRTUAL TABLE chunks_fts USING fts5(
    content,
    hash UNINDEXED,
    tokenize='porter unicode61'
);

CREATE TABLE cassette_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

---

## Graduation Criteria

Before graduation can be considered:

1. **Phase A Complete:** Session persistence tests passing (DONE)
2. **Phase B Complete:** Main cassette integration working
3. **Phase C Complete:** Auto-controlled context loop operational
4. **All Invariants Verified:** INV-CATALYTIC-01 through 07
5. **Compression Proven:** Benchmarks demonstrate catalytic gains
6. **Golden Demo Works:** Fresh clone can run full demo

---

## cassettes.json Entry (Future)

When ready to add CAT_CHAT as a cassette:

```json
{
  "id": "cat_chat",
  "name": "CAT Chat (Catalytic Bounded Chat)",
  "db_path": "NAVIGATION/CORTEX/cassettes/cat_chat.db",
  "enabled": true,
  "description": "Catalytic chat knowledge and indexed sessions",
  "capabilities": ["fts", "semantic_search"],
  "type": "generic",
  "mutability": "append_only",
  "schema_version": "1.0.0"
}
```

---

## Current Focus

**This phase focuses on READING from cassettes, not graduation.**

The immediate work is:
1. CassetteClient for reading main cassettes (B.1) - DONE
2. Symbol resolution via cassettes (B.2) - DONE
3. Write isolation documentation (B.3) - DONE
4. This graduation path documentation (B.4) - DONE

Graduation decisions will be made after Section C (Auto-Controlled Context) is complete and the system has proven its catalytic behavior.
