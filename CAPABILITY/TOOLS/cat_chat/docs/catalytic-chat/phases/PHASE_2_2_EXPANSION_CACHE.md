# Phase 2.2 — Expansion Cache Schema (Cat Chat)

Status: SPEC LOCKED  
Phase: Cat Chat Phase 2.2  
Purpose: Enforce bounded expansion and eliminate repeated context costs.

---

## Purpose

The Expansion Cache guarantees that **each unique symbol + slice is expanded at most once per content version** and then reused by reference.

This is the core compression valve of Cat Chat.

---

## Cache Identity (Non‑Negotiable)

Each cache entry is uniquely identified by the tuple:

```
(run_id, symbol_id, slice, section_content_hash)
```

If any element differs, the cache lookup must miss.

---

## Resolution Contract

A resolution request follows this exact pipeline:

```
resolve(symbol_id, slice, run_id)
  → validate symbol
  → resolve symbol → section_id
  → validate slice (fail‑closed)
  → read section_content_hash
  → cache lookup
     → HIT  → return payload
     → MISS → expand, store, return
```

No other execution paths are allowed.

---

## SQLite Schema (Primary Substrate)

```sql
CREATE TABLE IF NOT EXISTS expansions (
  run_id TEXT NOT NULL,
  symbol_id TEXT NOT NULL,
  slice TEXT NOT NULL,
  section_id TEXT NOT NULL,
  section_content_hash TEXT NOT NULL,

  payload TEXT NOT NULL,
  payload_hash TEXT NOT NULL,
  bytes_expanded INTEGER NOT NULL,

  created_at TEXT NOT NULL,

  PRIMARY KEY (run_id, symbol_id, slice, section_content_hash)
);

CREATE INDEX IF NOT EXISTS idx_expansions_symbol
  ON expansions(symbol_id);

CREATE INDEX IF NOT EXISTS idx_expansions_section
  ON expansions(section_id);
```

### Enforcement Rules
- `slice` must already be validated by the slice parser.
- `payload_hash` must be computed using the same normalization rules as Phase 1.
- `bytes_expanded` is the UTF‑8 byte length of `payload`.
- Rows are **append‑only**. No updates or deletes.

---

## JSONL Schema (Fallback Substrate)

**Path:** `CORTEX/_cache/expansions.jsonl`

```json
{
  "run_id": "run_2025_12_29_001",
  "symbol_id": "@CANON/immutability",
  "slice": "lines[0:80]",
  "section_id": "sec_000123",
  "section_content_hash": "9a41f3...",

  "payload": "expanded text here",
  "payload_hash": "b71c2d...",
  "bytes_expanded": 412,

  "created_at": "2025-12-29T18:04:00Z"
}
```

### JSONL Rules
- One record per line.
- Duplicate primary key is an error.
- Lookups require an in‑memory index keyed by the full identity tuple.

---

## Validation Rules (Fail‑Closed)

- Symbol must exist in SYMBOLS registry.
- Symbol must resolve to exactly one `section_id`.
- `slice=ALL` is forbidden.
- Negative or malformed slices are forbidden.
- Cache hit must never re‑expand content.
- Cache miss must expand exactly once, then store.

---

## Determinism Guarantees

- Same inputs → same cache key.
- Same cache key → identical payload.
- Content change → automatic cache invalidation via `section_content_hash`.
- Repeated resolves in a run incur **zero additional expansion cost**.

---

## Minimal CLI Surface (Phase 2.2)

```bash
cortex resolve @Symbol --slice "lines[0:80]" --run-id <run_id>
```

Behavior:
- Payload printed to stdout.
- Cache status printed to stderr:
  - `[CACHE HIT]`
  - `[CACHE MISS]`
- Non‑zero exit code on any validation failure.

---

## Non‑Goals (Explicit)

This phase does **not** include:
- Bundles
- Message schemas
- Vector search
- Agents or coordination
- Cross‑run eviction policies

Those belong to later phases.

---

## Completion Criteria

Phase 2.2 is complete when:
- Cache schema exists (SQLite + JSONL).
- Resolver enforces cache usage.
- Repeated resolves reuse cached payloads.
- CLI confirms hit/miss behavior.
- Determinism verified across runs.
