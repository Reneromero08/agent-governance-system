# Vector Sandbox (Experimental)

> **Status:** Phase 2.5 Experimental
> **Canonical:** NO - This is a sandbox for exploration only
> **Affects:** Does not modify Phase 2 behavior or Phase 3 scope

## Overview

This is a minimal SQLite-backed vector store for local experiments and tests. It supports "move in vectors" exploration without modifying canonical phases.

## Important Warnings

- **NOT part of Phase 2** - No changes to existing symbols, resolve, or expansion cache
- **NOT Phase 3** - No message cassette (messages/jobs/steps/receipts) implementation
- **Experimental** - Code may change or be removed without notice
- **Optional** - Never required for Phase 2 tests or canonical functionality

## Location

- Module: `catalytic_chat/experimental/vector_store.py`
- Tests: `tests/test_vector_store.py`

## API

### VectorStore Class

```python
from catalytic_chat.experimental.vector_store import VectorStore

# Initialize with default DB path (CORTEX/db/system1.db)
store = VectorStore()

# Or specify custom path
store = VectorStore(db_path=Path("custom.db"))

# Context manager for automatic cleanup
with VectorStore() as store:
    # ... operations
```

### Methods

#### put_vector(namespace, content_bytes, vector, meta) -> vector_id

Store a vector with associated content and metadata.

```python
vector_id = store.put_vector(
    namespace="my_ns",
    content_bytes=b"some content",
    vector=[0.1, 0.2, 0.3, 0.4],
    meta={"key": "value", "num": 42}
)
```

- `namespace`: String namespace for isolation (queries are scoped to namespace)
- `content_bytes`: Raw content bytes (hashed for deduplication)
- `vector`: List[float] embedding vector
- `meta`: Dict[str, Any] metadata (stored as JSON)

Returns: 16-character vector_id string

#### get_vector(vector_id) -> dict | None

Retrieve a vector by ID.

```python
result = store.get_vector(vector_id)
# Returns None if not found
# Otherwise: {
#     "vector_id": "...",
#     "namespace": "...",
#     "content_hash": "...",
#     "dims": 4,
#     "vector": [0.1, 0.2, 0.3, 0.4],
#     "meta": {...},
#     "created_at": "2025-12-29T..."
# }
```

#### query_topk(namespace, query_vector, k=5) -> list[dict]

Find top-k similar vectors in a namespace using cosine similarity.

```python
results = store.query_topk(
    namespace="my_ns",
    query_vector=[1.0, 0.0, 0.0, 0.0],
    k=5
)
# Returns list sorted by score (descending), then vector_id (for ties)
# Each result includes a "score" field (0.0 to 1.0)
```

#### delete_namespace(namespace) -> int

Delete all vectors in a namespace.

```python
count = store.delete_namespace("my_ns")
```

## Database Schema

```sql
CREATE TABLE vectors (
    vector_id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    dims INTEGER NOT NULL,
    vector_json TEXT NOT NULL,
    meta_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX idx_vectors_namespace ON vectors(namespace);
CREATE INDEX idx_vectors_content_hash ON vectors(content_hash);
```

## Distance Metric

Cosine similarity implemented in pure Python (no SQLite extension):

```
similarity(a, b) = dot(a, b) / (||a|| * ||b||)
```

Deterministic ordering on ties: sorted by vector_id ascending.

## Running Tests

```bash
# Run all tests (including vector store tests)
cd THOUGHT/LAB/CAT_CHAT
python -m pytest tests/test_vector_store.py -v

# Run specific test
python -m pytest tests/test_vector_store.py::test_vector_put_get_roundtrip -v
```

## Verification

```bash
# Verify import works
python -c "from catalytic_chat.experimental.vector_store import VectorStore; print('OK')"

# Verify all tests pass
python -m pytest -q
```

## Test Cases

- `test_vector_put_get_roundtrip` - Verify put/get returns identical data
- `test_vector_query_topk_deterministic` - Verify top-k ordering is stable
- `test_vector_reject_bad_dims` - Verify empty vectors are rejected
- `test_vector_namespace_isolation` - Verify queries are namespace-scoped
- Additional: context manager, replacement, deletion, empty namespace

## Design Decisions

1. **SQLite-only** - No external dependencies, uses existing DB conventions
2. **JSON storage** - Vectors stored as JSON arrays for portability
3. **Cosine similarity in Python** - Avoids SQLite extension complexity
4. **Namespace isolation** - Enables multi-tenant experiments without interference
5. **Content hashing** - Deduplicates identical content across vectors
