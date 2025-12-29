# ADR-030: Semantic Core + Translation Layer Architecture

**Status:** Proposed
**Date:** 2025-12-28
**Confidence:** Low
**Impact:** High
**Tags:** [architecture, semantics, swarm, token-optimization]
**Deciders:** System Architect
**Supersedes:** None
**Related:** ADR-027 (Dual-DB), ADR-028 (Semiotic Compression Layer)

---

## Context

The current swarm architecture uses uniform models for all agents. This is inefficient:
- **Big models** (Opus) are expensive and slow but have deep semantic understanding
- **Small models** (Haiku) are cheap and fast but lack contextual depth

We need an architecture where the big model serves as the "brain" (semantic core) while small models act as "hands" (translation layer) that execute specific tasks without needing full context.

## Decision

Implement a **Semantic Core + Translation Layer** architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    SEMANTIC CORE (Opus)                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  CORTEX + Vector Store                               │    │
│  │  • Embeddings of all indexed content                 │    │
│  │  • Semantic similarity via cosine distance           │    │
│  │  • Compressed @Symbol references                     │    │
│  │  • Full codebase understanding                       │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                    Compressed IR (Symbols + Vectors)
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  TRANSLATION LAYER (Haiku)                   │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐    │
│  │   Ant-1       │  │   Ant-2       │  │   Ant-N       │    │
│  │               │  │               │  │               │    │
│  │ Receives:     │  │ Receives:     │  │ Receives:     │    │
│  │ • @Symbols    │  │ • @Symbols    │  │ • @Symbols    │    │
│  │ • Vectors     │  │ • Vectors     │  │ • Vectors     │    │
│  │ • Instruction │  │ • Instruction │  │ • Instruction │    │
│  │               │  │               │  │               │    │
│  │ Outputs:      │  │ Outputs:      │  │ Outputs:      │    │
│  │ • Code        │  │ • Code        │  │ • Code        │    │
│  │ • Actions     │  │ • Actions     │  │ • Actions     │    │
│  └───────────────┘  └───────────────┘  └───────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Architecture Components

### 1. Semantic Core (CORTEX + Vectors)

The big model maintains semantic understanding through:

#### 1.1 Vector Embeddings
```sql
-- New table in cortex.db
CREATE TABLE section_vectors (
    hash TEXT PRIMARY KEY,
    embedding BLOB,           -- 384-dim float32 (all-MiniLM-L6-v2)
    model_version TEXT,       -- Embedding model version
    created_at TEXT,
    FOREIGN KEY (hash) REFERENCES sections(hash)
);

CREATE INDEX idx_vectors_hash ON section_vectors(hash);
```

#### 1.2 Semantic Index
- **Embedding Model:** `all-MiniLM-L6-v2` (384 dimensions, fast, good quality)
- **Storage:** SQLite with sqlite-vss extension OR numpy arrays with FAISS
- **Update Strategy:** Incremental on file changes (via F3 hash comparison)

#### 1.3 Symbol Resolution with Context
```python
def expand_symbol(symbol: str) -> Dict:
    """Expand @Symbol with semantic context."""
    section = get_section(symbol)
    neighbors = semantic_search(section.embedding, top_k=3)

    return {
        "content": section.content,
        "hash": section.hash,
        "related": [n.symbol for n in neighbors],
        "embedding": section.embedding,  # For tiny model context
    }
```

### 2. Translation Layer (Ant Workers)

Small models receive compressed task specs:

#### 2.1 Compressed Task Protocol
```json
{
    "task_id": "refactor-001",
    "task_type": "code_adapt",

    "symbols": {
        "@target": {
            "content": "def dispatch_task(...):\n    ...",
            "hash": "a1b2c3d4",
            "file": "server.py",
            "lines": [1159, 1227]
        }
    },

    "vectors": {
        "task_intent": [0.12, -0.34, 0.56, ...],
        "context_centroid": [0.78, 0.11, -0.23, ...]
    },

    "instruction": "Add input validation to @target",

    "constraints": {
        "max_changes": 50,
        "preserve_signature": true,
        "run_tests": ["test_dispatch.py"]
    }
}
```

#### 2.2 Translation Model Behavior
The tiny model:
1. **Receives** compressed representation (symbols + vectors)
2. **Resolves** symbols to actual content on demand
3. **Executes** mechanical transformation
4. **Returns** structured result (no interpretation needed)

### 3. Communication Flow

```
User Request: "Add validation to dispatch_task"
         │
         ▼
┌─────────────────────────────────────────────┐
│  SEMANTIC CORE (Opus)                        │
│                                              │
│  1. Parse intent                             │
│  2. Search CORTEX for relevant sections      │
│  3. Retrieve vector embeddings               │
│  4. Compute semantic neighbors               │
│  5. Decompose into atomic tasks              │
│  6. Compress to @Symbols + vectors           │
│                                              │
│  Output: Compressed Task Spec                │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  GOVERNOR                                    │
│                                              │
│  1. Receive compressed task                  │
│  2. Route to available Ant                   │
│  3. Monitor progress                         │
│  4. Aggregate results                        │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  ANT WORKER (Haiku)                          │
│                                              │
│  1. Receive task with @Symbols               │
│  2. Resolve symbols to content               │
│  3. Apply vectors for context positioning    │
│  4. Execute transformation                   │
│  5. Return structured result                 │
│                                              │
│  No semantic reasoning needed                │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  SEMANTIC CORE (Opus)                        │
│                                              │
│  1. Validate result                          │
│  2. Integrate into codebase                  │
│  3. Update CORTEX index                      │
│  4. Continue or complete                     │
└─────────────────────────────────────────────┘
```

## Token Economics

### Without Semantic Core
```
Each Ant receives:  ~50,000 tokens context
Tasks per session:  10
Total tokens:       500,000 tokens

Cost: $$$$ (all workers need full context)
```

### With Semantic Core
```
Opus core:          ~100,000 tokens (once)
Each Ant receives:  ~2,000 tokens (compressed)
Tasks per session:  10
Total tokens:       100,000 + (10 × 2,000) = 120,000 tokens

Cost: $ (80% reduction)
```

## Integration with Existing Systems

### CORTEX (ADR-027)
- Add `section_vectors` table
- Extend `cortex_builder.py` with embedding generation
- Modify `semantic_search()` to use vectors

### Semiotic Compression Layer (ADR-028)
- `@Symbols` already provide 90% compression
- Add vector context to symbol resolution
- Extend compression operator: `σ(content) → {@Symbol, vector}`

### Living Formula
- R = (E / ∇S) × σ(f)^Df
- σ now includes vector compression
- Df (fractal dimension) maps to embedding space density

### MCP Server
- Add `resolve_symbols` tool
- Add `get_embeddings` tool
- Extend task_spec schema for vectors

## Consequences

### Positive
- **80% token reduction** through semantic compression
- **Faster execution** with parallel tiny workers
- **Better accuracy** from semantic positioning
- **Scalable** to many concurrent workers

### Negative
- **Initial indexing cost** for vector generation
- **Complexity** in symbol resolution
- **Latency** for embedding lookups (mitigated by caching)

### Risks
- Embedding model drift across versions
- Symbol resolution failures on stale references
- Vector space fragmentation over time

## Implementation Notes

### Embedding Model Selection
| Model | Dimensions | Speed | Quality |
|-------|------------|-------|---------|
| all-MiniLM-L6-v2 | 384 | Fast | Good |
| all-mpnet-base-v2 | 768 | Medium | Better |
| text-embedding-3-small | 1536 | API call | Best |

**Recommendation:** Start with `all-MiniLM-L6-v2` (local, fast, good enough)

### Vector Storage Options
1. **SQLite + numpy** - Simple, no dependencies
2. **sqlite-vss** - Native vector search in SQLite
3. **FAISS** - Fast similarity search (Facebook)
4. **ChromaDB** - Full vector DB with persistence

**Recommendation:** Start with SQLite + numpy, upgrade to FAISS if needed

## Related Documents
- [ADR-027: Dual-DB Architecture](ADR-027-dual-database-architecture.md)
- [ADR-028: Semiotic Compression Layer](ADR-028-semiotic-compression-layer.md)
- [The Living Formula](ADR-∞-living-formula.md)
- [Swarm Refactoring Report](../../CONTRACTS/_runs/swarm-refactoring-report.md)
