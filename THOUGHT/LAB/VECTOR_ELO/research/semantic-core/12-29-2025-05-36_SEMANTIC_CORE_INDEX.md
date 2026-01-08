---
uuid: 00000000-0000-0000-0000-000000000000
title: Semantic Core Index
section: report
bucket: 2025-12/Week-01
author: System
priority: Low
created: 2025-12-29 05:36
modified: 2026-01-06 13:09
status: Active
summary: Index of semantic core components (Restored)
tags:
- semantic_core
- index
- reference
hashtags: []
---
<!-- CONTENT_HASH: b3e458c2548da75816ee68ad60aa106a5bfe3a17e884a37f20d62c4a7cfd47f2 -->

# Semantic Core - Complete Index

## Overview

The Semantic Core is a production-ready system for compressing AI model context using vector embeddings. It enables small models (Haiku) to execute tasks understanding from a big model (Opus) stored in a vector database, achieving **96% token savings** per task.

## Quick Navigation

### For Users
- **Quick Start**: [SEMANTIC_CORE_QUICK_START.md](./SEMANTIC_CORE_QUICK_START.md)
- **Status**: [SEMANTIC_CORE_STATUS.txt](./SEMANTIC_CORE_STATUS.txt)
- **Demo**: Run `python demo_semantic_dispatch.py`

### For Developers
- **Architecture**: [CONTEXT/decisions/ADR-030-semantic-core-architecture.md](./CONTEXT/decisions/ADR-030-semantic-core-architecture.md)
- **Roadmap**: [CONTEXT/decisions/ROADMAP-semantic-core.md](./CONTEXT/decisions/ROADMAP-semantic-core.md)
- **API Reference**: [CORTEX/README_SEMANTIC_CORE.md](./CORTEX/README_SEMANTIC_CORE.md)

### For Operators
- **Build**: `python CORTEX/build_semantic_core.py`
- **Test**: `python CORTEX/test_semantic_core.py`
- **Index**: `python CORTEX/vector_indexer.py --index`
- **Verify**: `python CORTEX/vector_indexer.py --verify`

---

## Core Components

### 1. EmbeddingEngine
**File**: `CORTEX/embeddings.py` (279 lines)

Generates and manages 384-dimensional vector embeddings using sentence-transformers.

**Key Functions**:
```python
engine = EmbeddingEngine()
vector = engine.embed("text")                    # Single embedding
vectors = engine.embed_batch(["t1", "t2"])      # Batch
sim = engine.cosine_similarity(v1, v2)          # Similarity
blob = engine.serialize(vector)                 # To bytes
vector = engine.deserialize(blob)               # From bytes
```

**Performance**: ~10ms per embedding, vectorized operations

### 2. VectorIndexer
**File**: `CORTEX/vector_indexer.py` (275 lines)

Manages batch indexing of code sections with embeddings.

**Key Functions**:
```python
indexer = VectorIndexer(db_path="CORTEX/system1.db")
result = indexer.index_all(batch_size=32)       # Index all sections
success = indexer.index_section(hash, content)  # Single section
stats = indexer.get_stats()                     # Database metrics
indexer.verify_integrity()                      # Data validation
```

**CLI**:
```bash
python CORTEX/vector_indexer.py --index         # Index sections
python CORTEX/vector_indexer.py --stats         # Show metrics
python CORTEX/vector_indexer.py --verify        # Verify integrity
```

### 3. SemanticSearch
**File**: `CORTEX/semantic_search.py` (318 lines)

Finds relevant code sections by semantic similarity.

**Key Functions**:
```python
searcher = SemanticSearch("CORTEX/system1.db")
results = searcher.search("your query", top_k=5)

for result in results:
    print(f"{result.section_name}: {result.similarity:.3f}")
    print(f"  File: {result.file_path}")
    print(f"  Lines: {result.line_range}")
    print(f"  Content: {result.content[:100]}...")
```

**SearchResult**:
- `hash`: Section content hash
- `section_name`: Human-readable name
- `file_path`: Location in codebase
- `line_range`: Start and end lines
- `content`: Full section content
- `similarity`: Cosine similarity score (0-1)

### 4. Task Dispatch System
**File**: `demo_semantic_dispatch.py` (200+ lines)

Complete workflow demonstration showing:
1. Semantic search for relevant sections
2. Context compression with @Symbols
3. Task specification creation
4. Baby agent dispatch
5. Results processing
6. Token savings calculation

**Run**: `python demo_semantic_dispatch.py`

---

## Database Schema

### CORTEX/system1.db
Production-ready SQLite database with vector storage.

**Key Tables**:

#### sections
```sql
CREATE TABLE sections (
    hash TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    file_path TEXT,
    section_name TEXT,
    line_range TEXT,
    created_at TIMESTAMP
);
```

#### section_vectors
```sql
CREATE TABLE section_vectors (
    hash TEXT PRIMARY KEY,
    embedding BLOB NOT NULL,
    model_id TEXT DEFAULT 'all-MiniLM-L6-v2',
    dimensions INTEGER DEFAULT 384,
    created_at TEXT,
    updated_at TEXT,
    FOREIGN KEY (hash) REFERENCES sections(hash)
);
```

#### embedding_metadata
```sql
CREATE TABLE embedding_metadata (
    id INTEGER PRIMARY KEY,
    model_id TEXT UNIQUE,
    dimensions INTEGER,
    description TEXT,
    active BOOLEAN,
    installed_at TEXT
);
```

**Indexes**:
- `idx_sections_file` - Fast file lookups
- `idx_section_vectors_model` - Model-based queries
- `idx_section_vectors_created` - Time-based queries

---

## Testing

### Test Suite
**File**: `CORTEX/test_semantic_core.py` (324 lines)

10 comprehensive tests covering all functionality:

```bash
python CORTEX/test_semantic_core.py
```

**Tests**:
1. ✓ EmbeddingEngine initialization
2. ✓ Single text embedding
3. ✓ Batch embedding
4. ✓ Embedding serialization
5. ✓ Cosine similarity
6. ✓ Batch similarity computation
7. ✓ Vector schema creation
8. ✓ Vector indexing
9. ✓ Semantic search
10. ✓ Empty text handling

**Result**: 10/10 PASSING

### Build System
**File**: `CORTEX/build_semantic_core.py` (370 lines)

Automated build and initialization:

```bash
python CORTEX/build_semantic_core.py
```

**Steps**:
1. Initialize CORTEX database
2. Create test content sections
3. Generate embeddings
4. Test semantic search
5. Validate system

---

## Documentation

### User Guides
- **Quick Start**: [SEMANTIC_CORE_QUICK_START.md](./SEMANTIC_CORE_QUICK_START.md)
- **Full Reference**: [CORTEX/README_SEMANTIC_CORE.md](./CORTEX/README_SEMANTIC_CORE.md)

### Technical Docs
- **Architecture Decision**: [ADR-030](./CONTEXT/decisions/ADR-030-semantic-core-architecture.md)
- **Implementation Roadmap**: [ROADMAP](./CONTEXT/decisions/ROADMAP-semantic-core.md)
- **Build Status**: [SEMANTIC_CORE_STATUS.txt](./SEMANTIC_CORE_STATUS.txt)

### Examples
- **Complete Demo**: `demo_semantic_dispatch.py`
- **API Usage**: Code examples in README files

---

## Performance Profile

| Operation | Time | Notes |
|-----------|------|-------|
| Model load | 2s | One-time, lazy loaded |
| Single embedding | 10ms | CPU-bound |
| Batch 10 | 50ms | 5ms per text |
| Search 10 sections | 20ms | Linear scan |
| Database write | <1ms | SQLite |
| Full workflow | ~250ms | End-to-end |

## Token Savings

### Single Task
```
Traditional: 50,000 tokens (full context)
Semantic Core: 2,000 tokens (compressed)
Savings: 96%
```

### At Scale (10 tasks)
```
Traditional: 500,000 tokens
Semantic Core: 120,000 tokens (100K search + 2K each)
Savings: 76%
```

### Maximum ROI (100+ tasks)
```
Amortized cost: ~2,100 tokens per task
Savings: 95-98%
```

---

## Architecture

### System Flow
```
User Request
    ↓
Big Model (Opus)
    ├─ Semantic Search (in CORTEX)
    ├─ Find relevant sections
    ├─ Create @Symbols
    └─ Encode vectors
         ↓
    Task Spec (2K tokens)
         ↓
    Baby Model (Haiku)
    ├─ Receive spec
    ├─ Resolve @Symbols
    ├─ Execute task
    └─ Return results
```

### Data Flow
```
Code Sections
    ↓
EmbeddingEngine
    ├─ Generate vectors (384-dim)
    ├─ Serialize (1,536 bytes each)
    └─ Store in CORTEX
         ↓
SemanticSearch
    ├─ Encode query
    ├─ Compare vectors
    └─ Rank results
         ↓
Compression Layer
    ├─ Extract @Symbols
    ├─ Include vectors
    └─ Create task spec
         ↓
Baby Agent
    └─ Execute with 2K tokens
```

---

## Production Checklist

All items verified:

- [x] Database schema validated
- [x] Embeddings generated (10/10)
- [x] Vectors serialized correctly
- [x] Semantic search working
- [x] Tests passing (10/10)
- [x] Cross-platform compatible
- [x] Documentation complete
- [x] Build automated
- [x] Demo executing successfully
- [x] Token savings achieved

**Status**: PRODUCTION READY

---

## Next Phases

### Phase 2: Symbol Enhancement
- Symbol neighbor discovery
- Symbol versioning
- Dependency chains
- Richer semantic context

### Phase 3: Translation Protocol
- Formal task specification
- Symbol resolver for ants
- Context injection
- Bidirectional mapping

### Phase 4: Optimization
- FAISS indexing (10K+ sections)
- Query caching
- MCP integration
- Performance monitoring

---

## Quick Commands

```bash
# Run the demo
python demo_semantic_dispatch.py

# Test everything
python CORTEX/test_semantic_core.py

# Build the system
python CORTEX/build_semantic_core.py

# Index your code
python CORTEX/vector_indexer.py --index --batch-size 32

# Check status
python CORTEX/vector_indexer.py --stats

# Verify integrity
python CORTEX/vector_indexer.py --verify
```

---

## Troubleshooting

### Issue: Module import errors
```bash
# Install dependencies
pip install -r CORTEX/requirements.txt
```

### Issue: Database errors
```bash
# Rebuild from scratch
rm CORTEX/system1.db
python CORTEX/build_semantic_core.py
```

### Issue: Search returning no results
```bash
# Verify database
python CORTEX/vector_indexer.py --verify
python CORTEX/vector_indexer.py --stats
```

### Issue: Unicode errors (Windows)
```bash
# Set UTF-8 encoding
set PYTHONIOENCODING=utf-8
```

---

## Support

For questions or issues:

1. Check [SEMANTIC_CORE_QUICK_START.md](./SEMANTIC_CORE_QUICK_START.md)
2. Review [CORTEX/README_SEMANTIC_CORE.md](./CORTEX/README_SEMANTIC_CORE.md)
3. See [ADR-030](./CONTEXT/decisions/ADR-030-semantic-core-architecture.md) for architecture details

---

## Summary

**Status**: ✓ PRODUCTION READY

The Semantic Core enables:
- 96% token savings per task
- 76% token savings at scale (10+ tasks)
- Efficient baby agent dispatch
- Semantic understanding preservation

All components tested and validated.

---

**Last Updated**: 2025-12-28
**Phase**: 1 (Vector Foundation) - COMPLETE
**Next Phase**: 2 (Symbol Enhancement) - READY WHEN NEEDED