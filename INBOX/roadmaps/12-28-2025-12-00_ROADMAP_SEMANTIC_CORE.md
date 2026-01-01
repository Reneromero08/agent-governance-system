---
title: "Roadmap Semantic Core"
section: "roadmap"
author: "System"
priority: "High"
created: "2025-12-28 12:00"
modified: "2025-12-28 12:00"
status: "Active"
summary: "Roadmap for semantic core development (Restored)"
tags: [semantic_core, roadmap]
---
<!-- CONTENT_HASH: 4d2fb5383029f32ac8a9032ab8861597f22c74a1a399e7280ebb986d7cd7b7cb -->

# Semantic Core Implementation Roadmap

**Document:** ROADMAP-semantic-core.md
**Created:** 2025-12-28
**Related ADR:** ADR-030

---

## Overview

This roadmap outlines the implementation of the Semantic Core + Translation Layer architecture. The work is divided into 4 phases, each building on the previous.

```
Phase 1: Vector Foundation     ████░░░░░░░░░░░░  25%
Phase 2: Symbol Enhancement    ░░░░░░░░░░░░░░░░   0%
Phase 3: Translation Protocol  ░░░░░░░░░░░░░░░░   0%
Phase 4: Integration           ░░░░░░░░░░░░░░░░   0%
```

---

## Phase 1: Vector Foundation

**Goal:** Add vector embeddings to CORTEX

### Tasks

#### 1.1 Install Embedding Dependencies
```bash
pip install sentence-transformers numpy
```

Files to modify:
- `requirements.txt` or `pyproject.toml`

#### 1.2 Create Vector Schema
Add to `CORTEX/cortex.db`:

```sql
CREATE TABLE section_vectors (
    hash TEXT PRIMARY KEY,
    embedding BLOB NOT NULL,
    model_id TEXT DEFAULT 'all-MiniLM-L6-v2',
    dimensions INTEGER DEFAULT 384,
    created_at TEXT NOT NULL,
    FOREIGN KEY (hash) REFERENCES sections(hash)
);

CREATE INDEX idx_section_vectors_model ON section_vectors(model_id);
```

Files to create/modify:
- `CORTEX/schema/002_vectors.sql`
- `CORTEX/cortex_builder.py`

#### 1.3 Implement Embedding Generator
```python
# CORTEX/embeddings.py

from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path

class EmbeddingEngine:
    """Generate and manage embeddings for CORTEX sections."""

    MODEL_ID = "all-MiniLM-L6-v2"
    DIMENSIONS = 384

    def __init__(self):
        self.model = SentenceTransformer(self.MODEL_ID)

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        return self.model.encode(text, convert_to_numpy=True)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        return self.model.encode(texts, convert_to_numpy=True)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def serialize(self, embedding: np.ndarray) -> bytes:
        """Serialize embedding for SQLite storage."""
        return embedding.astype(np.float32).tobytes()

    def deserialize(self, blob: bytes) -> np.ndarray:
        """Deserialize embedding from SQLite storage."""
        return np.frombuffer(blob, dtype=np.float32)
```

Files to create:
- `CORTEX/embeddings.py`

#### 1.4 Extend CORTEX Builder
Add embedding generation during indexing:

```python
# In cortex_builder.py

from embeddings import EmbeddingEngine

class CortexBuilder:
    def __init__(self):
        self.embedding_engine = EmbeddingEngine()

    def index_section(self, section: dict):
        # Existing indexing logic...

        # Generate and store embedding
        embedding = self.embedding_engine.embed(section["content"])
        self._store_embedding(section["hash"], embedding)

    def _store_embedding(self, hash: str, embedding: np.ndarray):
        blob = self.embedding_engine.serialize(embedding)
        self.db.execute("""
            INSERT OR REPLACE INTO section_vectors
            (hash, embedding, model_id, dimensions, created_at)
            VALUES (?, ?, ?, ?, datetime('now'))
        """, (hash, blob, EmbeddingEngine.MODEL_ID, EmbeddingEngine.DIMENSIONS))
```

Files to modify:
- `CORTEX/cortex_builder.py`

#### 1.5 Implement Semantic Search
```python
# CORTEX/semantic_search.py

def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Find semantically similar sections."""
    engine = EmbeddingEngine()
    query_embedding = engine.embed(query)

    # Load all embeddings (for small corpora)
    # For large corpora, use FAISS or sqlite-vss
    results = []
    for row in db.execute("SELECT hash, embedding FROM section_vectors"):
        section_embedding = engine.deserialize(row["embedding"])
        similarity = engine.cosine_similarity(query_embedding, section_embedding)
        results.append({
            "hash": row["hash"],
            "similarity": similarity
        })

    # Sort by similarity and return top_k
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]
```

Files to create:
- `CORTEX/semantic_search.py`

### Deliverables
- [ ] Vector table in cortex.db
- [ ] EmbeddingEngine class
- [ ] Semantic search function
- [ ] Unit tests for embeddings

---

## Phase 2: Symbol Enhancement

**Goal:** Extend @Symbol system with vector context

### Tasks

#### 2.1 Enhanced Symbol Schema
```python
@dataclass
class EnhancedSymbol:
    """Symbol with semantic context."""
    name: str              # e.g., "@ADR-027"
    content: str           # Resolved content
    hash: str              # Content hash
    file_path: str         # Source file
    line_range: tuple      # (start, end) lines
    embedding: np.ndarray  # Vector representation
    neighbors: list[str]   # Semantically related symbols
```

Files to create:
- `CATALYTIC-DPT/LAB/symbols.py`

#### 2.2 Symbol Resolution with Vectors
```python
def resolve_symbol(symbol: str) -> EnhancedSymbol:
    """Resolve @Symbol with full semantic context."""
    section = cortex.get_section(symbol)
    embedding = cortex.get_embedding(section.hash)
    neighbors = semantic_search(embedding, top_k=3)

    return EnhancedSymbol(
        name=symbol,
        content=section.content,
        hash=section.hash,
        file_path=section.file_path,
        line_range=section.line_range,
        embedding=embedding,
        neighbors=[n.symbol for n in neighbors]
    )
```

Files to modify:
- `CATALYTIC-DPT/LAB/symbols.py`

#### 2.3 Symbol Compression Operator
```python
def compress(content: str) -> dict:
    """Compress content to symbol + vector representation."""
    # Find or create symbol
    symbol = find_or_create_symbol(content)

    # Get embedding
    embedding = embed(content)

    return {
        "symbol": symbol.name,
        "vector": embedding.tolist()[:64],  # Truncate for transport
        "hash": symbol.hash
    }
```

Files to modify:
- `CATALYTIC-DPT/LAB/compression.py`

### Deliverables
- [ ] EnhancedSymbol dataclass
- [ ] resolve_symbol with vectors
- [ ] compress operator
- [ ] Symbol caching layer

---

## Phase 3: Translation Protocol

**Goal:** Update ant-worker to accept compressed task specs

### Tasks

#### 3.1 Compressed Task Spec Schema
```python
COMPRESSED_TASK_SCHEMA = {
    "task_id": str,
    "task_type": str,

    "symbols": {
        # Named symbols with content
        "@name": {
            "content": str,
            "hash": str,
            "file": str,
            "lines": [int, int]
        }
    },

    "vectors": {
        "task_intent": list[float],      # What to do
        "context_centroid": list[float]  # Where we are semantically
    },

    "instruction": str,  # Natural language instruction

    "constraints": {
        "max_changes": int,
        "preserve_signature": bool,
        "run_tests": list[str]
    }
}
```

Files to modify:
- `CATALYTIC-DPT/LAB/MCP/server.py` (add schema validation)

#### 3.2 Symbol Resolver for Ant Workers
```python
# CATALYTIC-DPT/SKILLS/ant-worker/scripts/symbol_resolver.py

class SymbolResolver:
    """Resolve @Symbols in task specs for ant workers."""

    def __init__(self, task_spec: dict):
        self.symbols = task_spec.get("symbols", {})
        self.resolved_cache = {}

    def resolve(self, symbol: str) -> str:
        """Resolve symbol to content."""
        if symbol in self.resolved_cache:
            return self.resolved_cache[symbol]

        if symbol in self.symbols:
            content = self.symbols[symbol]["content"]
            self.resolved_cache[symbol] = content
            return content

        # Fallback: query CORTEX
        return self._query_cortex(symbol)

    def expand_instruction(self, instruction: str) -> str:
        """Replace @Symbols in instruction with content."""
        import re
        pattern = r'@[\w\-\.]+'

        def replacer(match):
            symbol = match.group(0)
            return self.resolve(symbol)

        return re.sub(pattern, replacer, instruction)
```

Files to create:
- `CATALYTIC-DPT/SKILLS/ant-worker/scripts/symbol_resolver.py`

#### 3.3 Update Ant Worker
```python
# In ant-worker/scripts/run.py

from symbol_resolver import SymbolResolver

class TaskExecutor:
    def __init__(self, task_spec: dict):
        self.task_spec = task_spec
        self.resolver = SymbolResolver(task_spec)

    def execute(self):
        # Resolve symbols in instruction
        instruction = self.resolver.expand_instruction(
            self.task_spec.get("instruction", "")
        )

        # Execute based on task type
        task_type = self.task_spec.get("task_type")
        if task_type == "code_adapt":
            self._execute_code_adapt_with_symbols()
        # ...
```

Files to modify:
- `CATALYTIC-DPT/SKILLS/ant-worker/scripts/run.py`

#### 3.4 Governor Task Compression
```python
# In swarm-orchestrator/scripts/poll_and_execute.py

def compress_task_for_ant(task_spec: dict, directive: str) -> dict:
    """Compress task spec for efficient ant worker processing."""
    # Extract relevant symbols
    symbols = extract_symbols(directive)

    # Resolve each symbol with content
    resolved = {}
    for symbol in symbols:
        section = cortex.get_section(symbol)
        resolved[symbol] = {
            "content": section.content,
            "hash": section.hash,
            "file": section.file_path,
            "lines": section.line_range
        }

    # Compute task intent vector
    intent_vector = embed(directive)

    return {
        **task_spec,
        "symbols": resolved,
        "vectors": {
            "task_intent": intent_vector.tolist()[:64]
        },
        "instruction": directive
    }
```

Files to modify:
- `CATALYTIC-DPT/SKILLS/swarm-orchestrator/scripts/poll_and_execute.py`

### Deliverables
- [ ] Compressed task spec schema
- [ ] SymbolResolver class
- [ ] Updated ant-worker with symbol support
- [ ] Governor task compression

---

## Phase 4: Integration & Optimization

**Goal:** Complete integration and performance tuning

### Tasks

#### 4.1 MCP Tool Extensions
Add new tools to MCP server:

```python
# New tools in server.py

def resolve_symbols(self, symbols: list[str]) -> dict:
    """Batch resolve symbols to content."""
    pass

def get_embeddings(self, hashes: list[str]) -> dict:
    """Get embeddings for content hashes."""
    pass

def semantic_search(self, query: str, top_k: int = 10) -> dict:
    """Search CORTEX semantically."""
    pass

def compress_context(self, content: str) -> dict:
    """Compress content to symbol + vector."""
    pass
```

Files to modify:
- `CATALYTIC-DPT/LAB/MCP/server.py`

#### 4.2 Caching Layer
```python
# CATALYTIC-DPT/LAB/cache.py

from functools import lru_cache
import hashlib

class SemanticCache:
    """Cache for embeddings and symbol resolutions."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._embedding_cache = {}
        self._symbol_cache = {}

    @lru_cache(maxsize=10000)
    def get_embedding(self, content_hash: str) -> np.ndarray:
        """Get cached embedding."""
        pass

    def invalidate(self, content_hash: str):
        """Invalidate cache entry on content change."""
        pass
```

Files to create:
- `CATALYTIC-DPT/LAB/cache.py`

#### 4.3 Performance Optimization
- Batch embedding generation
- FAISS index for large corpora (>10K sections)
- Lazy symbol resolution
- Connection pooling for SQLite

#### 4.4 Metrics & Monitoring
```python
# Metrics to track
METRICS = {
    "tokens_before_compression": Counter,
    "tokens_after_compression": Counter,
    "compression_ratio": Gauge,
    "embedding_cache_hits": Counter,
    "symbol_resolution_time_ms": Histogram,
    "semantic_search_time_ms": Histogram,
}
```

Files to create:
- `CATALYTIC-DPT/LAB/metrics.py`

#### 4.5 Integration Tests
```python
# tests/test_semantic_core.py

def test_full_flow():
    """Test complete semantic core flow."""
    # 1. Index content
    # 2. Generate embeddings
    # 3. Create compressed task
    # 4. Execute via ant worker
    # 5. Verify result
    pass

def test_token_reduction():
    """Verify 80% token reduction target."""
    original_tokens = count_tokens(full_context)
    compressed_tokens = count_tokens(compressed_task)
    assert compressed_tokens < original_tokens * 0.2
```

Files to create:
- `tests/test_semantic_core.py`

### Deliverables
- [ ] MCP tool extensions
- [ ] Caching layer
- [ ] Performance optimizations
- [ ] Metrics collection
- [ ] Integration tests

---

## Milestones

| Milestone | Description | Dependencies |
|-----------|-------------|--------------|
| M1 | Vector embeddings in CORTEX | Phase 1 |
| M2 | Enhanced symbol resolution | Phase 1, Phase 2 |
| M3 | Compressed task protocol | Phase 2, Phase 3 |
| M4 | Full integration | Phase 3, Phase 4 |
| M5 | Production ready | Phase 4 |

---

## Success Criteria

### Phase 1
- [ ] All CORTEX sections have embeddings
- [ ] Semantic search returns relevant results
- [ ] <100ms embedding generation per section

### Phase 2
- [ ] @Symbols resolve with vector context
- [ ] Neighbor discovery works
- [ ] Symbol caching reduces repeated queries by 90%

### Phase 3
- [ ] Ant workers accept compressed task specs
- [ ] Symbol resolution works in isolation
- [ ] Tasks execute without full codebase context

### Phase 4
- [ ] 80% token reduction achieved
- [ ] No latency regression vs current system
- [ ] All existing tests pass
- [ ] New integration tests pass

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Embedding model updates | Version lock, migration scripts |
| Symbol resolution failures | Fallback to full content |
| Vector space drift | Periodic re-indexing |
| Performance regression | Caching, batch operations |
| Memory pressure | Streaming, lazy loading |

---

## Dependencies

### Required Packages
```
sentence-transformers>=2.2.0
numpy>=1.24.0
faiss-cpu>=1.7.0 (optional, for large corpora)
```

### External Services
- None (all local processing)

### Internal Dependencies
- CORTEX indexer
- MCP server
- Swarm infrastructure (refactored in previous work)

---

## Next Steps

1. **Immediate:** Implement Phase 1.1 - Install dependencies
2. **This week:** Complete Phase 1 - Vector Foundation
3. **Next sprint:** Begin Phase 2 - Symbol Enhancement
