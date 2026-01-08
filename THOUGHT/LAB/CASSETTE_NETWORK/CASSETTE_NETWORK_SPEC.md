# Cassette Network Specification

**Version**: 1.0
**Protocol**: SNP (Semantic Network Protocol)
**Status**: Production-ready foundation (Phase 0 complete)

---

## Overview

The Cassette Network is a federated database architecture where specialized databases (cassettes) plug into a semantic network hub for cross-database queries and token compression. Each cassette is a self-contained, portable unit that advertises its capabilities via a standardized handshake protocol.

---

## Core Concepts

### Cassette
A cassette is a specialized database unit that:
- Contains semantically related content (governance docs, code, research, etc.)
- Implements the `DatabaseCassette` interface
- Advertises capabilities via handshake protocol
- Is schema-independent (network adapts)
- Is hot-swappable (plug in/out as needed)
- Is independently maintained and versioned

### Network Hub
The `SemanticNetworkHub` is the central coordinator that:
- Registers cassettes via handshake
- Routes queries based on capabilities
- Aggregates and reranks results
- Monitors network health

### Capabilities
Each cassette advertises what it can do:
- `vectors` - Semantic search with embeddings
- `fts` - Full-text search via FTS5
- `semantic_search` - Semantic query routing
- `research` - Research paper chunks
- `ast` - AST metadata extraction
- `fixtures` - Test fixtures and schemas

---

## Protocol Specification

### Cassette Interface

```python
class DatabaseCassette(ABC):
    """Base class for all database cassettes."""

    def __init__(self, db_path: Path, cassette_id: str):
        self.db_path = db_path
        self.cassette_id = cassette_id
        self.capabilities: List[str] = []
        self.schema_version = "1.0"

    def handshake(self) -> Dict:
        """Return cassette metadata for network registration."""
        return {
            "cassette_id": self.cassette_id,
            "db_path": str(self.db_path),
            "db_hash": self._compute_hash(),
            "capabilities": self.capabilities,
            "schema_version": self.schema_version,
            "stats": self.get_stats()
        }

    @abstractmethod
    def query(self, query_text: str, top_k: int = 10) -> List[dict]:
        """Execute query and return results."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict:
        """Return cassette statistics."""
        pass

    def _compute_hash(self) -> str:
        """Compute DB content hash for verification."""
        pass
```

### Handshake Response Format

```json
{
  "cassette_id": "governance",
  "db_path": "NAVIGATION/CORTEX/db/system1.db",
  "db_hash": "52146bf67044eb0a",
  "capabilities": ["vectors", "fts", "semantic_search"],
  "schema_version": "1.0",
  "stats": {
    "total_chunks": 1548,
    "with_vectors": 1235,
    "files": 132,
    "content_types": ["CANON", "ADRs", "SKILLS", "MAPS"]
  }
}
```

### Query Response Format

```json
{
  "chunk_id": "abc123",
  "path": "LAW/CANON/CONSTITUTION/PREAMBLE.md",
  "hash": "def456",
  "content": "The governance framework establishes...",
  "source": "governance",
  "score": 0.95
}
```

### Network Hub Interface

```python
class SemanticNetworkHub:
    """Central hub for cassette network."""

    def register_cassette(self, cassette: DatabaseCassette) -> Dict:
        """Register a new cassette in the network."""
        pass

    def query_all(self, query: str, top_k: int = 10) -> Dict[str, List[dict]]:
        """Query all cassettes and aggregate results."""
        pass

    def query_by_capability(self, query: str, capability: str, top_k: int = 10):
        """Query only cassettes with specific capability."""
        pass

    def get_network_status(self) -> Dict:
        """Get status of all registered cassettes."""
        pass
```

---

## Architecture Design

### Cassette Types

| Cassette | Content | Capabilities | Mutability |
|----------|---------|--------------|------------|
| `canon.db` | LAW/ bucket (constitutional docs) | vectors, fts | Immutable |
| `governance.db` | CONTEXT/decisions + preferences | vectors, fts, semantic_search | Stable |
| `capability.db` | CAPABILITY/ bucket (code, skills) | vectors, fts, ast | Mutable |
| `navigation.db` | NAVIGATION/ bucket (maps, metadata) | vectors, fts | Mutable |
| `direction.db` | DIRECTION/ bucket (roadmaps) | vectors, fts | Mutable |
| `thought.db` | THOUGHT/ bucket (research, lab) | vectors, fts, research | Mutable |
| `memory.db` | MEMORY/ bucket (archive, reports) | vectors, fts | Append-only |
| `inbox.db` | INBOX/ bucket (staging, temp) | vectors, fts | Ephemeral |
| `resident.db` | AI memories (per-agent) | vectors, fts, agent_id | Read-write |

### Query Routing

```
Query: "error handling patterns"
    │
    ▼
┌───────────────────────────────────┐
│  SemanticNetworkHub.query_all()   │
└───────────────────────────────────┘
    │
    ├──→ governance.db → 3 results (ADRs about error handling)
    ├──→ capability.db → 5 results (TOOLS/*.py implementations)
    ├──→ thought.db    → 2 results (research notes)
    │
    ▼
┌───────────────────────────────────┐
│    Merge + Rerank by Score        │
└───────────────────────────────────┘
    │
    ▼
Top 10 results with provenance + similarity scores
```

### Capability-Based Routing

```python
# Query only cassettes with "research" capability
results = hub.query_by_capability("memory architecture", "research", top_k=5)
# Returns: {"agi-research": [...], "thought": [...]}

# Query only cassettes with "ast" capability
results = hub.query_by_capability("parse function", "ast", top_k=5)
# Returns: {"capability": [...]}
```

---

## Token Compression

### Symbol Referencing

The cassette network enables massive token compression through symbol references:

**Without @Symbol Compression:**
- Full chunk content: ~2,400 characters per chunk
- 3,991 chunks x 2,400 = 9,578,400 characters
- Estimated tokens: 2,394,600 tokens

**With @Symbol Compression:**
- Symbol format: `@C:{hash_short}`
- 3,991 symbols x ~15 chars = 59,865 characters
- Estimated tokens: 14,966 tokens

**Token Savings:**
- Absolute savings: 2,379,634 tokens
- Percentage reduction: **99.4%**

### Symbol Resolution

```python
# Symbol format
@C:abc123  # Content symbol pointing to chunk hash
@P:def456  # Paper symbol pointing to research document
@A:ghi789  # Agent memory symbol

# Resolution
symbol_resolve("@C:abc123") → memory_recall("abc123")
```

---

## Cartridge-First Architecture

### Principles

1. **Portability**: Each cassette DB is a sharable, tool-readable unit
2. **Verifiability**: Content-hash based IDs, receipts for all writes
3. **Rebuildability**: Derived acceleration layers are disposable

### Derived Engines

Allowed as accelerators (never source of truth):
- **Qdrant** - Interactive ANN and concurrency
- **FAISS** - Local ANN
- **Lance/Parquet** - Analytics interchange

### Required Properties

```yaml
cassette:
  # Deterministic schema migrations (receipted)
  migration_receipt: sha256:abc123

  # Stable IDs (content-hash based)
  id_scheme: content-hash

  # Byte-identical text preservation
  encoding: utf-8
  normalization: NFC

  # Export/import with receipts
  export_format: cartridge
  receipt_chain: merkle
```

---

## MCP Integration

### Exposed Tools

```python
# Semantic search scoped to cassettes
semantic_search(query: str, cassettes: List[str], limit: int) -> List[dict]

# Cross-cassette federated search
cassette_network_query(query: str, limit: int) -> List[dict]

# Cassette statistics
cassette_stats() -> Dict[str, dict]

# Memory operations
memory_save(text: str, cassette: str, metadata: dict) -> str  # returns hash
memory_query(query: str, cassettes: List[str], limit: int) -> List[dict]
memory_recall(hash: str) -> dict

# Symbol resolution
symbol_resolve(symbol: str) -> dict
semantic_neighbors(hash: str, limit: int, cassettes: List[str]) -> List[dict]
```

---

## Network Configuration

```yaml
network:
  version: "1.0"
  protocol: "SNP"  # Semantic Network Protocol

cassettes:
  - id: governance
    path: NAVIGATION/CORTEX/db/governance.db
    type: local
    auto_index: true
    capabilities:
      - vectors
      - fts
      - semantic_search

  - id: agi-research
    path: D:/CCC 2.0/AI/AGI/CONTEXT/research/_generated/system1.db
    type: external
    auto_index: false
    capabilities:
      - research
      - fts

indexing:
  model: all-MiniLM-L6-v2
  dimensions: 384
  batch_size: 100

compression:
  symbol_format: "@C:{hash_short}"
  max_expansion_depth: 3
  lazy_expansion: true
```

---

## Dual-System Architecture Alignment

The cassette network maintains separation from System 2 (immutable ledger):

| Aspect | System 1 (Cassettes) | System 2 (Ledger) |
|--------|---------------------|-------------------|
| Purpose | Fast semantic search | Immutable audit trail |
| Mutability | Read-write (most) | Append-only |
| Query type | Fuzzy semantic | Exact hash lookup |
| Storage | SQLite per-cassette | Single ledger DB |
| Receipts | Per-write receipts | Chain of custody |

**Key Rule**: System 2 is NOT a cassette. This preserves ADR-027 dual-system architecture.

---

## Error Handling

### Cassette Offline

```python
try:
    results = cassette.query(query, top_k)
except CassetteOfflineError:
    # Skip this cassette, continue with others
    results = []
    log.warning(f"Cassette {cassette.cassette_id} offline, skipping")
```

### Graceful Degradation

- If a cassette is unavailable, queries continue with remaining cassettes
- Results include provenance so clients know which cassettes responded
- Network status endpoint reports health of all cassettes

---

## References

- [cassette_protocol.py](NAVIGATION/CORTEX/network/cassette_protocol.py)
- [network_hub.py](NAVIGATION/CORTEX/network/network_hub.py)
- [CASSETTE_NETWORK_ROADMAP.md](CASSETTE_NETWORK_ROADMAP.md)
- [AGS_ROADMAP_MASTER.md](AGS_ROADMAP_MASTER.md) - Phase 6
