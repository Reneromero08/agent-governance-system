---
uuid: 00000000-0000-0000-0000-000000000000
title: Roadmap Database Cassette Network
section: roadmap
bucket: 2025-12/Week-52
author: System
priority: High
created: 2025-12-28 12:00
modified: 2026-01-06 13:09
status: Active
summary: Roadmap for database and cassette network coverage (Restored)
tags:
- database
- cassette
- roadmap
hashtags: []
---
<!-- CONTENT_HASH: 63217c4bc1ac781689eabd7423c076db594fedfa719572997e41a8cdd9aff81e -->

# Roadmap: Database Cassette Network Architecture

**Status**: DRAFT
**Date**: 2025-12-28
**Scope**: Local implementation (foundation for future global protocol)
**Vision**: Modular database network where specialized databases (cassettes) plug into a semantic network for cross-database queries and token compression

---

## Executive Summary

Refactor from **monolithic database** approach to **database cassette network** architecture.

### Current Problems

**Problem 1: Can't index code without polluting governance DB**
- `cortex.build.py` only indexes `*.md` files (governance docs)
- TOOLS/*.py implementations invisible to semantic search
- Forces false choice: index everything OR governance-only

**Problem 2: AGI research DB isolated**
- `AGI/CORTEX/system1.db` exists but can't join queries
- No cross-project semantic search
- Manual context switching between projects

**Problem 3: Schema conflicts**
- Governance needs: FTS5, vectors, semantic search
- Code needs: AST metadata, function signatures, imports
- Research needs: Citations, paper metadata, authors
- One schema trying to fit all → complexity

### Target Architecture

Multiple specialized databases networked together like cassettes in a deck

Each cassette:
- Specialized for specific content type (docs, code, fixtures, research)
- Advertises capabilities via handshake protocol
- Schema-independent (network adapts)
- Hot-swappable (plug in/out as needed)
- Independently maintained and versioned

---

## Vision: The Cassette Deck

```
┌─────────────────────────────────────────────────────────┐
│         SEMANTIC NETWORK HUB (semantic_network.py)      │
│                                                         │
│  Protocol: HANDSHAKE → QUERY → RESPONSE → HEARTBEAT    │
└─────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┬───────────────┐
        │                 │                 │               │
   ┌────▼────┐       ┌────▼────┐      ┌────▼────┐    ┌────▼────┐
   │ 📼 GOV  │       │ 📼 CODE │      │ 📼 RSCH │    │ 📼 CNTR │
   │ Cassette│       │ Cassette│      │ Cassette│    │ Cassette│
   └─────────┘       └─────────┘      └─────────┘    └─────────┘

   Governance        Code Impl      AGI Research    Contracts
   ↓                 ↓              ↓               ↓
   gov.db           code.db        research.db     contracts.db
   - ADRs           - TOOLS/*.py   - Papers        - Fixtures
   - CANON          - *.json       - Theory        - Schemas
   - SKILLS         - *.js/.ts     - Experiments   - Tests
   1,548 chunks     TBD chunks     24 chunks       TBD chunks
   [vectors, fts]   [code, ast]    [research]      [fixtures]
```

---

## Phase 0: Proof of Concept (Days 1-3)

### Goal
Validate cassette network architecture works locally **before** committing to full migration.

### Deliverables
1. **Create code.db** alongside system1.db (no migration yet)
2. **Index TOOLS/*.py** into code.db with AST metadata
3. **Demo cross-database query** showing governance + code results
4. **Benchmark performance** vs single-DB approach

### Implementation

```python
# proof_of_concept.py

# Step 1: Create code cassette (doesn't touch system1.db)
from CORTEX.cassettes.code_cassette import CodeCassette
code = CodeCassette()  # Creates CORTEX/code.db
code.index_directory("TOOLS/", pattern="*.py")

# Step 2: Create simple network
from CORTEX.network_hub import SemanticNetworkHub
from CORTEX.cassettes.governance_cassette import GovernanceCassette

hub = SemanticNetworkHub()
hub.register_cassette(GovernanceCassette())  # Uses existing system1.db
hub.register_cassette(code)  # Uses new code.db

# Step 3: Cross-cassette query
results = hub.query_all("error handling", top_k=10)

print("Governance results:", len(results['governance']))
print("Code results:", len(results['code']))

# Step 4: Benchmark
import timeit
single_db_time = timeit("search_system1('error handling')", number=100)
network_time = timeit("hub.query_all('error handling')", number=100)
print(f"Single DB: {single_db_time}ms")
print(f"Network: {network_time}ms")
print(f"Overhead: {network_time - single_db_time}ms")
```

### Decision Gate

**Proceed to Phase 1 only if:**
- ✅ Cross-database queries return merged results
- ✅ Network overhead <20% OR absolute latency <100ms
- ✅ Code cassette can index Python with AST metadata
- ✅ No data corruption in either database

**If decision gate fails:**
- Reconsider architecture
- Maybe monolithic DB is simpler
- Document why cassette network doesn't work

### Success Criteria
- [ ] code.db created with 50+ Python chunks
- [ ] AST metadata extracted (function names, classes)
- [ ] Cross-database query returns governance + code results
- [ ] Performance acceptable (<20% overhead or <100ms absolute)
- [ ] Can remove code.db and revert to system1.db if needed

---

## Phase 1: Foundation (Week 1)

### 1.1 Define Cassette Interface

**Goal**: Standardize how cassettes register and communicate

**Deliverables**:
- `CORTEX/cassette_protocol.py` - Base cassette class
- `CORTEX/network_hub.py` - Central coordinator
- Protocol spec document

**Cassette Interface**:
```python
class DatabaseCassette:
    """Base class for all database cassettes."""

    def __init__(self, db_path: Path, cassette_id: str):
        self.db_path = db_path
        self.cassette_id = cassette_id
        self.capabilities = []
        self.schema_version = "1.0"

    def handshake(self) -> dict:
        """Return cassette metadata for network registration."""
        return {
            "cassette_id": self.cassette_id,
            "db_path": str(self.db_path),
            "db_hash": self._compute_hash(),
            "capabilities": self.capabilities,
            "schema_version": self.schema_version,
            "stats": self.get_stats()
        }

    def query(self, query_text: str, top_k: int = 10) -> List[dict]:
        """Execute query and return results."""
        raise NotImplementedError

    def get_stats(self) -> dict:
        """Return cassette statistics."""
        raise NotImplementedError

    def _compute_hash(self) -> str:
        """Compute DB content hash for verification."""
        raise NotImplementedError
```

**Network Hub**:
```python
class SemanticNetworkHub:
    """Central hub for cassette network."""

    def __init__(self):
        self.cassettes: Dict[str, DatabaseCassette] = {}
        self.protocol_version = "1.0"

    def register_cassette(self, cassette: DatabaseCassette):
        """Register a new cassette in the network."""
        handshake = cassette.handshake()
        self.cassettes[handshake['cassette_id']] = cassette
        print(f"[NETWORK] Registered cassette: {handshake['cassette_id']}")
        return handshake

    def query_all(self, query: str, top_k: int = 10) -> Dict[str, List[dict]]:
        """Query all cassettes and aggregate results."""
        results = {}
        for cassette_id, cassette in self.cassettes.items():
            results[cassette_id] = cassette.query(query, top_k)
        return results

    def query_by_capability(self, query: str, capability: str, top_k: int = 10):
        """Query only cassettes with specific capability."""
        results = {}
        for cassette_id, cassette in self.cassettes.items():
            if capability in cassette.capabilities:
                results[cassette_id] = cassette.query(query, top_k)
        return results
```

**Acceptance Criteria**:
- [ ] Base classes implemented
- [ ] Protocol version negotiation works
- [ ] Handshake returns all required metadata
- [ ] Network hub can register/query cassettes

---

### 1.2 Create Governance Cassette (Refactor Existing)

**Goal**: Convert existing system1.db to first cassette

**Deliverables**:
- `CORTEX/cassettes/governance_cassette.py`
- Renamed: `CORTEX/system1.db` → `CORTEX/governance.db`
- Updated builder to only index governance docs

**Implementation**:
```python
class GovernanceCassette(DatabaseCassette):
    """Cassette for governance documents (CANON, ADRs, SKILLS)."""

    def __init__(self):
        super().__init__(
            db_path=Path("CORTEX/governance.db"),
            cassette_id="governance"
        )
        self.capabilities = ["vectors", "fts", "semantic_search"]

    def query(self, query_text: str, top_k: int = 10):
        """Query governance docs with semantic search."""
        from semantic_search import SemanticSearch
        search = SemanticSearch(self.db_path)
        results = search.search(query_text, top_k)
        return [self._format_result(r) for r in results]

    def get_stats(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM chunks")
        total_chunks = cursor.fetchone()[0]
        cursor = conn.execute("SELECT COUNT(*) FROM section_vectors")
        with_vectors = cursor.fetchone()[0]
        conn.close()
        return {
            "total_chunks": total_chunks,
            "with_vectors": with_vectors,
            "content_types": ["CANON", "ADRs", "SKILLS", "MAPS"]
        }
```

**Content Scope**:
- CANON/* (constitutional docs)
- CONTEXT/decisions/* (ADRs)
- SKILLS/*/SKILL.md (skill manifests)
- MAPS/* (system maps)
- README.md, AGENTS.md, AGS_ROADMAP_MASTER.md

**Acceptance Criteria**:
- [ ] governance.db contains only governance content
- [ ] Implements DatabaseCassette interface
- [ ] Semantic search working
- [ ] Stats reporting accurate
- [ ] Can register with network hub

---

### 1.3 Create Test Fixtures

**Goal**: Integrate cassette network with existing CONTRACTS testing system

**Deliverables**:
- `CONTRACTS/fixtures/cassette-network/` directory
- Update `CONTRACTS/runner.py` to test protocol
- Fixtures for handshake, query, response validation

**Fixture Structure**:
```
CONTRACTS/fixtures/cassette-network/
├── protocol-handshake/
│   ├── input.json       # Cassette metadata
│   ├── expected.json    # Valid handshake response
│   └── validate.py      # Check protocol compliance
├── cross-cassette-query/
│   ├── input.json       # Query "governance + code"
│   ├── expected.json    # Merged results
│   └── validate.py      # Verify result merging
├── capability-filtering/
│   ├── input.json       # Query by capability
│   ├── expected.json    # Only matching cassettes
│   └── validate.py
└── network-stats/
    ├── input.json       # Request network statistics
    ├── expected.json    # All cassette stats
    └── validate.py
```

**Example Fixture** (protocol-handshake/input.json):
```json
{
  "description": "Test cassette handshake protocol",
  "cassette": {
    "db_path": "CORTEX/governance.db",
    "cassette_id": "governance"
  },
  "expected_capabilities": ["vectors", "fts", "semantic_search"],
  "expected_schema_version": "1.0"
}
```

**Integration with CONTRACTS/runner.py**:
```python
# Add to runner.py
def run_cassette_tests():
    """Run all cassette network protocol tests."""
    fixtures_dir = Path("CONTRACTS/fixtures/cassette-network")
    for fixture in fixtures_dir.glob("*/"):
        result = run_fixture(fixture)
        print(f"[{'PASS' if result == 0 else 'FAIL'}] {fixture.name}")
```

**Acceptance Criteria**:
- [ ] Fixtures created for all protocol messages
- [ ] `python CONTRACTS/runner.py` includes cassette tests
- [ ] All protocol tests pass
- [ ] Validation scripts check message format compliance
- [ ] Can run tests in CI/CD

---

## Phase 2: Code Cassette (Week 2)

### 2.1 Create Code Cassette

**Goal**: Separate cassette for Python/JSON/JS implementations

**Deliverables**:
- `CORTEX/cassettes/code_cassette.py`
- `CORTEX/code.db` (new database)
- `CORTEX/code_builder.py` (builder script)

**Content Scope**:
- TOOLS/*.py (all Python implementations)
- SKILLS/*/run.py, validate.py (skill runners)
- **/*.json (configs, schemas, fixtures)
- **/*.js, **/*.ts (JavaScript/TypeScript)
- **/*.yaml, **/*.yml (configs)

**Special Features**:
```python
class CodeCassette(DatabaseCassette):
    """Cassette for code implementations."""

    def __init__(self):
        super().__init__(
            db_path=Path("CORTEX/code.db"),
            cassette_id="code"
        )
        self.capabilities = ["vectors", "fts", "ast", "syntax_highlighting"]

    def query(self, query_text: str, top_k: int = 10):
        """Query code with semantic + AST-aware search."""
        # Use semantic search as base
        results = self._semantic_search(query_text, top_k)

        # Enhance with AST metadata if available
        for result in results:
            if result['file_type'] == '.py':
                result['ast_metadata'] = self._extract_ast_metadata(result)

        return results

    def _extract_ast_metadata(self, result):
        """Extract function/class names from Python code."""
        import ast
        try:
            tree = ast.parse(result['content'])
            return {
                "functions": [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)],
                "classes": [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            }
        except:
            return {}
```

**Code Chunking Strategy**:
- Python: Chunk by function/class (AST-aware)
- JSON: Chunk by top-level keys
- JavaScript: Chunk by function/class/module
- Config files: Chunk by section

**Acceptance Criteria**:
- [ ] code.db indexes all Python/JSON/JS files
- [ ] AST metadata extraction working
- [ ] Semantic search on code working
- [ ] Capability "ast" advertised
- [ ] Network hub can query code cassette

---

### 2.2 Create Contracts Cassette

**Goal**: Separate cassette for test fixtures and schemas

**Deliverables**:
- `CORTEX/cassettes/contracts_cassette.py`
- `CORTEX/contracts.db`
- `CORTEX/contracts_builder.py`

**Content Scope**:
- CONTRACTS/*.json (test fixtures)
- CONTRACTS/schemas/*.json (JSON schemas)
- CATALYTIC-DPT/TESTBENCH/* (test data)

**Special Features**:
```python
class ContractsCassette(DatabaseCassette):
    """Cassette for contracts, fixtures, and schemas."""

    def __init__(self):
        super().__init__(
            db_path=Path("CORTEX/contracts.db"),
            cassette_id="contracts"
        )
        self.capabilities = ["fixtures", "schemas", "validation"]

    def get_fixture(self, fixture_name: str):
        """Retrieve specific test fixture by name."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT content FROM chunks WHERE file_path LIKE ?",
            (f"%{fixture_name}%",)
        )
        result = cursor.fetchone()
        conn.close()
        return json.loads(result[0]) if result else None

    def validate_against_schema(self, data: dict, schema_name: str):
        """Validate data against stored JSON schema."""
        import jsonschema
        schema = self.get_fixture(f"schemas/{schema_name}")
        jsonschema.validate(data, schema)
        return True
```

**Acceptance Criteria**:
- [ ] contracts.db indexes fixtures and schemas
- [ ] Can retrieve fixtures by name
- [ ] Schema validation works
- [ ] Capability "fixtures" and "schemas" advertised

---

## Phase 3: Integration (Week 3)

### 3.1 Update Semantic Core to Use Network

**Goal**: Refactor semantic_search.py to query network instead of single DB

**Deliverables**:
- `CORTEX/network_search.py` (new)
- Update `vector_indexer.py` to index all cassettes
- Update `demo_semantic_dispatch.py` to use network

**Network Search**:
```python
class NetworkSemanticSearch:
    """Semantic search across all cassettes in network."""

    def __init__(self, hub: SemanticNetworkHub):
        self.hub = hub

    def search(self, query: str, top_k: int = 10, cassettes: List[str] = None):
        """Search across multiple cassettes and merge results."""
        if cassettes:
            # Query specific cassettes
            results = {c: self.hub.cassettes[c].query(query, top_k)
                      for c in cassettes if c in self.hub.cassettes}
        else:
            # Query all cassettes with semantic_search capability
            results = self.hub.query_by_capability(query, "semantic_search", top_k)

        # Merge and re-rank by similarity
        merged = []
        for cassette_id, cassette_results in results.items():
            for result in cassette_results:
                result['cassette'] = cassette_id
                merged.append(result)

        # Sort by similarity descending
        merged.sort(key=lambda r: r.get('similarity', 0), reverse=True)

        return merged[:top_k]

    def search_by_content_type(self, query: str, content_type: str, top_k: int = 10):
        """Search specific content type (code, docs, fixtures)."""
        cassette_map = {
            "docs": ["governance"],
            "code": ["code"],
            "fixtures": ["contracts"],
            "research": ["agi-research"]
        }
        cassettes = cassette_map.get(content_type, [])
        return self.search(query, top_k, cassettes)
```

**Acceptance Criteria**:
- [ ] NetworkSemanticSearch queries all cassettes
- [ ] Results merged and ranked correctly
- [ ] Can filter by content type
- [ ] Demo works with network search

---

### 3.2 Create Network Configuration

**Goal**: Central config for cassette topology

**Deliverables**:
- `CORTEX/network_config.yaml`
- Auto-discovery of local cassettes
- Support for remote cassettes (future)

**Config Format**:
```yaml
network:
  version: "1.0"
  protocol: "SNP"  # Semantic Network Protocol

cassettes:
  - id: governance
    path: CORTEX/governance.db
    type: local
    auto_index: true
    capabilities:
      - vectors
      - fts
      - semantic_search

  - id: code
    path: CORTEX/code.db
    type: local
    auto_index: true
    capabilities:
      - vectors
      - fts
      - ast
      - semantic_search

  - id: contracts
    path: CORTEX/contracts.db
    type: local
    auto_index: true
    capabilities:
      - fixtures
      - schemas
      - validation

  - id: agi-research
    path: D:/CCC 2.0/AI/AGI/CORTEX/_generated/system1.db
    type: external
    auto_index: false
    capabilities:
      - research

indexing:
  model: all-MiniLM-L6-v2
  dimensions: 384
  batch_size: 100

compression:
  symbol_format: "@C:{hash_short}"
  max_expansion_depth: 3
  lazy_expansion: true
```

**Acceptance Criteria**:
- [ ] YAML config loads successfully
- [ ] Auto-discovery of cassettes works
- [ ] Can disable auto-indexing for specific cassettes
- [ ] Network initializes from config

---

## Phase 4: Migration (Week 4)

### 4.1 Migrate Existing Data

**Goal**: Split current system1.db into specialized cassettes

**Process**:
1. **Backup**: Copy system1.db → system1.db.backup
2. **Extract**: Query chunks by content type
3. **Rebuild**: Create governance.db, code.db, contracts.db
4. **Verify**: Check row counts, integrity
5. **Reindex**: Run vector_indexer on each cassette
6. **Test**: Run full test suite

**Migration Script**:
```python
# CORTEX/migrate_to_cassettes.py

def migrate():
    # 1. Backup
    shutil.copy("CORTEX/system1.db", "CORTEX/system1.db.backup")

    # 2. Connect to source
    source = sqlite3.connect("CORTEX/system1.db")

    # 3. Create cassettes
    gov = GovernanceCassette()
    code = CodeCassette()
    contracts = ContractsCassette()

    # 4. Extract and insert
    for row in source.execute("SELECT * FROM chunks"):
        file_path = row['file_path']

        if is_governance_file(file_path):
            gov.insert_chunk(row)
        elif is_code_file(file_path):
            code.insert_chunk(row)
        elif is_contract_file(file_path):
            contracts.insert_chunk(row)

    # 5. Verify
    print(f"Governance: {gov.get_stats()}")
    print(f"Code: {code.get_stats()}")
    print(f"Contracts: {contracts.get_stats()}")

    # 6. Reindex vectors
    from vector_indexer import VectorIndexer
    for cassette in [gov, code, contracts]:
        indexer = VectorIndexer(cassette.db_path)
        indexer.index_all()
```

**Acceptance Criteria**:
- [ ] All data migrated without loss
- [ ] Row counts verified
- [ ] Vectors regenerated
- [ ] Tests passing
- [ ] Backup created

---

### 4.2 Update All Consumers

**Goal**: Update all code that queries databases to use network

**Files to Update**:
- `CORTEX/query.py` → use NetworkSemanticSearch
- `demo_semantic_dispatch.py` → use network
- `TOOLS/export_semantic.py` → query network
- Any skills that query CORTEX

**Example Update**:
```python
# Before
from semantic_search import SemanticSearch
search = SemanticSearch(Path("CORTEX/system1.db"))
results = search.search("error handling", top_k=5)

# After
from network_search import NetworkSemanticSearch
from network_hub import SemanticNetworkHub
from cassettes import load_cassettes_from_config

hub = SemanticNetworkHub()
load_cassettes_from_config(hub, "CORTEX/network_config.yaml")

search = NetworkSemanticSearch(hub)
results = search.search("error handling", top_k=5)

# Or search specific content type
code_results = search.search_by_content_type("error handling", "code", top_k=5)
```

**Acceptance Criteria**:
- [ ] All consumers updated
- [ ] Demo runs successfully
- [ ] Tests passing
- [ ] No references to old system1.db path

---

## Phase 5: Advanced Features (Week 5+)

### 5.1 Remote Cassettes

**Goal**: Support cassettes on different machines

**Features**:
- TCP/socket-based protocol (not just file-based)
- Cassette discovery via broadcast
- Authentication and authorization
- Encrypted connections

**Config Example**:
```yaml
cassettes:
  - id: research-remote
    host: 192.168.1.100
    port: 8765
    type: remote
    auth:
      method: token
      token_file: ~/.cassette_tokens/research.token
```

---

### 5.2 Cassette Versioning

**Goal**: Handle schema migrations across cassettes

**Features**:
- Semantic versioning for cassettes
- Migration scripts between versions
- Backward compatibility checks
- Version negotiation in handshake

---

### 5.3 Lazy Loading & Caching

**Goal**: Optimize network queries

**Features**:
- Cache frequently accessed chunks
- Lazy load vectors only when needed
- Prefetch related chunks
- LRU cache eviction

---

### 5.4 MCP Integration

**Goal**: Expose cassette network via Model Context Protocol

**Features**:
- MCP server exposes cassettes as resources
- Claude can query network directly
- Supports tool calls for search/retrieval
- Cross-cassette queries via MCP

---

## Success Metrics

### Performance
- [ ] Search latency: <100ms across all cassettes
- [ ] Indexing throughput: >100 chunks/sec per cassette
- [ ] Network overhead: <10ms per cassette query

### Compression
- [ ] Maintain 96%+ token reduction
- [ ] Symbol expansion: <50ms average
- [ ] Cross-cassette references work

### Reliability
- [ ] 100% test coverage for protocol
- [ ] Zero data loss in migration
- [ ] Graceful degradation (cassette offline → skip)

### Usability
- [ ] Config-driven setup (no code changes)
- [ ] Auto-discovery works
- [ ] Clear error messages

---

## Risks & Mitigations

### Risk 1: Data Loss During Migration
**Mitigation**:
- Comprehensive backups before migration
- Dry-run mode in migration script
- Verification checksums

### Risk 2: Performance Degradation
**Mitigation**:
- Benchmark before/after
- Optimize hot paths
- Add caching layer

### Risk 3: Complexity Increase
**Mitigation**:
- Clear documentation
- Simple default config
- Backward compatibility layer (deprecated)

---

## Timeline Summary

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| **Phase 0** | Days 1-3 | Proof of concept (code.db + benchmarks) |
| **Phase 1** | Week 1 | Protocol + Governance cassette |
| **Phase 2** | Week 2 | Code + Contracts cassettes |
| **Phase 3** | Week 3 | Network integration |
| **Phase 4** | Week 4 | Migration + Updates |
| **Phase 5** | Week 5+ | Advanced features (local MCP, caching) |

**Total**: 4-5 weeks for local implementation

---

## Open Questions

1. **Should system2.db become a cassette too?**
   - **Decision: NO** - Keep system2.db separate from cassette network
   - Reason: System 2 is immutable ledger (slow/strict), cassettes are System 1 (fast/fuzzy)
   - Preserves ADR-027 dual-system architecture

2. **How to handle cassette conflicts?**
   - Same hash in multiple cassettes → prefer first result, mark duplicates
   - Or: Merge results with provenance metadata

3. **Cassette discovery protocol (local)?**
   - Config file only (CORTEX/network_config.yaml)
   - Auto-scan CORTEX/*.db for cassettes
   - Future: Network discovery for global protocol

4. **Should cassettes support write operations?**
   - Phase 0-4: Read-only
   - Future: Distributed writes with conflict resolution

---

## Future: Global Protocol

**Vision**: Cassette Network Protocol becomes internet-scale standard for semantic knowledge sharing.

### Architecture at Global Scale

```
┌──────────────────────────────────────────────────────┐
│      CASSETTE DISCOVERY NETWORK (like DNS)           │
└──────────────────────────────────────────────────────┘
           │
    ┌──────┼──────────┬──────────────┬───────────────┐
    │                 │              │               │
┌───▼────┐      ┌────▼────┐    ┌────▼────┐     ┌────▼────┐
│  Your  │      │   MIT   │    │ OpenAI  │     │  Other  │
│  Node  │      │Research │    │ Papers  │     │  Nodes  │
└────────┘      └─────────┘    └─────────┘     └─────────┘
```

### Protocol Evolution

**Phase 6: Internet Protocol (Months 3-6)**
- Real networking (TCP/IP, not file-based)
- Cassette URIs: `snp://example.com/cassette-id`
- Federated registry (like npm, PyPI)
- Public key infrastructure (PKI) for trust

**Phase 7: P2P Discovery (Months 6-12)**
- DHT-based discovery (no central registry)
- Peer-to-peer protocol
- Incentive mechanism (tokens or reciprocal sharing)

**Phase 8: Global Network (Year 2+)**
- Multi-language SDKs (Python, JS, Rust, Go)
- Public cassettes (Wikipedia, ArXiv, GitHub)
- DAO governance for protocol evolution
- RFC-style specification published

### Key Technical Challenges

1. **Trust Model**
   - All responses signed with cassette private key
   - Public key registry
   - Reputation system for cassette quality

2. **Economics**
   - Who pays for bandwidth/compute?
   - Options: Free public good, token market, reciprocal sharing

3. **Security**
   - Rate limiting (prevent DOS)
   - Sandbox responses (no code execution)
   - Proof-of-work for spam prevention

4. **Discovery**
   - Federated registry vs fully P2P
   - Capability-based search
   - Geographic routing

### Why This Could Work

- **Git for knowledge**: Like Git distributed code, cassettes distribute knowledge
- **Token efficiency**: 96%+ compression makes queries cheap
- **Semantic web, actually usable**: Vectors + @symbols vs broken RDF/OWL
- **AI-native**: Designed for LLM context, not human browsing

### Reference for Future

When ready for global protocol:
1. Extract local implementation as reference
2. Write RFC-style SNP specification
3. Create public cassettes (Wikipedia, ArXiv)
4. Deploy registry server
5. Publish whitepaper
6. Open source everything (MIT/Apache 2.0)

---

**Status**: Ready for review and implementation
**Next Step**: Review roadmap → Start Phase 0 proof of concept