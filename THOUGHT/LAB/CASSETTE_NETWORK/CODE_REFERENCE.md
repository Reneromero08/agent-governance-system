# Cassette Network Code Reference

This document provides quick access to all implementation code for the Cassette Network.

---

## Core Protocol

### cassette_protocol.py
**Path:** [NAVIGATION/CORTEX/network/cassette_protocol.py](../../NAVIGATION/CORTEX/network/cassette_protocol.py)

Base class for all database cassettes. Defines the standard contract:
- `handshake()` - Returns metadata for network registration
- `query(query_text, top_k)` - Execute search
- `get_stats()` - Return cassette statistics
- `_compute_hash()` - Compute DB content hash

### network_hub.py
**Path:** [NAVIGATION/CORTEX/network/network_hub.py](../../NAVIGATION/CORTEX/network/network_hub.py)

Central coordinator for cassette network:
- `register_cassette()` - Register cassette via handshake
- `query_all()` - Query all cassettes
- `query_by_capability()` - Query cassettes with specific capability
- `get_network_status()` - Health monitoring

---

## Cassette Implementations

### governance_cassette.py
**Path:** [NAVIGATION/CORTEX/network/cassettes/governance_cassette.py](../../NAVIGATION/CORTEX/network/cassettes/governance_cassette.py)

Wraps `system1.db` for governance documents:
- **Capabilities:** vectors, fts, semantic_search
- **Content:** CANON, ADRs, SKILLS, MAPS
- **Database:** `NAVIGATION/CORTEX/db/system1.db`

### agi_research_cassette.py
**Path:** [NAVIGATION/CORTEX/network/cassettes/agi_research_cassette.py](../../NAVIGATION/CORTEX/network/cassettes/agi_research_cassette.py)

Wraps AGI research database:
- **Capabilities:** research, fts
- **Content:** Papers, Experiments, Theory
- **Database:** `D:/CCC 2.0/AI/AGI/CONTEXT/research/_generated/system1.db`

### cat_chat_cassette.py
**Path:** [NAVIGATION/CORTEX/network/cassettes/cat_chat_cassette.py](../../NAVIGATION/CORTEX/network/cassettes/cat_chat_cassette.py)

Wraps CAT_CHAT documentation:
- **Capabilities:** vectors, fts, semantic_search, indexing_info
- **Content:** CAT_CHAT docs, indexing info, merge analysis
- **Database:** `THOUGHT/LAB/CAT_CHAT/cat_chat_index.db`

---

## Demo & Testing

### demo_cassette_network.py
**Path:** [NAVIGATION/CORTEX/network/demo_cassette_network.py](../../NAVIGATION/CORTEX/network/demo_cassette_network.py)

Demonstrates cross-cassette queries between AGS governance and AGI research.

### test_cassettes.py
**Path:** [NAVIGATION/CORTEX/network/test_cassettes.py](../../NAVIGATION/CORTEX/network/test_cassettes.py)

Test suite for cassette network functionality.

---

## Configuration Files

### cassettes.json
**Path:** [NAVIGATION/CORTEX/network/cassettes.json](../../NAVIGATION/CORTEX/network/cassettes.json)

JSON configuration for registered cassettes.

### generic_cassette.py
**Path:** [NAVIGATION/CORTEX/network/generic_cassette.py](../../NAVIGATION/CORTEX/network/generic_cassette.py)

Creates cassettes from JSON configuration without writing separate Python files:
- `GenericCassette` class - Auto-detects query strategy (FTS5, text search)
- `load_cassettes_from_json()` - Loads all cassettes from config file
- `create_cassette_from_config()` - Creates individual cassettes

---

## Original SNP Prototype

### semantic_network.py
**Path:** [CAPABILITY/TOOLS/cortex/semantic_network.py](../../CAPABILITY/TOOLS/cortex/semantic_network.py)

The original Semantic Network Protocol (SNP) prototype - predecessor to the cassette network:
- `CortexPeer` - Original peer class (replaced by `DatabaseCassette`)
- `CortexNetwork` - Original network class (replaced by `SemanticNetworkHub`)
- Message types: HANDSHAKE, QUERY, RESPONSE, HEARTBEAT

**Note:** This is the legacy implementation. Use `cassette_protocol.py` and `network_hub.py` instead.

---

## MCP Integration

### server.py
**Path:** [CAPABILITY/MCP/server.py](../../CAPABILITY/MCP/server.py)

MCP server with cassette network integration:
- `semantic_search` tool
- `cassette_network_query` tool (planned)
- `memory_save/query/recall` tools (planned)

### semantic_adapter.py
**Path:** [CAPABILITY/MCP/semantic_adapter.py](../../CAPABILITY/MCP/semantic_adapter.py)

Adapter that connects cassette network to MCP server:
- `SemanticMCPAdapter` class - Main adapter
- `semantic_search_tool()` - Vector-based search via MCP
- `cassette_network_query()` - Cross-cassette queries via MCP
- `_load_cassettes_from_config()` - Loads cassettes from JSON config
- `get_network_status()` - Returns network health status

---

## File Structure

```
NAVIGATION/CORTEX/network/
├── cassette_protocol.py     # Base class (DatabaseCassette)
├── network_hub.py           # Hub coordinator (SemanticNetworkHub)
├── demo_cassette_network.py # Demo script
├── test_cassettes.py        # Tests
├── cassettes.json           # Config (defines registered cassettes)
├── generic_cassette.py      # JSON-configured cassettes
└── cassettes/
    ├── __init__.py
    ├── governance_cassette.py
    ├── agi_research_cassette.py
    └── cat_chat_cassette.py

CAPABILITY/MCP/
├── server.py                # MCP server
└── semantic_adapter.py      # Cassette network MCP adapter

CAPABILITY/TOOLS/cortex/
└── semantic_network.py      # Original SNP prototype (legacy)
```

---

## Usage Example

```python
from network_hub import SemanticNetworkHub
from cassettes import GovernanceCassette, AResearchCassette

# Create hub
hub = SemanticNetworkHub()

# Register cassettes
hub.register_cassette(GovernanceCassette())
hub.register_cassette(AResearchCassette())

# Cross-cassette query
results = hub.query_all("governance architecture", top_k=5)

# Capability-based query
research_only = hub.query_by_capability("memory systems", "research", top_k=5)

# Network status
status = hub.get_network_status()
```
