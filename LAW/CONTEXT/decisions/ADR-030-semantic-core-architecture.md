---
id: "ADR-030"
title: "Semantic Core + Translation Layer Architecture"
status: "Proposed"
date: "2025-12-28"
confidence: "Low"
impact: "High"
tags: ["architecture", "semantics", "swarm", "token-optimization"]
---

<!-- CONTENT_HASH: UPDATE_PENDING -->

# ADR-030: Semantic Core + Translation Layer Architecture

**Deciders:** System Architect
**Related:** ADR-027 (Dual-DB), ADR-028 (Semiotic Compression Layer)

## Context

Current swarm uses uniform models for all agents. Inefficient:
- **Opus**: Expensive, slow, deep semantic understanding
- **Haiku**: Cheap, fast, lacks contextual depth

Need: Big model as "brain" (semantic core), small models as "hands" (translation layer).

## Decision

```
┌─────────────────────────────────────────────────────────────┐
│                    SEMANTIC CORE (Opus)                      │
│  • CORTEX + Vector Store (embeddings, @Symbol refs)          │
│  • Full codebase understanding                               │
│  • Compresses tasks to symbols + vectors                     │
└─────────────────────────────────────────────────────────────┘
                              │
                    Compressed IR (Symbols + Vectors)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  TRANSLATION LAYER (Haiku)                   │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐                │
│  │  Ant-1    │  │  Ant-2    │  │  Ant-N    │                │
│  │ @Symbols  │  │ @Symbols  │  │ @Symbols  │  → Code/Actions│
│  └───────────┘  └───────────┘  └───────────┘                │
└─────────────────────────────────────────────────────────────┘
```

## Architecture

### 1. Semantic Core
- **Embeddings**: `all-MiniLM-L6-v2` (384-dim, local, fast)
- **Storage**: SQLite `section_vectors` table + index
- **Symbol expansion**: Content + hash + semantic neighbors + embedding

### 2. Translation Layer (Compressed Task Protocol)
```json
{
    "task_id": "refactor-001",
    "symbols": {
        "@target": {"content": "...", "hash": "a1b2c3d4", "file": "server.py", "lines": [1159, 1227]}
    },
    "vectors": {"task_intent": [...], "context_centroid": [...]},
    "instruction": "Add input validation to @target",
    "constraints": {"max_changes": 50, "preserve_signature": true}
}
```

Ant behavior: Receive symbols → Resolve → Execute transformation → Return result (no semantic reasoning).

### 3. Communication Flow
1. **User request** → Opus parses intent, searches CORTEX, decomposes to atomic tasks
2. **Governor** routes compressed task to available Ant
3. **Ant** resolves symbols, applies vectors, executes, returns structured result
4. **Opus** validates, integrates, updates CORTEX, continues or completes

## Token Economics

| Mode | Context per worker | Tasks | Total | Cost |
|------|-------------------|-------|-------|------|
| Without Core | 50,000 | 10 | 500,000 | $$$$ |
| With Core | 2,000 (Opus: 100k once) | 10 | 120,000 | $ |

**80% token reduction** through semantic compression.

## Integration

- **CORTEX**: Add `section_vectors` table, extend builder with embeddings
- **ADR-028**: Extend compression: `σ(content) → {@Symbol, vector}`
- **MCP**: Add `resolve_symbols`, `get_embeddings` tools

## Consequences

**Positive**: 80% token reduction, faster parallel execution, better accuracy, scalable
**Negative**: Initial indexing cost, symbol resolution complexity, embedding lookup latency
**Risks**: Model drift, stale references, vector fragmentation

## Implementation

| Component | Recommendation |
|-----------|---------------|
| Embedding | `all-MiniLM-L6-v2` (local, fast, good enough) |
| Storage | SQLite + numpy, upgrade to FAISS if needed |

## Related
- ADR-027 (Dual-DB), ADR-028 (Semiotic Compression), ADR-∞ (Living Formula)
