---
title: "Agent Search Protocol"
status: "Active"
priority: "High"
created: "2026-01-02"
tags: ["governance", "agents", "search", "tools"]
---

# Agent Search Protocol

**Purpose:** Define when agents should use semantic search vs keyword search to maximize efficiency and leverage the vector index.

---

## Search Tool Decision Tree

```
┌─────────────────────────────────────────┐
│   What are you looking for?             │
└─────────────────┬───────────────────────┘
                  ↓
        ┌─────────────────────┐
        │ Exact string/path?  │
        └─────────┬───────────┘
                  │
         ┌────────┴────────┐
         │ YES             │ NO
         ↓                 ↓
    ┌─────────┐      ┌──────────────┐
    │ grep    │      │ Conceptual?  │
    │ search  │      └──────┬───────┘
    └─────────┘             │
                   ┌────────┴────────┐
                   │ YES             │ NO
                   ↓                 ↓
            ┌──────────────┐   ┌──────────┐
            │ semantic     │   │ Try both │
            │ search       │   │ (semantic│
            └──────────────┘   │ first)   │
                               └──────────┘
```

---

## Rule 1: Use Semantic Search for Conceptual Queries

**When to use:**
- Finding documents by topic ("How does compression work?")
- Discovering related concepts ("What's similar to catalytic computing?")
- Exploring connections ("What relates to vector execution?")
- Research questions ("What papers discuss RL for LLMs?")

**Tool:** `mcp_ags-mcp-server_semantic_search`

**Example:**
```
Query: "tiny model training with validator reward signal"
→ Finds: SEMIOTIC_COMPRESSION.md, TINY_COMPRESS_ROADMAP.md
```

---

## Rule 2: Use Keyword Search for Exact Matches

**When to use:**
- Finding exact strings ("SPECTRUM-02", "CMP-01")
- Locating file paths ("MEMORY/LLM_PACKER")
- Debugging (need exact line numbers)
- Code references (function names, class names)

**Tool:** `grep_search`

**Example:**
```
Query: "SPECTRUM-02"
→ Finds: All files with exact string "SPECTRUM-02"
```

---

## Rule 3: Use Cortex Query for Structured Lookups

**When to use:**
- Finding files by path pattern
- Looking up entities (ADRs, skills, tools)
- Navigating the file index

**Tool:** `mcp_ags-mcp-server_cortex_query`

**Example:**
```
Query: "ADR-027"
→ Finds: LAW/CONTEXT/decisions/ADR-027-dual-db-architecture.md
```

---

## Rule 4: Hybrid Approach for Ambiguous Queries

**When to use:**
- Query could be exact or conceptual
- Not sure which tool will work best

**Strategy:**
1. Try semantic search first (broader coverage)
2. If results are too vague, fall back to keyword search
3. If both fail, try Cortex query

**Example:**
```
Query: "LLM Packer"
→ Try semantic search first (finds conceptual docs)
→ If too broad, use grep_search for exact "LLM_PACKER" references
```

---

## Rule 5: Prefer Semantic Search for Initial Exploration

**Rationale:** Semantic search leverages the vector index (96% token savings) and discovers connections that keyword search misses.

**Exception:** If you need exact line numbers or debugging info, use keyword search immediately.

---

## Enforcement

### For Agents (AGENTS.md)
- **MUST** use semantic search for conceptual queries
- **MUST** use keyword search for exact string matches
- **SHOULD** try semantic search first for ambiguous queries
- **MAY** use hybrid approach if unsure

### For Humans (Code Review)
- Review agent search patterns in session logs
- Flag inefficient search strategies (e.g., keyword search for "how does X work?")
- Suggest semantic search when appropriate

---

## Metrics

Track search efficiency:
- **Semantic search hit rate:** % of queries that return useful results
- **Keyword search precision:** % of exact matches found
- **Hybrid search success:** % of ambiguous queries resolved

**Goal:** 90%+ of conceptual queries use semantic search, 90%+ of exact matches use keyword search.

---

## Examples

| Query | Tool | Rationale |
|-------|------|-----------|
| "How does SPECTRUM work?" | Semantic | Conceptual |
| "SPECTRUM-02" | Keyword | Exact string |
| "ADR-027" | Cortex | Structured lookup |
| "compression strategy" | Semantic | Conceptual |
| "MEMORY/LLM_PACKER" | Keyword | Exact path |
| "tiny model training" | Semantic | Conceptual |
| "crystallized intelligence" | Semantic | Conceptual (even if exact phrase not in docs) |
| "def verify_bundle" | Keyword | Exact function name |

---

## References

- **Semantic Search:** `mcp_ags-mcp-server_semantic_search` (384-dim vectors, cosine similarity)
- **Keyword Search:** `grep_search` (ripgrep, exact matches)
- **Cortex Query:** `mcp_ags-mcp-server_cortex_query` (file/entity index)

---

## Changelog

- **2026-01-02:** Initial version (codifying search strategy)
