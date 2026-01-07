<!-- CONTENT_HASH: 36e1b2bc23a1f9d5e7c8a9b2d3f4e5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2 -->

# SYSTEM BUCKETS

**Authority:** LAW/CANON  
**Version:** 2.0.0 (6-Bucket Migration)

This document defines the classification buckets of the Agent Governance System (AGS).
**All files and artifacts MUST belong to exactly one bucket.**

---

## 1. LAW (Supreme Authority)

**Purpose:** Define what is allowed. The "Constitution" and "Legislation" of the system.

**Key Directories:**
- `LAW/CANON/` - Constitutional rules (Invariants, Contract, Glossary).
- `LAW/CONTEXT/` - Decision trace (ADRs) and Preferences.
- `LAW/CONTRACTS/` - Mechanical enforcement (Schemas, Fixtures, Precedent).
- `AGENTS.md` - Operating contract.

**Prohibitions:**
- No execution logic (scripts).
- No speculative research.
- No history distortion.

---

## 2. CAPABILITY (Instruments)

**Purpose:** Define what the system can do. The "Tools" and "Skills" of the agents.

**Key Directories:**
- `CAPABILITY/SKILLS/` - Atomic agent toolkits.
- `CAPABILITY/TOOLS/` - Helper scripts, critics, and automation.
- `CAPABILITY/MCP/` - Client adapters and Semantic Core logic.
- `CAPABILITY/PIPELINES/` - DAG definitions.
- `CAPABILITY/PRIMITIVES/` - Low-level execution logic.
- `CAPABILITY/TESTBENCH/` - Validation suites.

**Prohibitions:**
- No self-authored governance rules.
- No navigation planning (Roadmaps).

---

## 3. NAVIGATION (Direction & Cortex)

**Purpose:** Define where we are going and how to find things. Consolidates the old **DIRECTION** and **CORTEX** buckets.

**Key Directories:**
- `NAVIGATION/MAPS/` - Ownership and data flow maps.
- `NAVIGATION/CORTEX/` - Semantic index and metadata.

**Operations:**
- **Index**: Build semantic models.
- **Map**: Define repo boundaries.
- **Orient**: Update roadmaps to reflect completion.

---

## 4. MEMORY (Historical Trace)

**Purpose:** Record what has happened across sessions.

**Key Directories:**
- `MEMORY/LLM_PACKER/` - Context compression and archive storage.
- `LAW/CONTEXT/archive/` - Archived decision history.
- `LAW/CONTRACTS/_runs/` - Temporary logs/outputs (Disposable but traceable).

**Prohibitions:**
- No new planning (Roadmaps).
- No modified rules.

---

## 5. THOUGHT (Experimental Labs)

**Purpose:** Explore possibilities and build prototypes without risking system stability.

**Key Directories:**
- `THOUGHT/LAB/` - Volatile features (e.g., `CAT_CHAT`, `NEO3000`, `TURBO_SWARM`).
- `THOUGHT/CONTEXT/` - Lab-specific research and notes.

**Prohibitions:**
- No binding force on the main system.
- No production dependencies.

---

## 6. INBOX (Human Gate)

**Purpose:** Centralized location for artifacts requiring human review or "God Mode" approval.

**Key Directories:**
- `INBOX/reports/` - Mandatory completion reports.
- `INBOX/research/` - In-progress study and findings.
- `INBOX/roadmaps/` - Roadmaps currently under review.
- `INBOX/decisions/` - Proposed policy/ADR changes.

---

## Directory → Bucket Mapping (V2)

| Directory | Bucket | Authority |
|-----------|--------|-----------|
| `LAW/CANON/` | **LAW** | SUPREME |
| `LAW/CONTEXT/` | **LAW** | SUPREME |
| `LAW/CONTRACTS/` | **LAW** | SUPREME |
| `AGENTS.md` | **LAW** | SUPREME |
| `CAPABILITY/SKILLS/` | **CAPABILITY** | INSTRUMENT |
| `CAPABILITY/TOOLS/` | **CAPABILITY** | INSTRUMENT |
| `CAPABILITY/MCP/` | **CAPABILITY** | INSTRUMENT |
| `CAPABILITY/TESTBENCH/` | **CAPABILITY** | INSTRUMENT |
| `NAVIGATION/MAPS/` | **NAVIGATION** | DIRECTION |
| `NAVIGATION/CORTEX/` | **NAVIGATION** | DIRECTION |
| `MEMORY/LLM_PACKER/` | **MEMORY** | HISTORY |
| `MEMORY/_packs/` | **MEMORY** | HISTORY |
| `THOUGHT/LAB/` | **THOUGHT** | EXPERIMENT |
| `INBOX/` | **INBOX** | GATE |

---

## Enforcement

Agents MUST:
1. Identify the target bucket before creating or moving a file.
2. Use the correct prefix (`LAW/`, `CAPABILITY/`, etc.) for all new components.
3. Treat files in `INBOX` as "Requests for Review" until moved to their final bucket.
