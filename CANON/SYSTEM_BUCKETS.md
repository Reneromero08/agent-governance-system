# SYSTEM BUCKETS

**Authority:** CANON  
**Version:** 1.0.0

This document defines the classification buckets used throughout the system.
**All artifacts MUST belong to exactly one bucket.**

---

## 1. LAW

**Purpose:** Define what is allowed.

**Includes:**
- `CANON/*`
- `CONTRACTS/*`
- `AGENTS.md`
- `CONTEXT/decisions/*`

**Prohibitions:**
- No execution logic
- No speculation
- No summaries that alter meaning

**Operations:**
- Append-only
- Change requires explicit authority

---

## 2. CAPABILITY

**Purpose:** Define what the system can do.

**Includes:**
- `SKILLS/*`
- `TOOLS/*`
- `MCP/*`
- `PRIMITIVES/*`
- `PIPELINES/*`

**Prohibitions:**
- No authority
- No direction
- No self-modifying behavior

**Operations:**
- Implement
- Test
- Version

---

## 3. NAVIGATION

**Purpose:** Define how information is found.

**Includes:**
- `CORTEX/*`
- `CONTEXT/maps/*`
- `INDEX.md` files (anywhere)

**Prohibitions:**
- No decisions
- No execution
- No authority claims

**Operations:**
- Index
- Query
- Resolve

---

## 4. DIRECTION

**Purpose:** Define what should be done next.

**Includes:**
- `AGS_ROADMAP_MASTER.md`
- `*_ROADMAP*.md` files
- `INBOX/roadmaps/*`
- Active plans

**Prohibitions:**
- No enforcement
- No execution

**Operations:**
- Revise
- Archive when superseded

---

## 5. THOUGHT

**Purpose:** Explore possibilities.

**Includes:**
- `CONTEXT/research/*`
- `CATALYTIC-DPT/LAB/*`
- `CONTEXT/demos/*`
- Experiments
- Notes
- Drafts

**Prohibitions:**
- No binding force
- No implicit decisions

**Operations:**
- Freeform writing
- Contradiction allowed

---

## 6. MEMORY

**Purpose:** Record what has happened.

**Includes:**
- `CONTEXT/archive/*`
- `CONTEXT/session_reports/*`
- `INBOX/reports/*`
- `MEMORY/*`
- Snapshots
- Historical records

**Prohibitions:**
- No future intent
- No edits after archival

**Operations:**
- Reference only

---

## Directory → Bucket Mapping

| Directory | Bucket |
|-----------|--------|
| `CANON/` | LAW |
| `CONTRACTS/` | LAW |
| `AGENTS.md` | LAW |
| `CONTEXT/decisions/` | LAW |
| `SKILLS/` | CAPABILITY |
| `TOOLS/` | CAPABILITY |
| `MCP/` | CAPABILITY |
| `PRIMITIVES/` | CAPABILITY |
| `PIPELINES/` | CAPABILITY |
| `CORTEX/` | NAVIGATION |
| `CONTEXT/maps/` | NAVIGATION |
| `AGS_ROADMAP_MASTER.md` | DIRECTION |
| `INBOX/roadmaps/` | DIRECTION |
| `CONTEXT/research/` | THOUGHT |
| `CATALYTIC-DPT/LAB/` | THOUGHT |
| `CONTEXT/demos/` | THOUGHT |
| `CONTEXT/archive/` | MEMORY |
| `CONTEXT/session_reports/` | MEMORY |
| `INBOX/reports/` | MEMORY |
| `MEMORY/` | MEMORY |

---

## Enforcement

Agents MUST:
1. Identify the bucket before modifying any artifact
2. Respect the prohibitions for that bucket
3. Use only the allowed operations for that bucket
4. Ask for clarification if bucket is ambiguous
