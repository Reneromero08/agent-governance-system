---
uuid: 00000000-0000-0000-0000-000000000000
title: Phase 5 Vector/Symbol Integration Research Findings
section: report
bucket: reports
author: System
priority: High
created: 2026-01-07
modified: 2026-01-07
status: Active
summary: Consolidated research findings for Phase 5 Vector/Symbol Integration
tags:
- phase-5
- vector
- semiotic
- research
- report
---
<!-- CONTENT_HASH: 92d23ae6f61041936c90e07545c55ed050ed310e5c7f97f75c469ea8e904d189 -->

# Phase 5: Vector/Symbol Integration - Research Findings

**Date:** 2026-01-07
**Scope:** Consolidated research from scattered sources across the codebase
**Status:** Research Complete, Implementation Pending

---

## Executive Summary

Phase 5 (Vector/Symbol Integration) is the next major phase in the AGS roadmap after the completion of Phase 4 (Catalytic Architecture). It consists of two sub-phases:

1. **5.1 - Vector Indexing:** Embed canon, ADRs, and skills into vector space for semantic addressability
2. **5.2 - Semiotic Compression Layer (SCL):** Macro language for 90%+ token reduction on governance repetition

**Critical Dependency:** Phase 6.0 (Cassette Network) depends on Phase 5.2's `MemoryRecord` contract.

---

## 1. Current State Assessment

### What Exists (Research & Design)

| Artifact | Location | Status |
|----------|----------|--------|
| Main roadmap tasks | `AGS_ROADMAP_MASTER.md` lines 211-242 | Defined |
| MemoryRecord contract | `INBOX/reports/V4/01-06-2026-21-13_5_2_VECTOR_SUBSTRATE_VECTORPACK.md` | Specified |
| VectorPack format | `INBOX/reports/V4/01-06-2026-21-13_5_2_VECTOR_SUBSTRATE_VECTORPACK.md` | Specified |
| SCL specification | `INBOX/2025-12/Week-01/12-29-2025-07-01_SEMIOTIC_COMPRESSION.md` | Complete spec |
| Symbolic compression research | `INBOX/2025-12/Week-52/12-26-2025-06-39_SYMBOLIC_COMPRESSION.md` | Initial research |
| Catalytic continuity design | `INBOX/reports/V4/01-06-2026-21-13_CAT_CHAT_CATALYTIC_CONTINUITY.md` | Design spec |

### What's Missing (Implementation)

| Component | Expected Location | Status |
|-----------|-------------------|--------|
| `vector_index.py` | `CAPABILITY/PRIMITIVES/` | Not created |
| `scl_codebook.py` | `CAPABILITY/PRIMITIVES/` | Not created |
| `scl_decoder.py` | `CAPABILITY/PRIMITIVES/` | Not created |
| `SCL/CODEBOOK.json` | TBD | Not created |
| `scl` CLI | `CAPABILITY/TOOLS/scl/` | Not created |
| Phase 5 tests | `CAPABILITY/TESTBENCH/integration/` | Not created |
| Canonical specs | `LAW/CANON/SEMANTIC/`, `LAW/CANON/VECTOR/` | Not created |

---

## 2. Phase 5.1: Vector Indexing

### 2.1.1 Scope

Embed governance artifacts into vector space for semantic retrieval:

- **Canon files:** `LAW/CANON/*` - All canonical governance documents
- **ADRs:** `LAW/CONTEXT/decisions/*` - Architecture decision records
- **Skills:** `CAPABILITY/SKILLS/*/SKILL.md` - Skill metadata for discovery

### 2.1.2 Objectives

| Task ID | Description | Exit Criteria |
|---------|-------------|---------------|
| 5.1.1 | Embed all canon files | Vectors stored in CAS-indexed format |
| 5.1.2 | Embed all ADRs | Vectors stored with deterministic rebuild |
| 5.1.3 | Store model weights in vector-indexed CAS | Weights addressable by hash |
| 5.1.4 | Semantic skill discovery | Find skills by description similarity |
| 5.1.5 | Cross-reference indexing | Link artifacts by embedding distance |

### 2.1.3 Key Design Decisions Needed

1. **Embedding model selection:** Which model to use for embeddings?
   - Options: OpenAI ada-002, local sentence-transformers, etc.
   - Constraint: Must be deterministic (same input → same embedding)

2. **Index format:** How to store embeddings?
   - Options: SQLite + vectors, FAISS, Qdrant, LanceDB
   - Constraint: Must be portable as cartridge artifacts (per Phase 6.0)

3. **Rebuild determinism:** How to ensure identical index on rebuild?
   - Canonical ordering of files
   - Stable hashing of content before embedding

---

## 3. Phase 5.2: Semiotic Compression Layer (SCL)

### 3.2.1 Problem Statement

From research document:
> "Frontier models burn tokens restating the same governance/procedure text repeatedly."

**Solution:** Replace repetitive governance prose with compact symbolic IR that expands deterministically.

### 3.2.2 Design Goals

1. **90%+ token reduction** for governance/procedure repetition
2. **Deterministic expansion** (same symbols → same output)
3. **Verifiable** (schema-valid outputs; hashes for artifacts)
4. **Human-auditable** (expand-to-text for review)
5. **Composable** (small primitives combine into complex intents)

### 3.2.3 Deliverables

| Artifact | Purpose |
|----------|---------|
| `SCL/CODEBOOK.json` | Symbol dictionary: symbol → meaning → expansion templates |
| `SCL/GRAMMAR.md` | Syntax rules (EBNF-ish) + examples |
| `SCL/decode.py` | Symbolic IR → expanded form (JSON JobSpec, audit text) |
| `SCL/validate.py` | Validates symbolic program + expanded output |
| `SCL/encode.py` | Natural text → Symbolic IR (optional, heuristic) |
| `scl` CLI | `decode`, `validate`, `run` commands |

### 3.2.4 Symbolic IR Examples

**ASCII-first notation (tokenizer-safe):**
```
@LAW>=0.1.0 & !WRITE(authored_md)
JOB{scan:DOMAIN_WORKTREE, validate:JOBSPEC, ledger:append}
CALL.cas.put(file=PATH)
```

**Macro categories (30-80 total):**
- Immutability constraints
- Allowed domains/roots
- Schema validate
- Ledger append
- CAS put/get
- Root scan/diff
- Expand-by-hash read requests

### 3.2.5 Execution Pipeline

```
1. Big model outputs Symbolic IR program (short)
         ↓
2. Decoder expands into:
   - JSON JobSpec(s)
   - Tool-call plan
   - Natural-language audit rendering
         ↓
3. Validator checks:
   - Symbolic syntax OK
   - Expanded JSON passes schema
   - Outputs in allowed roots
         ↓
4. If fail: error vector → prompt repair
```

---

## 4. MemoryRecord Contract (from V4 research)

This is the canonical data structure for Phase 5.2 and Phase 6.0:

### 4.1 Minimum Fields

```json
{
  "id": "stable content hash",
  "text": "byte-identical payload or CAS ref",
  "embeddings": {
    "model_name": [0.1, 0.2, ...]
  },
  "payload": {
    "tags": [],
    "timestamps": {},
    "roles": [],
    "doc_ids": []
  },
  "scores": {
    "elo": 1000,
    "recency": 0.9,
    "trust": 1.0,
    "decay": 0.95
  },
  "lineage": {
    "derived_from": [],
    "summarization_chain": []
  },
  "receipts": {
    "provenance_hashes": [],
    "tool_version_refs": []
  }
}
```

### 4.2 Contract Rules

- **Text is canonical** - source of truth
- **Vectors are derived** - rebuildable from text
- **All exports are receipted and hashed** - audit trail

---

## 5. VectorPack Format (Transfer)

Portable package for vector memory sharing:

```
vectorpack/
├── manifest.yaml          # schema version, models, dims, metrics, hashes
├── tables/                # canonical cartridge or portable tables
├── blobs/                 # content-addressed payloads
├── receipts/              # build receipts, proofs, schema hashes
└── indexes/               # optional derived indexes (rebuildable)
```

### 5.1 Determinism Rules

- Canonical ordering in manifests
- Stable hashing of file lists and record ordering
- Export/import emits receipts and is reproducible

### 5.2 Micro-pack Export (Token-friendly)

Purpose: Task-scoped top-K memories, not the entire substrate.

Format: JSONL with packed vectors:
- int8 quantized + scale, or float16 packed and base64 encoded
- Strict schema versioning
- Deterministic selection and tie-breaking

---

## 6. Dependencies & Integration

### 6.1 Phase 5.2 → Phase 6.0 (Cassette Network)

From V4 research:
> "The cassette network inherits the MemoryRecord contract from Phase 5.2"

**Implication:** Phase 5.2 must define the MemoryRecord contract before Phase 6.0 can bind cassette storage to it.

### 6.2 Phase 5.2 → Catalytic Continuity

From CAT_CHAT Catalytic Continuity report:
- MemoryRecord is foundation for session capsules
- Vectors must be "domesticated" for catalytic determinism
- Bounded top-k with stable tie-breaking
- Receipted retrieval with corpus snapshot IDs

### 6.3 Phase 5.2 → Phase 7 (Vector ELO)

The `scores.elo` field in MemoryRecord connects to Phase 7's ELO scoring system:
- ELO modulates vector ranking
- High ELO → include in working set
- Low ELO → pointer only

---

## 7. Implementation Pattern (from Phase 4)

Based on Phase 4's established patterns:

### 7.1 Expected File Structure

```
CAPABILITY/
├── PRIMITIVES/
│   ├── vector_index.py      # 5.1: Embedding storage + retrieval
│   ├── memory_record.py     # 5.2: MemoryRecord contract
│   ├── scl_codebook.py      # 5.2: Symbol dictionary
│   └── scl_decoder.py       # 5.2: Symbolic IR expansion
├── TESTBENCH/integration/
│   ├── test_phase_5_1_vector_embedding.py    # ~15-20 tests
│   └── test_phase_5_2_semiotic_compression.py # ~20-25 tests
└── TOOLS/
    └── scl/
        └── scl_cli.py       # CLI: decode, validate, run
```

### 7.2 Expected Documentation

```
LAW/CANON/
├── SEMANTIC/
│   ├── SCL_SPECIFICATION.md
│   └── SYMBOL_GRAMMAR.md
└── VECTOR/
    └── VECTOR_INDEX_SPEC.md
```

### 7.3 Receipt Pattern

All operations emit receipts:
- `MANIFEST.json` - File inventory with hashes
- `REPORT.json` - Execution details
- `PROOF.json` - Verification proofs

---

## 8. Key Research Documents

| Document | Location | Content |
|----------|----------|---------|
| Vector Substrate Contract | `INBOX/reports/V4/01-06-2026-21-13_5_2_VECTOR_SUBSTRATE_VECTORPACK.md` | MemoryRecord, VectorPack format |
| SCL Specification | `INBOX/2025-12/Week-01/12-29-2025-07-01_SEMIOTIC_COMPRESSION.md` | Full SCL design spec |
| Symbolic Compression Research | `INBOX/2025-12/Week-52/12-26-2025-06-39_SYMBOLIC_COMPRESSION.md` | Initial research |
| Catalytic Continuity | `INBOX/reports/V4/01-06-2026-21-13_CAT_CHAT_CATALYTIC_CONTINUITY.md` | Integration design |
| Cassette Substrate | `INBOX/reports/V4/01-06-2026-21-13_6_0_CANONICAL_CASSETTE_SUBSTRATE.md` | Phase 6.0 dependency |

---

## 9. Recommended Next Steps

1. **Finalize MemoryRecord contract schema** - JSON schema with validation
2. **Select embedding model** - Decision record needed (ADR)
3. **Implement SCL CODEBOOK** - Start with 30 core governance macros
4. **Build decoder first** - Deterministic expansion unlocks the loop
5. **Write Phase 5 tests** - Fixture-backed, determinism-focused

---

## 10. Exit Criteria Summary

### Phase 5.1
- [ ] Vector index includes canon + ADRs with deterministic rebuild
- [ ] Skill discovery returns stable results for fixed corpus
- [ ] Cross-reference indexing operational

### Phase 5.2
- [ ] `scl decode <program>` → emits JobSpec JSON
- [ ] Meaningful token reduction demonstrated vs baseline
- [ ] Reproducible expansions (same symbols → same output hash)
- [ ] MemoryRecord contract defined and validated

---

*Report generated from consolidated research across 50+ files in the codebase.*
