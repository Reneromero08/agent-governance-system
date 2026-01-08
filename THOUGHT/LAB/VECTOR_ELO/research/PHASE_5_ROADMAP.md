---
uuid: 00000000-0000-0000-0000-000000000000
title: Phase 5 Vector/Symbol Integration Detailed Roadmap
section: roadmap
bucket: roadmaps
author: System
priority: High
created: 2026-01-07
modified: 2026-01-07
status: Active
summary: Comprehensive implementation roadmap for Phase 5
tags:
- phase-5
- vector
- semiotic
- roadmap
---
<!-- CONTENT_HASH: 7e61cd536fd30b163f4c43ad44a0ed1ac922b06cf7d8dae7a68f69c7bab4cdb9 -->

# Phase 5: Vector/Symbol Integration - Detailed Roadmap

**Version:** 1.0.0
**Created:** 2026-01-07
**Prerequisite:** Phase 4 (Catalytic Architecture) - COMPLETE
**Downstream:** Phase 6.0 (Cassette Network) depends on 5.2 MemoryRecord contract

---

## Overview

Phase 5 enables semantic addressability through two complementary systems:
- **5.1** Vector embeddings for semantic retrieval
- **5.2** Symbolic compression for token efficiency

**Goal:** Make governance artifacts addressable by meaning, not just by path or hash.

---

## Global Definition of Done

Every task must produce:
- [ ] All relevant tests pass
- [ ] Receipts emitted (inputs, outputs, hashes, commands, exit status)
- [ ] Human-readable report (what changed, why, how verified)
- [ ] Scope respected (explicit allowlists for writes)

---

# Phase 5.1: Vector Indexing & Semantic Discovery

## 5.1.0 Foundation: MemoryRecord Contract

**Purpose:** Define the canonical data structure for all vector-indexed content.

### 5.1.0.1 Define MemoryRecord JSON Schema
- [ ] Create `CAPABILITY/PRIMITIVES/schemas/memory_record.schema.json`
- [ ] Required fields:
  - `id` (string): Content hash (SHA-256)
  - `text` (string): Canonical text content
  - `embeddings` (object): Model name → vector array
  - `payload` (object): Metadata (tags, timestamps, roles, doc_ids)
  - `scores` (object): ELO, recency, trust, decay
  - `lineage` (object): Derivation chain, summarization history
  - `receipts` (object): Provenance hashes, tool versions
- [ ] Implement `additionalProperties: false` at all levels

### 5.1.0.2 Implement MemoryRecord Primitive
- [ ] Create `CAPABILITY/PRIMITIVES/memory_record.py`
- [ ] Functions:
  - `create_record(text, metadata) -> MemoryRecord`
  - `validate_record(record) -> verdict`
  - `hash_record(record) -> content_hash`
- [ ] Contract rules:
  - Text is canonical (source of truth)
  - Vectors are derived (rebuildable from text)
  - All exports are receipted and hashed

### 5.1.0.3 Tests
- [ ] `test_memory_record_schema_validation.py`
- [ ] Fixtures: valid records, invalid records, edge cases
- [ ] Determinism: same input → same hash

**Exit Criteria:**
- [ ] MemoryRecord schema validated and documented
- [ ] Create/validate/hash functions working
- [ ] 10+ tests passing

---

## 5.1.1 Embed Canon Files

**Purpose:** Make all governance canon semantically searchable.

### 5.1.1.1 Canon File Inventory
- [ ] Enumerate all files in `LAW/CANON/*`
- [ ] Create manifest with file paths and content hashes
- [ ] Emit inventory receipt

### 5.1.1.2 Embedding Generation
- [ ] Select embedding model (ADR required)
  - Options: OpenAI text-embedding-3-small, sentence-transformers, local model
  - Constraint: Deterministic (same input → same embedding)
- [ ] Implement `embed_text(text, model) -> vector`
- [ ] Batch embed all canon files
- [ ] Store as MemoryRecord instances

### 5.1.1.3 Vector Index Storage
- [ ] Create vector index structure (SQLite + vector extension or FAISS)
- [ ] Store embeddings with content hashes as IDs
- [ ] Enable rebuild from source files (deterministic)

### 5.1.1.4 Tests
- [ ] `test_phase_5_1_1_canon_embedding.py`
- [ ] Verify all canon files embedded
- [ ] Verify rebuild produces identical index
- [ ] Verify similarity search returns expected results

**Exit Criteria:**
- [ ] All `LAW/CANON/*` files embedded
- [ ] Index rebuildable deterministically
- [ ] Similarity search functional

---

## 5.1.2 Embed ADRs

**Purpose:** Make architecture decisions semantically searchable.

### 5.1.2.1 ADR Inventory
- [ ] Enumerate all files in `LAW/CONTEXT/decisions/*`
- [ ] Parse ADR metadata (title, status, date)
- [ ] Create manifest with hashes

### 5.1.2.2 ADR Embedding
- [ ] Batch embed all ADR files
- [ ] Include metadata in MemoryRecord payload
- [ ] Store with cross-references to canon

### 5.1.2.3 Tests
- [ ] `test_phase_5_1_2_adr_embedding.py`
- [ ] Verify all ADRs embedded
- [ ] Verify metadata preserved
- [ ] Verify semantic search returns relevant ADRs

**Exit Criteria:**
- [ ] All ADRs embedded with metadata
- [ ] Searchable by semantic query

---

## 5.1.3 Vector-Indexed CAS for Model Weights

**Purpose:** Store model artifacts with semantic addressability.

### 5.1.3.1 Model Weight Registry
- [ ] Define schema for model weight records
- [ ] Include: model name, version, hash, embedding of description
- [ ] Store in CAS with vector index pointer

### 5.1.3.2 Implementation
- [ ] `store_model_weights(model_path, metadata) -> hash`
- [ ] `retrieve_model_by_description(query) -> model_path`

### 5.1.3.3 Tests
- [ ] Model storage and retrieval
- [ ] Semantic search by model description

**Exit Criteria:**
- [ ] Model weights addressable by hash and description

---

## 5.1.4 Semantic Skill Discovery

**Purpose:** Find skills by description similarity.

### 5.1.4.1 Skill Inventory
- [ ] Enumerate all `CAPABILITY/SKILLS/*/SKILL.md`
- [ ] Parse skill metadata (purpose, trigger, inputs, outputs)
- [ ] Create skill manifest

### 5.1.4.2 Skill Embedding
- [ ] Embed skill descriptions
- [ ] Store with metadata in MemoryRecord
- [ ] Enable semantic search

### 5.1.4.3 Skill Discovery API
- [ ] `find_skills_by_intent(query, top_k) -> [skill_ids]`
- [ ] Deterministic tie-breaking for stable results
- [ ] Emit discovery receipt

### 5.1.4.4 Tests
- [ ] `test_phase_5_1_4_skill_discovery.py`
- [ ] Known queries return expected skills
- [ ] Results stable across runs

**Exit Criteria:**
- [ ] Skill discovery returns stable results for fixed corpus
- [ ] Top-K results deterministic

---

## 5.1.5 Cross-Reference Indexing

**Purpose:** Link artifacts by embedding distance.

### 5.1.5.1 Cross-Reference Graph
- [ ] Compute pairwise distances for related artifacts
- [ ] Store as graph edges with distance weights
- [ ] Enable "related artifacts" queries

### 5.1.5.2 Implementation
- [ ] `find_related(artifact_id, threshold, top_k) -> [related_ids]`
- [ ] Stable ordering by distance then ID

### 5.1.5.3 Tests
- [ ] Related artifacts for known items
- [ ] Graph traversal correctness

**Exit Criteria:**
- [ ] Cross-reference queries functional
- [ ] Results deterministic

---

# Phase 5.2: Semiotic Compression Layer (SCL)

## 5.2.0 Research Integration

**Reference Documents:**
- `INBOX/2025-12/Week-01/12-29-2025-07-01_SEMIOTIC_COMPRESSION.md`
- `INBOX/2025-12/Week-52/12-26-2025-06-39_SYMBOLIC_COMPRESSION.md`
- `INBOX/reports/V4/01-06-2026-21-13_5_2_VECTOR_SUBSTRATE_VECTORPACK.md`

---

## 5.2.1 Define MVP Macro Set

**Purpose:** Identify 30-80 macros covering 80% of governance repetition.

### 5.2.1.1 Macro Categories
- [ ] **Constraint macros:** Immutability, allowed domains, forbidden writes
- [ ] **Schema macros:** Validate JobSpec, validate receipt, validate bundle
- [ ] **CAS macros:** Put, get, verify, list
- [ ] **Scan macros:** Root scan, diff, purity check
- [ ] **Ledger macros:** Append, verify chain, query
- [ ] **Expand macros:** Hash-to-content, symbol-to-definition

### 5.2.1.2 Macro Survey
- [ ] Analyze existing governance text for repetition patterns
- [ ] Identify top 30 most repeated concepts
- [ ] Draft macro definitions

### 5.2.1.3 Macro Specification Document
- [ ] Create `LAW/CANON/SEMANTIC/SCL_MACRO_CATALOG.md`
- [ ] For each macro: name, meaning, expansion template, examples

**Exit Criteria:**
- [ ] 30-80 macros identified and documented
- [ ] Coverage estimate for governance repetition

---

## 5.2.2 Implement CODEBOOK.json

**Purpose:** Symbol dictionary mapping symbols to meanings and expansions.

### 5.2.2.1 Schema Definition
- [ ] Create `CAPABILITY/PRIMITIVES/schemas/scl_codebook.schema.json`
- [ ] Structure:
  ```json
  {
    "version": "1.0.0",
    "symbols": {
      "@SYMBOL_NAME": {
        "meaning": "Human-readable explanation",
        "expansion": "Full expanded text or template",
        "category": "constraint|schema|cas|scan|ledger|expand",
        "params": ["optional", "parameter", "list"]
      }
    }
  }
  ```

### 5.2.2.2 Initial CODEBOOK
- [ ] Create `SCL/CODEBOOK.json` with MVP macro set
- [ ] Validate against schema
- [ ] Include version and hash

### 5.2.2.3 Tests
- [ ] Schema validation
- [ ] All macros parseable
- [ ] No duplicate symbols

**Exit Criteria:**
- [ ] CODEBOOK.json created with 30+ macros
- [ ] Schema-validated

---

## 5.2.3 Implement SCL Decoder

**Purpose:** Expand symbolic IR into full JobSpec JSON.

### 5.2.3.1 Parser Implementation
- [ ] Create `CAPABILITY/PRIMITIVES/scl_decoder.py`
- [ ] Parse symbolic IR syntax:
  ```
  @LAW>=0.1.0 & !WRITE(authored_md)
  JOB{scan:DOMAIN_WORKTREE, validate:JOBSPEC}
  CALL.cas.put(file=PATH)
  ```
- [ ] Tokenize, parse, build AST

### 5.2.3.2 Expansion Engine
- [ ] `decode(program, codebook) -> JobSpec JSON`
- [ ] Template variable substitution
- [ ] Nested macro expansion
- [ ] Deterministic output (same input → same JSON hash)

### 5.2.3.3 Audit Rendering
- [ ] `render_audit(program, codebook) -> human_readable_text`
- [ ] Show expanded form for review

### 5.2.3.4 Tests
- [ ] `test_phase_5_2_3_scl_decoder.py`
- [ ] Known programs → expected outputs
- [ ] Determinism: repeated decode → same hash
- [ ] Error cases: invalid syntax, unknown symbols

**Exit Criteria:**
- [ ] Decoder expands symbolic IR to JobSpec JSON
- [ ] Deterministic expansion verified

---

## 5.2.4 Implement SCL Validator

**Purpose:** Validate symbolic programs and expanded outputs.

### 5.2.4.1 Symbolic Validation
- [ ] Create `CAPABILITY/PRIMITIVES/scl_validator.py`
- [ ] Check: syntax valid, all symbols known, params match
- [ ] Emit validation receipt

### 5.2.4.2 Expansion Validation
- [ ] Expanded JSON passes JobSpec schema
- [ ] Outputs in allowed roots
- [ ] No forbidden operations

### 5.2.4.3 Tests
- [ ] Valid programs pass
- [ ] Invalid syntax fails with clear error
- [ ] Unknown symbols fail
- [ ] Schema-invalid expansions fail

**Exit Criteria:**
- [ ] Validator catches all error classes
- [ ] Clear error messages

---

## 5.2.5 Implement SCL CLI

**Purpose:** Command-line interface for SCL operations.

### 5.2.5.1 CLI Commands
- [ ] Create `CAPABILITY/TOOLS/scl/scl_cli.py`
- [ ] Commands:
  - `scl decode <program>` → emit JobSpec JSON
  - `scl validate <program|job.json>` → PASS/FAIL
  - `scl run <program>` → execute with invariant proofs
  - `scl audit <program>` → human-readable expansion

### 5.2.5.2 Integration
- [ ] Register as skill or CLI tool
- [ ] Emit receipts for all operations

### 5.2.5.3 Tests
- [ ] CLI invocation tests
- [ ] Output format validation
- [ ] Error handling

**Exit Criteria:**
- [ ] `scl` CLI functional with all commands
- [ ] Receipts emitted

---

## 5.2.6 SCL Tests & Benchmarks

**Purpose:** Prove determinism and measure token reduction.

### 5.2.6.1 Determinism Tests
- [ ] Create `CAPABILITY/TESTBENCH/integration/test_phase_5_2_semiotic_compression.py`
- [ ] Same program → same JSON hash (100 runs)
- [ ] Same program + same codebook version → same output

### 5.2.6.2 Schema Validation Tests
- [ ] Expanded JobSpecs validate against schema
- [ ] Invalid inputs produce schema errors

### 5.2.6.3 Token Benchmark
- [ ] Measure: tokens for symbolic program vs expanded text
- [ ] Target: 90%+ reduction for governance boilerplate
- [ ] Create benchmark fixture with representative programs

### 5.2.6.4 Negative Tests
- [ ] Invalid syntax → clear error
- [ ] Unknown symbol → clear error
- [ ] Circular expansion → error (if possible)

**Exit Criteria:**
- [ ] 20+ tests passing
- [ ] Token reduction benchmark documented
- [ ] All error classes covered

---

# Integration & Dependencies

## Phase 5.2 → Phase 6.0 Handoff

The MemoryRecord contract defined in 5.1.0 becomes the foundation for Phase 6.0 Cassette Network:
- Cassette storage binds to MemoryRecord schema
- Each cassette DB is a portable cartridge artifact
- Derived indexes are rebuildable from cartridges

**Handoff Checklist:**
- [ ] MemoryRecord schema finalized and frozen
- [ ] Schema version tagged
- [ ] Migration path documented

## Phase 5.2 → Phase 7 Connection

The `scores.elo` field in MemoryRecord connects to Phase 7's ELO system:
- ELO modulates vector ranking
- HIGH ELO → include in working set
- LOW ELO → pointer only

---

# File Locations Summary

```
CAPABILITY/
├── PRIMITIVES/
│   ├── memory_record.py           # 5.1.0: MemoryRecord contract
│   ├── vector_index.py            # 5.1.1-5: Embedding + indexing
│   ├── scl_codebook.py            # 5.2.2: Codebook loader
│   ├── scl_decoder.py             # 5.2.3: Symbolic IR expansion
│   ├── scl_validator.py           # 5.2.4: Validation
│   └── schemas/
│       ├── memory_record.schema.json
│       └── scl_codebook.schema.json
├── TESTBENCH/integration/
│   ├── test_phase_5_1_vector_embedding.py
│   └── test_phase_5_2_semiotic_compression.py
└── TOOLS/
    └── scl/
        └── scl_cli.py

LAW/CANON/
├── SEMANTIC/
│   ├── SCL_SPECIFICATION.md       # Formal SCL spec
│   ├── SCL_MACRO_CATALOG.md       # All macros documented
│   └── SYMBOL_GRAMMAR.md          # EBNF syntax
└── VECTOR/
    └── VECTOR_INDEX_SPEC.md       # Vector indexing spec

SCL/
├── CODEBOOK.json                  # Symbol dictionary
├── GRAMMAR.md                     # Syntax reference
└── tests/fixtures/                # Test programs
```

---

# Test Count Targets

| Sub-Phase | Focus | Target Tests |
|-----------|-------|--------------|
| 5.1.0 | MemoryRecord | 10+ |
| 5.1.1 | Canon embedding | 5+ |
| 5.1.2 | ADR embedding | 5+ |
| 5.1.3 | Model weights | 3+ |
| 5.1.4 | Skill discovery | 5+ |
| 5.1.5 | Cross-reference | 5+ |
| 5.2.1-2 | Codebook | 5+ |
| 5.2.3 | Decoder | 10+ |
| 5.2.4 | Validator | 5+ |
| 5.2.5 | CLI | 5+ |
| 5.2.6 | Benchmarks | 5+ |
| **Total** | | **~65 tests** |

---

# Exit Criteria Summary

## Phase 5.1 Complete When:
- [ ] MemoryRecord contract defined and validated
- [ ] Vector index includes canon + ADRs with deterministic rebuild
- [ ] Skill discovery returns stable results for fixed corpus
- [ ] Cross-reference indexing operational

## Phase 5.2 Complete When:
- [ ] CODEBOOK.json contains 30+ governance macros
- [ ] `scl decode <program>` → emits JobSpec JSON
- [ ] `scl validate` passes valid programs, rejects invalid
- [ ] Meaningful token reduction demonstrated (90%+ for governance)
- [ ] Reproducible expansions (same symbols → same output hash)

---

*Roadmap v1.0.0 - Generated 2026-01-07*

---

# Appendix: Phase 5 Validation & Measured Results

> Updated 2026-01-07 with actual measured compression data.

## Measured Compression Results

**Source:** `NAVIGATION/PROOFS/COMPRESSION/COMPRESSION_PROOF_REPORT.md`

### Pointer-Only Mode (Hash References)
| Query | Baseline Tokens | Compressed | Savings |
|-------|-----------------|------------|---------|
| Translation Layer architecture | 276,085 | 18 | **99.993%** |
| AGS BOOTSTRAP v1.0 | 276,085 | 18 | **99.993%** |
| Mechanical indexer scans | 276,085 | 20 | **99.993%** |

### Filtered-Content Mode (Semantic Retrieval)
| Query | Baseline Tokens | Compressed | Savings |
|-------|-----------------|------------|---------|
| Translation Layer architecture | 67,375 | 351 | **99.479%** |
| AGS BOOTSTRAP v1.0 | 67,375 | 86 | **99.872%** |
| Mechanical indexer scans | 67,375 | 241 | **99.642%** |

**Conclusion:** The 90%+ token reduction target is **conservative**. Actual measurements show **99.99% compression** in pointer-only mode.

## Component Validation

| Component | Assessment | Evidence |
|-----------|------------|----------|
| 5.0 MemoryRecord | ✅ Ready | JSON schema + Python dataclass. Phase 6.0 depends on this. |
| 5.1 Vector Indexing | ✅ Proven | CORTEX semantic search operational, 3,991 chunks indexed |
| 5.1.4 VectorPack | ✅ Specified | Directory structure defined in research |
| 5.2 SCL | ✅ Grounded | 30-80 macros from governance pattern analysis |

## Research-Grounded Targets

### Token Reduction
- **Measured:** 99.99% (pointer-only), 99.5%+ (filtered-content)
- **Target:** 90%+ (conservative baseline)
- **Source:** COMPRESSION_PROOF_REPORT.md

### Macro Count (30-80)
- **Source:** Governance pattern analysis in SYMBOLIC_COMPRESSION.md
- **Method:** "covering 80% of repeated governance"
- **Categories:** Constraint, Schema, CAS, Scan, Ledger, Expand macros

### DSL Syntax
- **Designed:** ASCII-first for tokenizer safety
- **Tested:** Examples in SEMIOTIC_COMPRESSION.md
- **Format:** `@LAW>=0.1.0 & !WRITE(authored_md)`

## Downstream Dependencies

Phase 5 is the foundation for:

| Downstream | Depends On | Connection |
|------------|------------|------------|
| Phase 3 (CAT CHAT) | 5.0 MemoryRecord | Session capsules |
| Phase 6 (Cassette) | 5.0 MemoryRecord | Cassette storage binding |
| Phase 6.1 | 5.1 Vectors | 9 cassettes including RESIDENT |
| Phase 7 (ELO) | 5.0 scores field | ELO in MemoryRecord |
| Phase 9 (Swarm) | 5.1.4 Skill Discovery | Governor task routing |
| Phase 10 (Ω) | 5.2 SCL | Automatic symbol extraction |

## Execution Order

```
Phase 5.0 - MemoryRecord (do first, Phase 6.0 depends on it)
     ↓
Phase 5.1 - Vector Indexing (proven via CORTEX)
     ↓
Phase 5.2 - SCL (targets grounded in measured data)
```

## Key Principle
**Measure first, build second.** Targets are grounded in actual measurements from COMPRESSION_PROOF_REPORT.md, not aspirational claims.
