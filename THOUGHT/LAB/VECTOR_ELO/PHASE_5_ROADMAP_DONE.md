---
title: Phase 5 Vector/Symbol Integration Finished Roadmap
section: roadmap
version: 1.8.0
created: 2026-01-07
modified: 2026-01-08
status: Archived
summary: Archived completed Phase 5 tasks (5.1 Vector Indexing, 5.2.1-5.2.4 SCL foundation)
tags:
- phase-5
- vector
- semiotic
- roadmap
- finished
---
<!-- CONTENT_HASH: c4f02d44195b52aa752612a968112eaf32f654a8596f90bb596037eb5c167025 -->

# Phase 5: Vector/Symbol Integration - Detailed Roadmap

## BREAKTHROUGH: Semantic Symbol Compression (2026-01-08)

**Discovery:** Single-token CJK symbols achieve **56,370x compression** when receiver has referenced content in shared context.

| Symbol | Tokens | Expands To | Ratio | Status |
|--------|--------|------------|-------|--------|
| 法 | 1 | All canon (56,370 tokens) | **56,370x** | PROVEN |
| 真 | 1 | Semiotic Foundation (1,455 tokens) | **1,455x** | PROVEN |
| 道 | 1 | Context-activated (4 meanings) | **24x** | PROVEN |

**Proof:** `NAVIGATION/PROOFS/COMPRESSION/SEMANTIC_SYMBOL_PROOF_REPORT.md`

**Implementation:** `CAPABILITY/TOOLS/codebook_lookup.py` (MCP tool: `codebook_lookup`)

**Stacking Insight:**
- L1 only (法): 56,370x but dumps everything (low alignment)
- L1+L2 (法.query): 843x but returns only relevant chunks (high alignment)
- Formula: `density = shared_context ^ alignment`

**Impact on Roadmap:** Phase 5.2.3 (SCL Decoder) simplified. Stack codebook_lookup + CORTEX FTS instead of building complex AST decoder.

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

# Phase 5.1: Vector Indexing & Semantic Discovery ✅

## 5.1.0 Foundation: MemoryRecord Contract ✅

**Purpose:** Define the canonical data structure for all vector-indexed content.

### 5.1.0.1 Define MemoryRecord JSON Schema
- [x] Create `LAW/SCHEMAS/memory_record.schema.json` (**DONE** 2026-01-08)
- [x] Required fields:
  - `id` (string): Content hash (SHA-256)
  - `text` (string): Canonical text content
  - `embeddings` (object): Model name → vector array
  - `payload` (object): Metadata (tags, timestamps, roles, doc_ids)
  - `scores` (object): ELO, recency, trust, decay
  - `lineage` (object): Derivation chain, summarization history
  - `receipts` (object): Provenance hashes, tool versions
- [x] Implement `additionalProperties: false` at all levels

### 5.1.0.2 Implement MemoryRecord Primitive
- [x] Create `CAPABILITY/PRIMITIVES/memory_record.py` (**DONE** 2026-01-08)
- [x] Functions:
  - `create_record(text, metadata) -> MemoryRecord`
  - `validate_record(record) -> verdict`
  - `hash_record(record) -> content_hash`
  - `add_embedding(record, name, vector, model) -> record`
  - `to_json(record) -> str` / `from_json(str) -> record`
  - `canonical_bytes(record) -> bytes`
  - `full_hash(record) -> str`
- [x] Contract rules:
  - Text is canonical (source of truth)
  - Vectors are derived (rebuildable from text)
  - All exports are receipted and hashed

### 5.1.0.3 Tests
- [x] `CAPABILITY/TESTBENCH/core/test_memory_record.py` (**DONE** 2026-01-08)
- [x] Fixtures: valid records, invalid records, edge cases
- [x] Determinism: same input → same hash
- [x] **23 tests passing**

**Exit Criteria:**
- [x] MemoryRecord schema validated and documented
- [x] Create/validate/hash functions working
- [x] 23 tests passing (exceeds 10+ requirement)

---

## 5.1.1 Embed Canon Files ✅ COMPLETE (2026-01-08)

**Purpose:** Make all governance canon semantically searchable.

### 5.1.1.1 Canon File Inventory
- [x] Enumerate all files in `LAW/CANON/*` (**DONE** - 32 files)
- [x] Create manifest with file paths and content hashes (**DONE**)
- [x] Emit inventory receipt (**DONE**)

### 5.1.1.2 Embedding Generation
- [x] Select embedding model: `all-MiniLM-L6-v2` (ADR-030)
  - Uses sentence-transformers (local, deterministic)
  - 384 dimensions, float32
- [x] Implement `embed_text(text, model) -> vector` (**DONE** via EmbeddingEngine)
- [x] Batch embed all canon files (**DONE** - 32 files embedded)
- [x] Store as MemoryRecord instances (**DONE**)

### 5.1.1.3 Vector Index Storage
- [x] Create vector index structure: SQLite-based `canon_index.db` (**DONE**)
- [x] Store embeddings with content hashes as IDs (**DONE**)
- [x] Enable rebuild from source files (deterministic) (**DONE**)

### 5.1.1.4 Tests
- [x] `test_phase_5_1_1_canon_embedding.py` (**DONE** - 23 tests)
- [x] Verify all canon files embedded (**DONE**)
- [x] Verify rebuild produces identical index (**DONE**)
- [x] Verify similarity search returns expected results (**DONE**)

**Exit Criteria:**
- [x] All `LAW/CANON/*` files embedded (32 files)
- [x] Index rebuildable deterministically
- [x] Similarity search functional

**Implementation:**
- `CAPABILITY/PRIMITIVES/canon_index.py` - Core indexing primitive
- `CAPABILITY/TESTBENCH/integration/test_phase_5_1_1_canon_embedding.py` - 23 tests
- `NAVIGATION/CORTEX/db/canon_index.db` - Embedded index

---

## 5.1.2 Embed ADRs ✅ COMPLETE (2026-01-08)

**Purpose:** Make architecture decisions semantically searchable.

### 5.1.2.1 ADR Inventory
- [x] Enumerate all files in `LAW/CONTEXT/decisions/*` (**DONE** - 37 ADRs)
- [x] Parse ADR metadata (title, status, date) (**DONE** - YAML frontmatter)
- [x] Create manifest with hashes (**DONE**)

### 5.1.2.2 ADR Embedding
- [x] Batch embed all ADR files (**DONE** - 37 files embedded)
- [x] Include metadata in MemoryRecord payload (**DONE** - id, title, status, date, etc.)
- [x] Store with cross-references to canon (**DONE** - 184 cross-references)

### 5.1.2.3 Tests
- [x] `test_phase_5_1_2_adr_embedding.py` (**DONE** - 28 tests passing)
- [x] Verify all ADRs embedded (**DONE**)
- [x] Verify metadata preserved (**DONE**)
- [x] Verify semantic search returns relevant ADRs (**DONE**)

**Exit Criteria:**
- [x] All ADRs embedded with metadata (37 ADRs, 31 Accepted, 4 Proposed, 2 Unknown)
- [x] Searchable by semantic query

**Implementation:**
- `CAPABILITY/PRIMITIVES/adr_index.py` - ADR indexing primitive
- `CAPABILITY/TESTBENCH/integration/test_phase_5_1_2_adr_embedding.py` - 28 tests
- `NAVIGATION/CORTEX/db/adr_index.db` - Embedded index with cross-references

---

## 5.1.3 Vector-Indexed CAS for Model Weights ✅ COMPLETE (2026-01-08)

**Purpose:** Store model artifacts with semantic addressability.

### 5.1.3.1 Model Weight Registry
- [x] Define schema for model weight records (**DONE** - ModelRecord TypedDict)
- [x] Include: model name, version, hash, embedding of description (**DONE**)
- [x] Store in CAS with vector index pointer (**DONE** - weights_hash field)

**ModelRecord Schema:**
```python
ModelRecord = {
    "id": str,           # SHA-256(name@version) - deterministic
    "name": str,         # Model name (e.g., "all-MiniLM-L6-v2")
    "version": str,      # Version string
    "description": str,  # For semantic search (embeddable)
    "format": str,       # pytorch, safetensors, onnx, etc.
    "weights_hash": str, # SHA-256 of weights (CAS reference)
    "size_bytes": int,   # Model size
    "embedding": bytes,  # Description embedding (384 dims)
    "metadata": dict,    # architecture, params, license, etc.
    "source": str,       # huggingface, local, custom
    "created_at": str,
    "updated_at": str,
}
```

### 5.1.3.2 Implementation
- [x] `register_model(name, version, description, format, ...) -> result` (**DONE**)
- [x] `get_model(name, version) -> ModelRecord` (**DONE**)
- [x] `get_model_by_weights_hash(hash) -> ModelRecord` (**DONE** - deduplication)
- [x] `search_models(query, top_k) -> [results]` (**DONE** - semantic search)
- [x] `list_models(format_filter) -> [models]` (**DONE**)
- [x] `get_registry_stats() -> stats` (**DONE**)
- [x] `verify_registry() -> verification_result` (**DONE**)

### 5.1.3.3 Tests
- [x] `test_phase_5_1_3_model_registry.py` (**DONE** - 28 tests passing)
- [x] Model ID determinism (name@version -> hash) (**DONE**)
- [x] Model registration and retrieval (**DONE**)
- [x] Semantic search by model description (**DONE**)
- [x] Deduplication by weights hash (**DONE**)
- [x] Registry statistics and verification (**DONE**)

**Exit Criteria:**
- [x] Model weights addressable by hash and description
- [x] Semantic search functional
- [x] Registry integrity verification working

**Implementation:**
- `CAPABILITY/PRIMITIVES/model_registry.py` - Core registry primitive
- `CAPABILITY/TESTBENCH/integration/test_phase_5_1_3_model_registry.py` - 28 tests
- `NAVIGATION/CORTEX/db/model_registry.db` - Registry database (created on first use)

---

## 5.1.4 Semantic Skill Discovery ✅ COMPLETE (2026-01-08)

**Purpose:** Find skills by description similarity.

### 5.1.4.1 Skill Inventory
- [x] Enumerate all `CAPABILITY/SKILLS/*/SKILL.md` (**DONE** - 24 skills)
- [x] Parse skill metadata (purpose, trigger, inputs, outputs) (**DONE** - YAML frontmatter + markdown)
- [x] Create skill manifest (**DONE** - with content hashes)

### 5.1.4.2 Skill Embedding
- [x] Embed skill descriptions (**DONE** - all-MiniLM-L6-v2, 384 dims)
- [x] Store with metadata in MemoryRecord-compatible format (**DONE**)
- [x] Enable semantic search (**DONE** - cosine similarity)

### 5.1.4.3 Skill Discovery API
- [x] `find_skills_by_intent(query, top_k) -> [skill_ids]` (**DONE**)
- [x] Deterministic tie-breaking for stable results (**DONE** - score desc, skill_id asc)
- [x] Emit discovery receipt (**DONE**)

### 5.1.4.4 Tests
- [x] `test_phase_5_1_4_skill_discovery.py` (**DONE** - 32 tests passing)
- [x] Known queries return expected skills (**DONE** - verified)
- [x] Results stable across runs (**DONE** - determinism tests passing)

### 5.1.4.5 MCP Integration
- [x] Added `skill_discovery` MCP tool (**DONE**)
- [x] Tool schema in `schemas/tools.json` (**DONE**)
- [x] Server handler in `server.py` (**DONE**)

**Exit Criteria:**
- [x] Skill discovery returns stable results for fixed corpus
- [x] Top-K results deterministic

**Implementation:**
- `CAPABILITY/PRIMITIVES/skill_index.py` - Core primitive (inventory, embedding, search)
- `CAPABILITY/TESTBENCH/integration/test_phase_5_1_4_skill_discovery.py` - 32 tests
- `NAVIGATION/CORTEX/db/skill_index.db` - SQLite index with embeddings
- `CAPABILITY/MCP/server.py` - MCP tool handler
- `CAPABILITY/MCP/schemas/tools.json` - Tool schema

**Example Results:**
- Query: "verify canon changes and enforce changelog"
- Top Result: `governance/canon-governance-check` (similarity: 0.589)

---

## 5.1.5 Cross-Reference Indexing ✅ COMPLETE (2026-01-08)

**Purpose:** Link artifacts by embedding distance.

### 5.1.5.1 Cross-Reference Graph
- [x] Compute pairwise distances for related artifacts (**DONE**)
- [x] Store as graph edges with distance weights (**DONE** - SQLite with similarity scores)
- [x] Enable "related artifacts" queries (**DONE**)

### 5.1.5.2 Implementation
- [x] `find_related(artifact_id, threshold, top_k) -> [related_ids]` (**DONE**)
- [x] Stable ordering by distance then ID (**DONE** - similarity desc, artifact_id asc)

### 5.1.5.3 Tests
- [x] Related artifacts for known items (**DONE** - 20 tests passing)
- [x] Graph traversal correctness (**DONE** - deterministic ordering verified)

**Exit Criteria:**
- [x] Cross-reference queries functional
- [x] Results deterministic

**Implementation:**
- `CAPABILITY/PRIMITIVES/cross_ref_index.py` - Core primitive (build, query, stats)
- `CAPABILITY/TESTBENCH/integration/test_phase_5_1_5_1_cross_refs.py` - 20 tests
- `NAVIGATION/CORTEX/db/cross_ref_index.db` - SQLite graph database
- `CAPABILITY/MCP/server.py` - MCP tool handler
- `CAPABILITY/MCP/schemas/tools.json` - Tool schema

**Features:**
- Unified graph across canon, ADR, and skill artifacts
- Configurable similarity threshold and top_k per artifact
- Cross-type relationships (e.g., canon files linked to related ADRs)
- Batch similarity computation using numpy
- Build history tracking with receipts

**Example Usage:**
```python
# Build cross-reference graph
result = build_cross_refs(threshold=0.3, top_k_per_artifact=10)
# Returns: {artifacts_count: N, refs_count: M, duration_seconds: X}

# Find related artifacts
related = find_related("canon:LAW/CANON/GOVERNANCE/IMMUTABILITY.md", top_k=5)
# Returns: {artifact_id, related: [{artifact_id, artifact_type, artifact_path, similarity, metadata}], total_candidates}
```

---

# Phase 5.2: Semiotic Compression Layer (SCL)

## 5.2.0 Research Integration

**Reference Documents:**
- `INBOX/2025-12/Week-01/12-29-2025-07-01_SEMIOTIC_COMPRESSION.md`
- `INBOX/2025-12/Week-52/12-26-2025-06-39_SYMBOLIC_COMPRESSION.md`
- `INBOX/reports/V4/01-06-2026-21-13_5_2_VECTOR_SUBSTRATE_VECTORPACK.md`

---

## 5.2.1 Define MVP Macro Set ✅ COMPLETE (2026-01-08)

**Purpose:** Identify 30-80 macros covering 80% of governance repetition.

> **IMPLEMENTED (2026-01-08):** Compact macro grammar with radicals, operators, and rule lookups. 50 macros total achieving 60% token savings vs verbose @-prefix scheme.

### 5.2.1.1 Macro Categories
- [x] **Domain radicals:** C, I, V, L, G, S, R, A, J, P (10 radicals, all 1 token) (**DONE**)
- [x] **Operators:** *, !, ?, &, |, ., : (7 operators, all 1 token) (**DONE**)
- [x] **Contract rules:** C1-C13 (13 rules with summary/full expansion) (**DONE**)
- [x] **Invariants:** I1-I20 (20 invariants with INV-ID mapping) (**DONE**)

### 5.2.1.2 Macro Grammar
- [x] Grammar: `RADICAL[OPERATOR][NUMBER][:CONTEXT]` (**DONE**)
- [x] Examples: C3, I5, C*, V!, C3:build (**DONE**)
- [x] Parser in `codebook_lookup.py` (**DONE**)

### 5.2.1.3 Macro Specification Document
- [x] Updated `CODEBOOK.json` with grammar, radicals, operators (**DONE**)
- [x] Legacy mappings for migration (@DOMAIN_GOVERNANCE → G) (**DONE**)
- [x] Token metrics documented (60% savings) (**DONE**)

**Exit Criteria:**
- [x] 50 macros identified and documented (**DONE** - exceeds 30 minimum)
- [x] Token savings measured: 60% vs verbose (**DONE**)
- [x] Test suite: 21 tests passing (**DONE**)

---

## 5.2.2 Semiotic Symbol Vocabulary ✅ COMPLETE (2026-01-08)

**Purpose:** Define CJK symbols for semantic compression without phonetic mixing.

> **SIMPLIFIED (2026-01-08):** Instead of a separate CODEBOOK.json schema, symbols are defined directly in `codebook_lookup.py` with a human reference document. Phonetic glosses removed - symbols point directly to semantic regions.

### 5.2.2.1 Symbol Vocabulary
- [x] Core domain pointers: 法, 真, 契, 恆, 驗 (**DONE**)
- [x] Governance operations: 證, 變, 冊, 錄, 限, 許, 禁, 雜, 復 (**DONE**)
- [x] Validation operations: 試, 查, 載, 存, 掃, 核 (**DONE**)
- [x] Structural symbols: 道, 圖, 鏈, 根, 枝 (**DONE**)
- [x] Compound symbols: 法.驗, 法.契, 證.雜, 冊.雜 (**DONE**)

### 5.2.2.2 Codifier Document
- [x] Create `LAW/CANON/SEMANTIC/CODIFIER.md` (**DONE** - human reference)
- [x] Document symbol → path mappings (**DONE**)
- [x] Measured compression ratios (**DONE**)
- [x] Usage patterns (**DONE**)

### 5.2.2.3 Pure Symbolic Approach
- [x] Remove phonetic glosses from code (**DONE** - no "name": "law" mixing)
- [x] Symbols point directly to paths (**DONE**)
- [x] Compression ratios tracked per symbol (**DONE**)

**Exit Criteria:**
- [x] 25+ symbols defined (29 total)
- [x] No phonetic/ideographic mixing
- [x] Human reference document created

**Implementation:**
- `CAPABILITY/TOOLS/codebook_lookup.py` - Symbol definitions (SEMANTIC_SYMBOLS dict)
- `THOUGHT/LAB/FORMULA/CODIFIER.md` - Human translation reference (符典)
- `CAPABILITY/MCP/schemas/tools.json` - MCP schema updated

---

## 5.2.3 Implement Stacked Symbol Resolution ✅

**Purpose:** Stack symbol resolution with CORTEX for precision retrieval.

> **SIMPLIFIED (2026-01-08):** Instead of building a complex AST decoder, stack the working `codebook_lookup.py` with existing CORTEX FTS. Same outcome, less complexity.

### 5.2.3.1 Stacked Resolution (SIMPLIFIED PATH) ✅ COMPLETE
- [x] Create `CAPABILITY/TOOLS/codebook_lookup.py` - **DONE**
- [x] CJK single-token symbols (法, 真, 契, 驗, 恆, 道) - **DONE**
- [x] MCP integration (codebook_lookup tool) - **DONE**
- [x] Add `query` parameter for FTS within domain - **DONE**
- [x] Add `semantic` parameter for vector search within domain - **DONE**
- [x] Stack: symbol → domain narrowing → CORTEX search → relevant chunks - **DONE**

### 5.2.3.2 Stacked Query Syntax
```python
# L1 only (dumps everything)
codebook_lookup(id="法", expand=True)  # → 56,370 tokens

# L1+L2 (FTS precision)
codebook_lookup(id="法", query="verification")  # → ~4,200 tokens

# L1+L3 (semantic precision)
codebook_lookup(id="法", semantic="verification protocols")  # → ~2,000 tokens
```

### 5.2.3.3 Tests ✅ COMPLETE
- [x] `test_phase_5_2_3_stacked_resolution.py` - **5 tests passing**
- [x] Symbol only → full domain content - **DONE**
- [x] Symbol + query → FTS filtered content - **DONE**
- [x] Symbol + semantic → vector filtered content - **DONE**
- [x] Compression ratios measured and receipted - **DONE**

**Exit Criteria:**
- [x] Symbol resolution working (56,370x proven)
- [x] Stacked resolution with CORTEX integration - **DONE**
- [x] Precision retrieval (high alignment) verified - **DONE**

### Original Complex Path (DEFERRED)
The original plan for AST parsing and template expansion is deferred. If the simplified stacked approach proves insufficient, revisit:
- Parser implementation with tokenize/parse/AST
- Template variable substitution
- Nested macro expansion

**Rationale:** Follow the entropy gradient. The simpler path achieves the same compression goals with existing components.

---

## 5.2.4 Implement SCL Validator ✅

**Purpose:** Validate symbolic programs and expanded outputs.

### 5.2.4.1 Symbolic Validation ✅ COMPLETE
- [x] Create `CAPABILITY/PRIMITIVES/scl_validator.py` - **DONE**
- [x] Check: syntax valid, all symbols known, params match - **DONE**
- [x] L1 (Syntax), L2 (Symbol), L3 (Semantic) validation layers - **DONE**

### 5.2.4.2 Expansion Validation ✅ COMPLETE
- [x] Expanded JSON passes JobSpec schema - **DONE**
- [x] Outputs in allowed roots (C8/I6 enforcement) - **DONE**
- [x] No forbidden operations - **DONE**

### 5.2.4.3 Tests ✅ COMPLETE
- [x] Valid programs pass - **38 tests passing**
- [x] Invalid syntax fails with clear error - **DONE**
- [x] Unknown symbols fail - **DONE**
- [x] Schema-invalid expansions fail - **DONE**

**Exit Criteria:**
- [x] Validator catches all error classes - **DONE**
- [x] Clear error messages - **DONE**

**Implementation:**
- `CAPABILITY/PRIMITIVES/scl_validator.py` - 4-layer validator (L1-L4)
- `CAPABILITY/TESTBENCH/integration/test_phase_5_2_4_scl_validator.py` - 38 tests

---

