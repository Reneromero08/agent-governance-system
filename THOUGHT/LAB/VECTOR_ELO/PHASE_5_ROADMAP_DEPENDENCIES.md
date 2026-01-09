---
title: Phase 5 Dependencies
section: roadmap
version: 1.8.0
created: 2026-01-07
modified: 2026-01-08
status: Active
summary: Phase 5 dependencies, file locations, and test targets
tags:
- phase-5
- vector
- semiotic
- roadmap
---
<!-- CONTENT_HASH: 108dcfd5b1989b960d4079776ed13f1064ffb9e51b28a73b32871903c5d0db19 -->

# Phase Dependencies

## Critical Path
5.1 → 5.2 → 6.0 → 6.x (each layer compounds on previous)

## Phase 5.2 → Phase 6.0 Handoff (CAS Layer)

The MemoryRecord contract defined in 5.1.0 becomes the foundation for Phase 6.0 Cassette Network:
- Cassette storage binds to MemoryRecord schema
- Each cassette DB is a portable cartridge artifact
- Derived indexes are rebuildable from cartridges
- **CAS compression:** Content stored external to LLM context, only hash pointers in window

**Handoff Checklist:**
- [ ] MemoryRecord schema finalized and frozen
- [ ] Schema version tagged
- [ ] Migration path documented
- [ ] CAS external storage architecture validated

## Phase 6.x → Session Cache Layer

Future phase to implement session-level compression:
- Query 1 (cold): Full symbolic exchange
- Query 2-N (warm): Hash confirmation only (~1 token)
- Requires: Session state persistence, cache invalidation strategy

**Dependency:** Requires 6.0 Cassette Network operational

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
│   ├── token_receipt.py           # 5.2.7: TokenReceipt primitive
│   ├── token_session.py           # 5.2.7: Session aggregator
│   └── schemas/
│       ├── memory_record.schema.json
│       ├── scl_codebook.schema.json
│       └── token_receipt.schema.json
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
│   ├── SYMBOL_GRAMMAR.md          # EBNF syntax
│   └── TOKEN_RECEIPT_SPEC.md      # 5.2.7: TokenReceipt law
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
| 5.2.7 | Token Accountability | 10+ |
| **Total** | | **~75 tests** |
