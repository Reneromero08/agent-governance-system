# Vector Research Index

**Location:** `THOUGHT/LAB/VECTOR_ELO/research/`
**Created:** 2026-01-07
**Purpose:** Consolidated vector database, embedding, and semantic memory research

---

## Directory Structure

```
research/
├── vector-substrate/     # VectorPack format, MemoryRecord contract
├── cassette-network/     # Cassette protocol, semantic network hub
├── semantic-core/        # Semantic core architecture, indexing
└── phase-5/              # Phase 5 Vector/Symbol Integration roadmaps
```

---

## vector-substrate/

Core specifications for vector data structures and transfer formats.

| File | Description |
|------|-------------|
| `01-06-2026-21-13_5_2_VECTOR_SUBSTRATE_VECTORPACK.md` | MemoryRecord contract, VectorPack format, micro-pack export |

---

## cassette-network/

Cassette protocol and semantic network implementation.

| File | Description |
|------|-------------|
| `01-06-2026-21-13_6_0_CANONICAL_CASSETTE_SUBSTRATE.md` | Canonical cassette substrate (cartridge-first) |
| `12-28-2025-23-12_CASSETTE_NETWORK_IMPLEMENTATION_REPORT.md` | Phase 0 POC complete, 3,991 chunks indexed |
| `12-28-2025-23-12_SEMANTIC_DATABASE_NETWORK_REPORT.md` | Semantic database network architecture |
| `12-28-2025-12-00_ROADMAP_DATABASE_CASSETTE_NETWORK.md` | Database cassette network roadmap |
| `12-28-2025-12-00_CASSETTE_NETWORK_ROADMAP_FINAL_CHECKLIST.md` | Final checklist for implementation |
| `01-06-2026-12-58_CASSETTE_NETWORK_ROADMAP_1.md` | Archived roadmap v1 |
| `01-06-2026-12-58_CASSETTE_NETWORK_ROADMAP_2.md` | Archived roadmap v2 |

---

## semantic-core/

Semantic core development and indexing infrastructure.

| File | Description |
|------|-------------|
| `12-28-2025-12-00_ROADMAP_SEMANTIC_CORE.md` | Semantic Core + Translation Layer roadmap |
| `12-29-2025-05-36_SEMANTIC_CORE_PHASE1_FINAL_REPORT.md` | Phase 1 final report |
| `12-29-2025-05-36_SEMANTIC_CORE_IMPLEMENTATION_REPORT.md` | Implementation details |
| `12-29-2025-05-36_SEMANTIC_CORE_BUILD_COMPLETE.md` | Build completion report |
| `12-29-2025-05-36_SEMANTIC_CORE_WORKFLOW_COMPLETE.md` | Workflow completion |
| `12-29-2025-05-36_SEMANTIC_CORE_PHASE1_COMPLETE.md` | Phase 1 completion |
| `12-23-2025-12-00_SYSTEM1_SYSTEM2_DUAL_DB.md` | System 1/System 2 dual database architecture |
| `12-29-2025-03-47_CORTEX_ROADMAP.md` | CORTEX system roadmap |
| `12-28-2025-23-12_MECHANICAL_INDEXING_REPORT.md` | Mechanical indexing report |

---

## phase-5/

Phase 5 Vector/Symbol Integration planning and research.

| File | Description |
|------|-------------|
| `01-07-2026_PHASE_5_VECTOR_SYMBOL_INTEGRATION.md` | Detailed Phase 5 roadmap (65+ tests) |
| `01-07-2026_PHASE_5_RESEARCH_FINDINGS.md` | Consolidated research findings |
| `12-29-2025-07-01_SEMIOTIC_COMPRESSION.md` | Semiotic Compression Layer (SCL) spec |
| `12-26-2025-06-39_SYMBOLIC_COMPRESSION.md` | Original symbolic compression research |
| `01-06-2026-21-13_CAT_CHAT_CATALYTIC_CONTINUITY.md` | Catalytic continuity design |

---

## Key Specifications

### MemoryRecord Contract
From `vector-substrate/01-06-2026-21-13_5_2_VECTOR_SUBSTRATE_VECTORPACK.md`:
- `id`: Content hash (SHA-256)
- `text`: Canonical text (source of truth)
- `embeddings`: Model name → vector array
- `payload`: Metadata (tags, timestamps, roles)
- `scores`: ELO, recency, trust, decay
- `lineage`: Derivation chain
- `receipts`: Provenance hashes

### Key Principles
1. **Text is canonical** - source of truth
2. **Vectors are derived** - rebuildable from text
3. **All exports are receipted and hashed**
4. **Measure first, build second** (Phase 5 Reality Check)

---

## Related Files (Not Moved)

These files reference vector concepts but remain in their original locations:

- `THOUGHT/LAB/VECTOR_ELO/VECTOR_ELO_ROADMAP.md` - ELO scoring system
- `THOUGHT/LAB/VECTOR_ELO/VECTOR_ELO_SPEC.md` - ELO specification
- `THOUGHT/LAB/CAT_CHAT/catalytic_chat/experimental/vector_store.py` - Implementation
- `AGS_ROADMAP_MASTER.md` - Phase 5 tasks (references moved docs)
