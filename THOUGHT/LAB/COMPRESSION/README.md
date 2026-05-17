# Compression Lab (Phase 5)

**Status:** Complete (all 529 tests pass, Global DoD met)

Semiotic Compression Layer (SCL) and Semantic Pointer Compression (SPC) for governance artifact token efficiency.

## Contents

| File | Purpose |
|------|---------|
| `ROADMAP.md` | Phase 5 detailed roadmap (5.1: vectors, 5.2: SCL, 5.3: SPC formalization) |
| `PHASE_5_DONE.md` | Archived completed Phase 5.1-5.2.4 tasks |
| `DEPENDENCIES.md` | Cross-phase dependencies and file locations |
| `CODIFIER.md` | Semiotic Codifier (CJK symbols, ASCII macros) |
| `research/` | Research artifacts, papers, and symbol studies |
| `experiments/` | Compression benchmarks |

## Key Results

- **56,370x** semantic compression (single CJK glyph expands to full canon)
- **96.4%** SCL token reduction (334 NL tokens -> 12 symbolic tokens)
- **92.2%** SPC compression with 100% exact match rate
- **0.89 CDR** (concept density ratio: concept_units/token)
- **529 tests** passing across 16 test files
- **MemoryRecord** contract for all vector-indexed content

## Protocols

- SPC: Semantic Pointer Compression (`LAW/CANON/SEMANTIC/SPC_SPEC.md`)
- GOV_IR: Governance Intermediate Representation (`LAW/CANON/SEMANTIC/GOV_IR_SPEC.md`)
- CODEBOOK SYNC: Shared side-information handshake (`LAW/CANON/SEMANTIC/CODEBOOK_SYNC_PROTOCOL.md`)
- TOKEN RECEIPT: Token accountability (`LAW/CANON/SEMANTIC/TOKEN_RECEIPT_SPEC.md`)

## Sibling Labs

- `THOUGHT/LAB/VECTOR_ELO/` — ELO scoring and memory pruning
- `THOUGHT/LAB/EIGEN_ALIGNMENT/` — Cross-model eigenvalue alignment protocol
