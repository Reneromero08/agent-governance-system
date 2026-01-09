---
title: Semiotic Codifier
status: Canonical
version: 1.0.0
created: 2026-01-08
author: Rene + Claude
purpose: Human reference for symbolic vocabulary
---
<!-- CONTENT_HASH: PENDING -->

# 符典 (Fúdiǎn) - The Semiotic Codifier

**This document is for HUMAN REFERENCE only.**

Symbols do not carry glosses in code. They point directly to semantic regions.
The receiver accesses meaning through shared context, not phonetic transcription.

---

## Principle: Semantic Density

> The symbol 道 doesn't compress four concepts into one token. It *points at* a region
> of semantic space. The receiver doesn't decode meanings. They access the region directly.
> — Platonic Compression Thesis

Mixing ideographs with phonetic glosses is oxymoronic. The symbol IS the compression.

---

## Core Domain Symbols

| 符 | Domain | Path | Measured Compression |
|:--:|--------|------|---------------------|
| 法 | Canon Law | `LAW/CANON/*` | 56,370× |
| 真 | Truth Foundation | `LAW/CANON/FOUNDATION/THE_SEMIOTIC_FOUNDATION_OF_TRUTH.md` | 8,200× |
| 契 | Contract | `LAW/CANON/CONSTITUTION/CONTRACT.md` | 4,100× |
| 恆 | Invariants | `LAW/CANON/CONSTITUTION/INVARIANTS.md` | 5,600× |
| 驗 | Verification | `LAW/CANON/GOVERNANCE/VERIFICATION.md` | 3,800× |

---

## Governance Operations

| 符 | Operation | Semantic Region |
|:--:|-----------|-----------------|
| 證 | Receipt/Proof | `NAVIGATION/RECEIPTS/*`, proof artifacts |
| 變 | Mutation/Change | Catalytic transformations, state changes |
| 冊 | Registry/Catalog | Model registry, skill registry, indices |
| 錄 | Index/Record | CORTEX indices, search structures |
| 限 | Constraint/Limit | Boundary rules, validation constraints |
| 許 | Permission/Allow | Allowed domains, output roots |
| 禁 | Forbidden/Deny | Prohibited operations, violations |
| 雜 | Hash/Digest | Content hashes, integrity proofs |
| 復 | Restore/Return | Rollback, recovery operations |

---

## Validation Operations

| 符 | Operation | Semantic Region |
|:--:|-----------|-----------------|
| 試 | Test/Validate | Test execution, fixture validation |
| 查 | Search/Query | Semantic search, CORTEX queries |
| 載 | Load/Read | File access, content retrieval |
| 存 | Store/Write | Artifact creation, CAS operations |
| 掃 | Scan/Enumerate | Directory scans, inventory operations |
| 核 | Verify/Audit | Integrity checks, compliance audits |

---

## Structural Symbols

| 符 | Structure | Semantic Region |
|:--:|-----------|-----------------|
| 道 | Path/Principle | Context-activated: path, principle, method, speech |
| 圖 | Graph/Map | Relationship structures, cross-references |
| 鏈 | Chain/Sequence | DAG nodes, execution chains |
| 根 | Root | Base paths, origin points |
| 枝 | Branch/Tree | Hierarchical structures, git branches |

---

## Compound Symbols

Symbols compose with `.` operator for precision:

| Compound | Meaning | Expansion |
|----------|---------|-----------|
| 法.驗 | Canon verification | Verification files within LAW/CANON |
| 法.契 | Canon contract | CONTRACT.md specifically |
| 證.雜 | Receipt hash | Hash of receipt content |
| 冊.雜 | Registry fingerprint | Hash of registry state |

---

## Polysemic Symbols

Some symbols activate different meanings based on context:

### 道 (dào)
```
CONTEXT_PATH    → "the path, the way to follow"
CONTEXT_PRINCIPLE → "the underlying principle"
CONTEXT_SPEECH  → "to speak, to express"
CONTEXT_METHOD  → "the technique, the approach"
```

The receiver resolves via context, not lookup table.

---

## Usage Patterns

### In Prompts (Agent-to-Agent)
```
法.驗 → execute verification protocol
證 → emit receipt
```

### In Code (Resolution)
```python
resolve("法")        # → Path("LAW/CANON")
resolve("法.驗")     # → Path("LAW/CANON/GOVERNANCE/VERIFICATION.md")
expand("法")         # → [full canon content, 56,370 tokens]
```

### Stacked Resolution (L1/L2/L3)
```
法                    # L1: Full domain (56,370 tokens)
法 + query("verify")  # L2: FTS filtered (~4,000 tokens)
法 + semantic("verification protocols")  # L3: Vector filtered (~2,000 tokens)
```

---

## Symbol Selection Criteria

1. **Single Token**: Symbol must tokenize as 1 token (verified via tiktoken)
2. **Semantic Load**: Symbol carries meaning in source language
3. **Non-Collision**: No overlap with existing @ codebook entries
4. **Governance Relevance**: Maps to AGS domain concept

---

## Relationship to @ Codebook

The @ codebook (`LAW/CANON/META/CODEBOOK.md`) uses ASCII identifiers:
- @C3 → Contract Rule 3
- @I5 → Invariant 5
- @S12 → Skill 12

CJK symbols are a SEPARATE compression layer:
- 法 → Entire LAW/CANON domain
- 契 → CONTRACT.md (contains all @C rules)

Both systems coexist. Use @ for rule-level precision, 符 for domain-level compression.

---

## Verification

All symbols are validated by:
1. Single-token verification via `tiktoken` (cl100k_base)
2. Path resolution test (symbol → valid filesystem path)
3. Expansion test (content accessible and hashable)
4. Determinism test (same symbol → same content hash)

---

*This codifier is the Rosetta Stone for AGS semiotic compression.*
*Symbols compress. Glosses are for humans. Code uses pure mappings.*
