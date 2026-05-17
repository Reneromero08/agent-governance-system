<!-- CONTENT_HASH: PENDING -->

# CAT_CHAT: Compressed Report

**CAT_CHAT** (Catalytic Chat) is a deterministic, verifiable chat substrate --
virtual memory for LLMs. Defined across ~15 active markdown docs plus
specs and archives in `THOUGHT/LAB/CAT_CHAT/`.

## Core Definition (from CAT_CHAT_README_1.1.md)

> A deterministic chat substrate: models write compact, structured messages
> that reference canonical material via **symbols**, and workers expand only
> **bounded slices** of source text when needed. Reproducible, auditable,
> fail-closed.

## Core Objects (from CAT_CHAT_CONTRACT.md)

| Object | Purpose | Fail-Closed Triggers |
|--------|---------|----------------------|
| Section | SHA-256 content unit from source files | Missing file, bad range, hash mismatch |
| Symbol | `@NAMESPACE/name` pointer to sections | Invalid format, unknown target, unresolvable |
| Message | Structured intent + refs + ops + budgets | Empty intent, invalid symbols, missing budgets |
| Expansion | Bounded slice retrieval (`lines[a:b]`) | `slice=ALL`, exceeds bounds, hash mismatch |
| Receipt | Immutable execution record (chain-verified) | Missing fields, incomplete records |

## The Catalytic Loop (from ROADMAP v2.0, Phase C)

Every turn:
1. Score ALL stored items against current query
2. Hydrate high-E items into working set (bounded by budget)
3. Generate response using only working set
4. Compress old turns to hash pointers
5. Evict lowest-E items when budget exceeded

E-score = Born rule: `|psi>phi|^2` (validated r=0.999 in Q44 research).

## Roadmap Status (from CAT_CHAT_ROADMAP_2.0.md)

| Phase | Component | Status |
|-------|-----------|--------|
| A | Session Persistence Tests | DONE |
| B | Cassette Network Integration | DONE |
| C | Auto-Controlled Context Loop | Core DONE (C.6.3 pending) |
| D | SPC Pointer Compression | DONE |
| E | Vector Fallback | DONE |
| F | Docs Index (FTS) | DONE |
| G | Bundle Replay & Verification | DONE |
| H | Specs & Golden Demo | DONE |
| I | Measurement & Benchmarking | DONE |
| J | Recursive E-Score Hierarchy | DONE |

## Research Validation (from ROADMAP)

- Born rule E-score: r=0.999 (Q44)
- SQuAD hierarchy: 85% recall (97% of brute force), 5.6x speedup
- Iso-temporal context: +16.7% recall on 99 arxiv papers
- Symbol compression: 56,370x token reduction

## Relationship to AGS (from WRITE_ISOLATION.md, GRADUATION_PATH.md)

CAT_CHAT is a **consumer** of the main AGS cassette network. It reads from all
9 main cassettes but writes only to `_generated/cat_chat.db`. The
CassetteClient has no write methods -- enforced at module load.

Still lives in `THOUGHT/LAB/` sandbox. 3 graduation options documented:
A) merge to resident.db, B) new cassette, C) keep in LAB permanently.

## Specs (from docs/specs/)

| Spec | Version | What It Defines |
|------|---------|-----------------|
| BUNDLE_SPEC.md | 5.0.0 | Bundle format, hashing, completeness gates |
| RECEIPT_SPEC.md | 1.0.0 | Receipt format, chain integrity, Merkle root |
| TRUST_SPEC.md | 1.0.0 | Trust policies, validator pinning |
| EXECUTION_SPEC.md | 1.0.0 | Execution semantics, exit codes |

## Ground Rules (from README)

- Deterministic on stdout, logs/progress to stderr
- No timestamps or randomness in IDs
- Bounded slices only -- `ALL` is an invariant violation
- Fail-closed on all constraint violations
