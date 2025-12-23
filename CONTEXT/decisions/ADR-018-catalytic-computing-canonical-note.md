# ADR-018: Catalytic Computing Canonical Note

**Status:** Accepted
**Date:** 2025-12-23
**Confidence:** High
**Impact:** Medium
**Tags:** [governance, catalytic-computing, memory-model, restore-guarantee]

## Problem

Agents encountering "catalytic computing" terminology in AGS may:
- Hallucinate implementation details not grounded in theory
- Confuse the formal complexity-theory model with the engineering pattern
- Apply the metaphor incorrectly (e.g., treating it as a license to mutate Canon)
- Miss the core insight: restore guarantees enable powerful scratch-space operations

No canonical document exists that separates theory from engineering translation, defines clear boundaries, and provides actionable patterns.

## Decision

Create `CANON/CATALYTIC_COMPUTING.md` as the authoritative reference for catalytic computing in AGS.

The document will:
1. **Synopsis the formal model** - One paragraph defining catalytic space (clean + catalytic memory, restore constraint)
2. **State key theoretical results** - What the complexity theory proved (TC^1 in catalytic logspace, TreeEval advances)
3. **Map to AGS concepts** - Clean context (tokens) vs catalytic store (disk/indexes/caches)
4. **Define five engineering patterns** - Practical applications with examples
5. **Explicitly reject misinterpretations** - What catalytic computing is NOT
6. **Reference CMP-01** - Point to the operational protocol for implementation details

## Rationale

- **Prevents hallucination**: Single authoritative source stops agents from inventing implementations
- **Enables future work**: C3 (summarization), F2 (scratch layer), and large refactors depend on understanding the restore guarantee
- **Separates concerns**: Theory provides legitimacy; engineering patterns provide utility
- **Low risk**: Pure documentation, no code changes, no invariant modifications

## Consequences

- Agents can understand why "memory that is full" can still be computationally useful
- The restore guarantee becomes a governance primitive, not just a convenience
- Future catalytic workflows (CMP-01) have a conceptual foundation
- Clear boundaries prevent over-application of the metaphor

## Related

- CMP-01: Catalytic Mutation Protocol (operational specification in CONTEXT/research/)
- ADR-015: Logging output roots (establishes allowed artifact roots)
- INV-006: Output roots invariant (enforces where artifacts can be written)
