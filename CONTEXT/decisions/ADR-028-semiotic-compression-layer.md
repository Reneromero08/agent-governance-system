# ADR-028: Semiotic Compression Layer (SCL)

**Status:** Accepted
**Date:** 2025-12-28
**Confidence:** High
**Impact:** High
**Tags:** [compression, semiotics, tokens, optimization]

## Context
Standard RAG and context-stuffing methods are token-inefficient. "The Living Formula" suggests maximizing minimal information ($f$) raised to the power of fractal depth ($D_f$). We observe that 90% of token usage in agentic systems is repetitive boilerplate (structure, headers, known context).

## Decision
We establish the **Semiotic Compression Layer (SCL)** as the primary method for agent-to-agent and system-to-agent communication.

### Core Mechanism: The Symbol (`@ID`)
Instead of transmitting full text, we transmit **Symbols**.
- **@C3**: "Canon > User" (implies full text of Contract Rule 3).
- **@T:critic**: "The Critic Tool" (implies path, usage, arguments).
- **@F0**: "The Living Formula" (implies the entire driver philosophy).

### The Symbolic IR (Intermediate Representation)
Agents operating in "Autonomous Mode" (Ants) shall prefer a **Symbolic IR** over natural language.
- **Input**: `{"task": "@S12", "target": "@T:api/v1"}`
- **Output**: `{"status": "success", "ref": "hash:a8f9..."}`

### Requirements
1. **Codebook (@B0)**: The Rosetta Stone. Must be auto-generated and strictly versioned.
2. **Expansion Tooling**: Agents must have a tool (`codebook_lookup`) to "explode" a symbol into full text if they lack semantic understanding.
3. **Canonical Prefix**: All SCL symbols use the `@` prefix for easy regex detection.

## Consequences
1. **90% Token Reduction**: LITE packs and prompts shrink drastically.
2. **Precision**: Ambiguity is removed. `@I5` means exactly Invariant 5, nothing else.
3. **Drift Resistance**: Changing the text of Rule 3 doesn't break the symbol `@C3`, preserving historical references (until major version bump).
4. **Implementation**: Requires `AGS_ROADMAP_MASTER.md` Lane I updates (P1).
