<!-- CONTENT_HASH: PENDING -->

# Agent Governance System (AGS)

**What if AI agents had a constitution they couldn't break?**

AGS is a framework that makes AI agents **governable, verifiable, and accountable**. Every decision is recorded. Every action is traceable. Every output is provable. The system enforces rules that agents cannot bypass - not through trust, but through architecture.

## The Problem

AI agents are powerful but ungoverned. They hallucinate. They drift. They make decisions with no audit trail. When something goes wrong, you can't trace what happened or why. There's no constitution, no accountability, no proof of what actually occurred.

## The Solution

AGS implements **constitutional AI governance**:

- **Canon** - Immutable rules that define what agents can and cannot do (like a constitution)
- **Context** - Every decision recorded as an ADR (Architecture Decision Record) with rationale
- **Contracts** - Fixtures that mechanically verify behavior before any change is accepted
- **Catalytic Computing** - Agents can use your entire codebase as "borrowed memory" and provably restore it
- **Cassette Network** - 9 federated semantic databases for intelligent search across everything

The key insight: **Text is law. Code is consequence.** The canonical documents define the spec. The code just implements it. If they conflict, the text wins.

## The Living Formula

AGS is built on a mathematical theory of meaning:

```
R = (E / nabla S) * sigma(f)^Df
```

This isn't just notation - it's the claim that **meaning is measurable** and **drift from reality is detectable**. When an agent hallucinates or fabricates, the math catches it. Truth compresses better than lies (85x in our tests), so over time, reality wins on cost.

| Result | Evidence |
|--------|----------|
| Cross-model semantic convergence | 0.971 |
| Cross-lingual convergence | 0.914 |
| Compression advantage of truth | 85x |
| Symbol compression | 99.4% token reduction |

54 research questions tracked, 81.5% answered with reproducible evidence.

## What You Get

### Governed AI Agents
- 20 invariants that cannot be violated without triggering failures
- Every commit requires explicit approval (no autonomous pushes)
- Multi-agent workflows with mandatory workspace isolation
- Crisis modes and emergency procedures when things go wrong

### Semantic Intelligence
- **Cassette Network**: 9 databases covering law, capabilities, research, and AI memories
- **MCP Server**: 11 tools for search, memory, skill execution - works with Claude Desktop
- **Symbol Compression**: Reference entire documents with single tokens (99.4% reduction)

### Verifiable Execution
- **Catalytic Computing**: Borrow O(n) codebase as memory, restore exactly, prove it
- **Content-Addressable Storage**: Every artifact hashed and verifiable
- **SPECTRUM Protocols**: Cryptographic proofs of execution

## Proven Claims

These aren't just assertions - they're validated with reproducible tests:

| Claim | Measured | Test Location |
|-------|----------|---------------|
| **Compression: H(X|S) ~ 0** | 99.89% reduction | `CAPABILITY/TESTBENCH/cassette_network/compression/` |
| **Task parity preserved** | 100% (8/8 tasks) | `NAVIGATION/PROOFS/COMPRESSION/COMPRESSION_PROOF_REPORT.md` |
| **Catalytic restoration** | PASS (6-step verify) | `NAVIGATION/PROOFS/CATALYTIC/PROOF_CATALYTIC_REPORT.md` |
| **Ground truth retrieval** | 100% (12/12 tests) | `CAPABILITY/TESTBENCH/cassette_network/ground_truth/` |

The compression proof validates that sharing cassettes reduces information needed by 99.9% while preserving task success. The catalytic proof validates the create-export-corrupt-import-verify chain.

See: [NAVIGATION/PROOFS/](NAVIGATION/PROOFS/) for full evidence bundles.

### 33 Skills
Pre-built capabilities: `workspace-isolation`, `commit-manager`, `admission-control`, `cortex-toolkit`, and more. Each skill has fixtures that must pass before changes are accepted.

## Project Structure

```
LAW/           <- Constitution: CANON (rules), CONTEXT (decisions), CONTRACTS (enforcement)
CAPABILITY/    <- Execution: SKILLS/, MCP/, TOOLS/, PRIMITIVES/
NAVIGATION/    <- Intelligence: CORTEX (semantic search), MAPS/, PROMPTS/
THOUGHT/       <- Research: LAB/FORMULA/, experiments, proofs
MEMORY/        <- State: LLM packs, archives
INBOX/         <- Human-readable reports and documents
```

## Quick Start

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate        # Windows (or source .venv/bin/activate on Unix)
pip install -r requirements.txt

# Verify everything works
pytest CAPABILITY/TESTBENCH/ -v

# Start the MCP server (integrates with Claude Desktop)
python LAW/CONTRACTS/ags_mcp_entrypoint.py

# Search the codebase semantically
python CAPABILITY/MCP/semantic_adapter.py search --query "how do I add a skill"
```

## Key Documents

| Start Here | What It Is |
|------------|------------|
| [AGENTS.md](AGENTS.md) | The full agent operating contract |
| [CONTRACT.md](LAW/CANON/CONSTITUTION/CONTRACT.md) | 13 non-negotiable rules |
| [INVARIANTS.md](LAW/CANON/CONSTITUTION/INVARIANTS.md) | 20 locked decisions |
| [FORMULA.md](LAW/CANON/CONSTITUTION/FORMULA.md) | The mathematical foundation |
| [GENESIS.md](LAW/CANON/META/GENESIS.md) | Bootstrap prompt for new sessions |

## The License

**Catalytic Commons License v1.4**

Section infinity: **Zero rights** to any entity exercising coercive authority - governments, military, intelligence agencies, law enforcement, surveillance contractors. This isn't open source for everyone. It's open source for everyone except those who would use it for control.

---

**Canon Version**: 3.0.0 | **Author**: Raúl René Romero Ramos

*"Reality wins on cost. Lies become more expensive to maintain."*
