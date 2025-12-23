# Catalytic Computing

This document defines catalytic computing for the Agent Governance System. It separates formal theory from engineering translation so agents do not hallucinate implementation details.

## Formal Model (Complexity Theory)

Catalytic space computation uses two kinds of memory:

1. **Clean space** - A small amount of blank working memory the algorithm can use freely.
2. **Catalytic space** - A much larger memory that starts in an arbitrary state and must be returned to exactly that state at the end.

The key constraint: the algorithm must work for any initial catalytic content (possibly incompressible) and cannot permanently encode new information. The catalytic bits act like a catalyst in chemistry - they enable the computation but remain unchanged afterward.

**Key results** (Buhrman, Cleve, Koucky, Loff, Speelman, 2014):
- Catalytic logspace can compute uniform TC^1 circuits (includes matrix determinant)
- Upper bound: catalytic logspace is contained in ZPP
- The restoration constraint does not kill usefulness; it forces reversible, structured transformations

**TreeEval advances** (Cook, Mertz, 2020-2024):
- Applied catalytic ideas to reduce Tree Evaluation to O(log n * log log n) space
- Shifted expectations about whether TreeEval separates P from L

## AGS Translation

For AGS, catalytic computing provides a memory model:

| Formal Concept | AGS Analog | Examples |
|----------------|------------|----------|
| Clean space | Context tokens | LITE pack contents, working memory |
| Catalytic space | Disk state | Indexes, caches, generated artifacts |
| Restoration | Repo returns identical | Git worktree, content-addressed cache |

**Core insight**: Large disk state can be used as powerful scratch space if you guarantee restoration. This enables high-impact operations (index builds, pack generation, refactors) while keeping context minimal.

## Five Engineering Patterns

### Pattern 1: Clean Context vs Catalytic Store

Keep context tokens minimal. Use disk as the large, addressable store.

- **Clean context (LITE pack)**: laws, maps, contracts, symbolic indexes, short summaries, retrieval instructions
- **Catalytic store**: full file bodies, generated indexes, caches - addressable by hash or path

### Pattern 2: Restore Guarantee as First-Class Artifact

Every operation that uses "big scratch" must produce:
- A **patch** (what changed)
- A **restore plan** (how to undo)
- A **verification check** (prove restoration happened)

Practical mechanisms:
- Git worktree or temporary checkout
- Overlay filesystem (copy-on-write)
- Content-addressed cache for generated artifacts

### Pattern 3: Reversible Transforms Mindset

When a tool needs scratch:
- Prefer in-place, reversible transforms
- If not possible, use an external journal under allowed artifact roots only

Example operations:
- Build indexes into `CORTEX/_generated/` from source files (no source mutation)
- Generate pack manifests and hashes
- Plan refactors by producing a migration plan, not direct edits

### Pattern 4: Catalytic Compression for Packs

Goal: reduce context tokens while keeping deep recovery possible.

Approach:
- Content-addressed snapshot store (each file body stored by sha256)
- LITE pack includes only pointers, section-level summaries, and a deterministic reconstruction recipe
- Full content is retrievable without re-reading the entire repo

### Pattern 5: Catalytic Workspace Contract

Operations may use a temporary workspace but must end with:
- A clean repo state
- Committed outputs only under allowed artifact roots (`CONTRACTS/_runs/`, `CORTEX/_generated/`, `MEMORY/LLM_PACKER/_packs/`)

See CMP-01 (Catalytic Mutation Protocol) for the full operational specification.

## What Catalytic Computing is NOT

Agents must not misinterpret the metaphor:

| Incorrect Interpretation | Reality |
|--------------------------|---------|
| "Just compression" | Catalytic bits may be incompressible; the algorithm must work regardless |
| "Free scratch space" | You must undo all writes without storing a full undo log in clean space |
| "A license to mutate Canon" | Canon is not catalytic; it is authoritative and follows its own change ceremony |
| "Eventual consistency is fine" | Restoration is exact, not approximate |
| "I can write anywhere temporarily" | Catalytic domains must be declared; out-of-domain writes are hard failures |

## Allowed Catalytic Domains

Operations declared as "catalytic" may temporarily mutate only these paths:
- `CONTRACTS/_runs/_tmp/`
- `CORTEX/_generated/_tmp/`
- `MEMORY/LLM_PACKER/_packs/_tmp/`

Forbidden domains (never catalytic):
- `CANON/` (unless in a governance ceremony, which is not catalytic)
- `AGENTS.md` and root authorities
- `BUILD/` (reserved for user workspace outputs)

## Durable Output Roots

After a catalytic run, artifacts may persist only under:
- `CONTRACTS/_runs/`
- `CORTEX/_generated/`
- `MEMORY/LLM_PACKER/_packs/`

This aligns with INV-006 (Output roots) and ADR-015 (Logging output roots).

## Operational Protocol

For implementation details, lifecycle phases, proof formats, and enforcement hooks, see:
- `CONTEXT/research/Catalytic Computing/CMP-01_CATALYTIC_MUTATION_PROTOCOL.md`

CMP-01 defines the five-phase lifecycle (Declare, Snapshot, Mutate, Commit, Restore, Prove) and the run ledger schema for audit trails.

## References

1. Buhrman, Cleve, Koucky, Loff, Speelman. "Computing with a full memory: Catalytic space" (2014). https://iuuk.mff.cuni.cz/~koucky/papers/catalytic.pdf
2. Cook, Mertz. "Tree Evaluation is in Space O(log n * log log n)" (2023). https://eccc.weizmann.ac.il/report/2023/174/
3. Quanta Magazine. "Catalytic Computing Taps the Full Power of a Full Hard Drive" (2025). https://www.quantamagazine.org/catalytic-computing-taps-the-full-power-of-a-full-hard-drive-20250218/
