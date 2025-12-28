# Integrity Law (CMP/SPECTRUM)

**Essence ($E$):** Truth is found in declared artifacts and deterministic verification, not narrative history.

## 1. CMP-01: Runtime Guard
Every execution MUST pass through the CMP-01 validator.
- **Durable Roots**: `CONTRACTS/_runs/`, `CORTEX/_generated/`, `MEMORY/LLM_PACKER/_packs/`.
- **Forbidden**: `CANON/`, `AGENTS.md`, `BUILD/`.
- **Pre-run**: Validate JobSpec schema, paths, and forbidden overlaps.
- **Post-run**: Verify every declared durable output exists and is bounded.

## 2. SPECTRUM-01: The Resume Bundle
A run is accepted for resumption ONLY if it produces a verifiable bundle:
- `TASK_SPEC.json` (Immutable JobSpec)
- `STATUS.json` (Success/Failure state)
- `OUTPUT_HASHES.json` (Merkle-tree of produced artifacts)
- `validator_version` (Compatibility gate)

## 3. SPECTRUM-03: Chained Temporal Integrity
Sequential runs form a chain.
- A run may only reference inputs present in the current outputs or previously verified runs in the same chain.
- Any change to a middle run’s output invalidates the entire chain suffix.

## 4. The 5 Invariants of Integrity
1. **Declared Truth**: If an output is durable, it must be declared and hashed.
2. **Disposable State**: Catalytic domains (`_tmp/`) are for work only; they never hold system truth.
3. **Narrative Independence**: History (logs/chats) is a reference only; it is not required for execution success.
4. **Byte-Level Precision**: Tampering is detected at the bit level by SHA-256.
5. **Partial Ordering**: Time is a DAG of sealed checkpoints, not a linear narrative.
