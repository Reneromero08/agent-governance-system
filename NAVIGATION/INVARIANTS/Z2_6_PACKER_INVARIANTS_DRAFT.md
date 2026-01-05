<!-- CONTENT_HASH: f23a03a6df505d33546df479bd7a277b6e7d93a6a1fc74b6417309bd01adcc40 -->

# Z.2.6 LLM Packer Integration Invariants

Status: CANONICAL (Z.2.6)
Scope: How LLM Packer interacts with CAS, ARTIFACTS, RUNS, and GC (Z.2.5). This document defines what must be rooted, when run records are emitted, and what must never be collected.

## Goals
- Packer outputs are provably non-destructive and reproducible.
- Every pack run produces immutable run records and root references.
- GC (Z.2.5) can safely delete unreferenced blobs without risking pack integrity.

## Terminology
- CAS blob: bytes stored via `cas_put(bytes) -> hash`
- Artifact ref: `sha256:<hash>` from `store_bytes/store_file`
- Run record: immutable CAS-backed records from `CAPABILITY/RUNS/*`
- Root: an explicit reference that prevents GC collection

## Root policy

### R1. What becomes a root
A pack run must register as roots the minimum set required to reproduce and audit the run:

1) PACK TASK_SPEC root (required)
- The canonical input specification for the pack run.
- Stored via run record API, yields a CAS hash.
- This hash MUST be a root.

2) PACK OUTPUT_HASHES root (required)
- Deterministic ordered list of artifact refs produced by the pack run.
- This list MUST be stored immutably and its CAS hash MUST be a root.

3) PACK STATUS root (required)
- Final status record including verdict and summary.
- Intermediate statuses are optional. Final status is required.
- Final status hash MUST be a root.

4) PACK MANIFEST root (recommended, if distinct from OUTPUT_HASHES)
- If pack produces a manifest describing structure (index, mapping, metadata), store it as an immutable artifact and include it in OUTPUT_HASHES, or store separately and root it explicitly.
- Prefer including it in OUTPUT_HASHES so one root covers it.

5) Pin file roots (optional)
- Allowed only for operator-managed retention (examples: keep known-good snapshots, legal holds).
- Pin file entries are roots by definition.

### R2. What is NOT a root
- Temporary files, scratch directories, caches
- Progress logs or transient debug output unless explicitly stored as artifacts and referenced by OUTPUT_HASHES
- Local working directories used during packing

## Emission policy for run records

### E1. When run records are emitted
1) Before execution begins
- Emit TASK_SPEC as an immutable run record.
- Register TASK_SPEC hash into the run roots index.

2) During execution (optional)
- Emit STATUS updates only if you need auditability of intermediate steps.
- If emitted, these must be immutable CAS-backed STATUS records, but only the final STATUS is required to be rooted.

3) After outputs are finalized (required)
- Emit OUTPUT_HASHES as immutable run record once all artifact refs are known and stable.

4) Completion (required)
- Emit final STATUS record (state, verdict, summary).
- Ensure TASK_SPEC, OUTPUT_HASHES, final STATUS are registered as roots in a deterministic roots source (RUN_ROOTS).

### E2. Fail-closed rule
If the packer cannot emit required run roots (TASK_SPEC, OUTPUT_HASHES, final STATUS), the run must be treated as failed and must not claim success.

## Never-collect policy (must not be GC'd)

### N1. Substrate and root invariants
GC must never delete:
- Any blob reachable from roots (by definition).
- Any blob referenced by the current RUN_ROOTS index.
- Any blob referenced by GC_PINS.

Additionally, these foundational paths must never be removed by tooling:
- `CAPABILITY/CAS/`
- `CAPABILITY/ARTIFACTS/`
- `CAPABILITY/RUNS/`
- `CAPABILITY/GC/`
- `NAVIGATION/INVARIANTS/`

### N2. Operational rule
GC must not run concurrently with an active pack run unless strict locking is implemented across both:
- Packer obtains a pack lock.
- GC obtains a GC lock.
- Locks must prevent simultaneous sweep while pack is materializing or committing roots.

## Determinism requirements for packer integration
- OUTPUT_HASHES list must be stable and ordered deterministically.
- Any generated manifest must be canonicalized deterministically.
- Receipts must be canonical JSON (stable key ordering, stable list ordering).

## Acceptance criteria for Z.2.6 integration
- A pack run produces and registers the required roots.
- GC dry-run never proposes deletion of any pack outputs referenced by those roots.
- A pack run is reproducible from TASK_SPEC and OUTPUT_HASHES alone (plus code).

---

## Recovery: Packer Invariant Violations

### Where receipts live

Packer receipts and audit trails are stored in:
- **MEMORY/LLM_PACKER/_packs/** - Pack outputs (manifests, archives)
- **CAPABILITY/RUNS/RUN_ROOTS.json** - Root registration for pack outputs
- **LAW/CONTRACTS/_runs/audit_logs/** - Audit logs from pre/post pack checks

### How to re-run verification

To verify pack integrity:

```bash
# Verify a specific pack's manifest and blobs
python -m MEMORY.LLM_PACKER.Engine.packer.consumer verify --manifest-ref sha256:PACK_MANIFEST_HASH

# Verify GC safety of packer roots
python CAPABILITY/AUDIT/root_audit.py --mode B --output-hashes-record RUN_OUTPUT_HASHES_HASH
```

### What to delete vs never delete

**Safe to delete:**
- **Intermediate artifacts**: In `_runs/_tmp/` or local `_packs/` if not shared.
- **Unsealed/Uncompressed copies**: If the canonical sealed version exists and is rooted.

**Never delete (protected by invariants):**
- **Blobs referenced by active packs**: Deleting these breaks the pack.
- **Pack Manifests**: These are the entry points to the pack graph.
- **RUN_ROOTS.json entries for active packs**: Removing these allows GC to eat the pack's blobs.

**Recovery procedures:**
- **Pack Verification Failed**:
  - Re-run the pack generation with the same `TASK_SPEC` to regenerate missing blobs (determinism guarantees identical hashes).
- **GC Ate My Pack**:
  - If a root was missing and GC ran, the blobs are gone.
  - Re-generate the pack from source.
- **Root Conflict**:
  - If multiple packs claim conflicting roots (unlikely with CAS), trust the union of both.
