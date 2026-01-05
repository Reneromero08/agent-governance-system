<!-- CONTENT_HASH: d259cefff4358a2d79f51343e25a66c1a1fa62db6bce490b60c1f13c492f65d3 -->

# Z2 CAS + Artifact + Run Invariants (Fence)
Status: CANONICAL (Z.2.1–Z.2.4)
Last updated: 2026-01-02

Purpose
This document freezes the non-negotiable invariants introduced by Lane Z, phases Z.2.1–Z.2.4.
Agents and humans must treat these as hard constraints. Any change requires a new explicit roadmap task.

---

## Z.2.1 Core CAS Invariants

CAS Identity
- Content identity is the SHA-256 hash of raw bytes only.
- Hash format is lowercase hex (64 chars).
- Identical bytes MUST always produce the same hash.
- Different bytes SHOULD produce different hashes (cryptographic collision resistance assumed).

Immutability
- CAS objects are write-once.
- Existing objects MUST NOT be rewritten.

Storage Determinism
- Object paths are derived deterministically from the hash.
- No environment-dependent path logic.

Integrity
- CAS writes are atomic.
- CAS verifies integrity by re-reading and re-hashing after write.
- Reads fail closed on missing objects or corruption.

Failure Mode
- No silent fallbacks.
- Invalid hash format, missing object, or corruption MUST raise explicit errors.

---

## Z.2.2 Artifact Store Invariants

Canonical Reference
- CAS-backed artifact references use:
  - `sha256:<64-lowercase-hex>`
- Legacy file path references remain plain strings (no prefix).

Dual-Mode Compatibility
- The system MUST accept both:
  - legacy paths
  - `sha256:` CAS refs
- Z.2.2 does not require call-site migration; it enables gradual migration.

Determinism
- Storing the same bytes yields the same `sha256:` ref (because CAS is deterministic).

Materialization
- `materialize(ref, out_path, atomic=True)` must be deterministic.
- Atomic materialization writes to a temp file then replaces target.

Failure Mode
- Fail closed for:
  - invalid `sha256:` refs
  - missing CAS objects
  - corrupted CAS objects
  - missing legacy files
- No warnings-only behavior and no best-effort fallback.

Explicit Non-Features (NOT IMPLEMENTED IN Z.2.2)
- No garbage collection (GC)
- No eviction
- No pinning
- No retention policies
- No provenance chaining

---

## Z.2.3 Immutable Run Artifact Invariants

Run Artifact Types
Z.2.3 defines immutable run artifacts stored via CAS:
- TASK_SPEC
- STATUS
- OUTPUT_HASHES

Immutability
- Each artifact is immutable once written.
- No in-place updates, no mutation semantics.
- Re-writing identical content yields the same hash (CAS identity).

Canonical Encoding
- Structured artifacts MUST use canonical, deterministic encoding.
- Ordering must be stable (especially OUTPUT_HASHES list).

TASK_SPEC
- Represents the exact input specification used for a run.
- Must be byte-identical to the canonical encoding of the provided spec.
- No implicit timestamps or environment-dependent fields.

STATUS
- Represents run state and outcome as a small structured record.
- No implicit timestamps or environment-dependent fields unless provided explicitly by caller.

OUTPUT_HASHES
- Deterministic ordered list of CAS hashes produced by a run.
- Ordering must be stable and reproducible.

Failure Mode
- Fail closed on invalid inputs, invalid hashes, missing objects, or corruption.
- No silent fallbacks.

---

## Z.2.4 Deduplication Invariants

Deduplication Guarantee
- Identical content MUST share storage.
- Identical content MUST NOT be rewritten on subsequent storage operations.

CAS Deduplication
- CAS implements deduplication via content addressing:
  - Same bytes → same SHA-256 hash
  - Same hash → same storage path
- Write-once semantics ensure no rewrites:
  - If object exists at computed path, cas_put returns hash without writing
  - Underlying file is NOT modified on duplicate puts

Artifact Store Deduplication
- Artifact store inherits CAS deduplication:
  - store_bytes(data) twice → same "sha256:<hash>" ref
  - store_file on identical files → same "sha256:<hash>" ref
- Deduplication is deterministic and automatic (no explicit dedup API needed)

Mechanical Proof
- Z.2.4 compliance is mechanically proven by tests:
  - `CAPABILITY/TESTBENCH/cas/test_cas_dedup.py`
    - Proves cas_put returns same hash for identical bytes
    - Proves underlying object is not rewritten (via file mtime verification)
  - `CAPABILITY/TESTBENCH/artifacts/test_artifact_dedup.py`
    - Proves store_bytes returns same ref for identical bytes
    - Proves store_file returns same ref for identical files

Explicit Non-Features (NOT IMPLEMENTED IN Z.2.4)
- No reference counting
- No garbage collection triggers
- No storage reclamation
- No dedup statistics or reporting

---

## Forbidden Before Explicit Roadmap Tasks

The following are NOT allowed to be introduced implicitly within Z.2.1–Z.2.4 scope and require a new explicit roadmap item:

Enforcement
- Mandatory hash-only mode (disallowing legacy paths)
- Runtime enforcement gates requiring CAS refs
- “Trusted mode” or bypass flags

Lifecycle Policy
- GC, pinning, eviction, retention windows
- Automatic pruning or cleanup policies

Provenance / Chaining
- Provenance graphs, DAG chaining, signing, attestation
- Cross-artifact linking beyond simple ref storage

Orchestration Coupling
- Integrating these primitives directly into runners, skills, MCP, or planning logic without an explicit roadmap step

---

## Commit Hygiene Guidance

What belongs in git:
- CAS implementation + tests
- Artifact store implementation + tests
- Run artifacts implementation + tests
- Governance guardrail tests
- Documentation/spec fences like this file

What does NOT belong in git:
- CAS runtime storage contents (e.g. CAPABILITY/CAS/storage/*)
- transient test caches
- generated scratch artifacts unless explicitly declared as canonical build outputs


---

## Recovery: CAS and Run Invariant Violations

### Where receipts live

CAS and Run-related receipts and audit trails are stored in:

- **CAPABILITY/CAS/storage/** - CAS object storage (content-addressed blobs)
  - Objects stored at deterministic paths: `storage/{prefix}/{hash}`
  - Each object is immutable and verified on write
- **CAPABILITY/RUNS/RUN_ROOTS.json** - Active run roots for GC protection
  - List of CAS hashes that are protected from garbage collection
  - Must be valid JSON array of 64-char lowercase hex hashes
- **LAW/CONTRACTS/_runs/** - Run execution records and receipts
  - `_runs/{run_id}/TASK_SPEC.json` - Immutable task specification
  - `_runs/{run_id}/STATUS.json` - Run status and outcome
  - `_runs/{run_id}/OUTPUT_HASHES.json` - Deterministic output hash list
  - `_runs/{run_id}/ERRORS.json` - Structured error records
- **LAW/CONTRACTS/_runs/audit_logs/** - Root audit results
  - `root_audit.jsonl` - Output from root_audit.py

### How to re-run verification

To verify CAS and Run invariant compliance:

```bash
# Verify CAS integrity (check all stored objects)
python -c "
from CAPABILITY.CAS import cas as cas_mod
from pathlib import Path
storage = cas_mod._CAS_ROOT / 'storage'
for obj in storage.rglob('*'):
    if obj.is_file():
        try:
            hash_hex = obj.name
            data = cas_mod.cas_get(hash_hex)
            print(f'✓ {hash_hex[:12]}...')
        except Exception as e:
            print(f'✗ {hash_hex[:12]}... CORRUPT: {e}')
"

# Verify run bundle integrity
python -c "
from CAPABILITY.RUNS.bundles import run_bundle_verify
bundle_ref = 'sha256:YOUR_BUNDLE_HASH_HERE'
receipt = run_bundle_verify(bundle_ref)
print(f'Status: {receipt.verification_status}')
print(f'Errors: {receipt.errors}')
"

# Run root audit (verifies reachability from roots)
python CAPABILITY/AUDIT/root_audit.py --verbose

# Verify RUN_ROOTS.json format
python -c "
import json
from pathlib import Path
roots = json.loads(Path('CAPABILITY/RUNS/RUN_ROOTS.json').read_text())
assert isinstance(roots, list), 'RUN_ROOTS must be a list'
for i, h in enumerate(roots):
    assert len(h) == 64 and h.islower() and h.isalnum(), f'Invalid hash at index {i}: {h}'
print(f'✓ RUN_ROOTS.json valid ({len(roots)} roots)')
"
```

### What to delete vs never delete

**Safe to delete (with caution):**
- **Unrooted CAS objects** - Only via GC with `allow_empty_roots=False` protection
  - Never manually delete from `CAPABILITY/CAS/storage/`
  - Use `python CAPABILITY/GC/gc.py --dry-run` to preview deletions
- **Temporary run artifacts** - Under `LAW/CONTRACTS/_runs/_tmp/`
  - These are catalytic domains and disposable by design

**Never delete (protected by invariants):**
- **Rooted CAS objects** - Any hash in `RUN_ROOTS.json` or `GC_PINS.json`
  - Deletion breaks referential integrity
  - Only GC can safely delete after root removal
- **RUN_ROOTS.json** and **GC_PINS.json** - Root tracking files
  - Modify only via explicit ceremony
  - Deletion causes catastrophic GC failure (fail-closed protection)
- **Run bundle manifests** - Immutable proof-carrying records
  - Referenced by bundle_ref in downstream systems
  - Deletion breaks audit trail
- **TASK_SPEC, STATUS, OUTPUT_HASHES** - Immutable run artifacts
  - These are the source of truth for run verification
  - Re-running may not produce identical hashes if inputs changed

**Recovery procedures:**
- **CAS object corrupted**: Delete corrupted object, re-store from source material
  ```bash
  # Identify corruption
  python -c "from CAPABILITY.CAS import cas as cas_mod; cas_mod.cas_get('HASH_HERE')"
  # If corrupt, delete and re-store
  rm CAPABILITY/CAS/storage/prefix/HASH_HERE
  python -c "from CAPABILITY.CAS import cas as cas_mod; print(cas_mod.cas_put(SOURCE_BYTES))"
  ```
- **RUN_ROOTS.json malformed**: Restore from git history or rebuild from known good roots
  ```bash
  git checkout HEAD~ -- CAPABILITY/RUNS/RUN_ROOTS.json
  ```
- **Run bundle verification failed**: Check if referenced blobs exist, restore missing blobs from backup
  ```bash
  python -c "
  from CAPABILITY.RUNS.bundles import run_bundle_verify
  receipt = run_bundle_verify('sha256:BUNDLE_HASH')
  print('Missing artifacts:', [k for k, v in receipt.artifact_status.items() if not v])
  "
  ```
- **Root audit failed**: Identify unreachable outputs, add missing roots to RUN_ROOTS.json
  ```bash
  python CAPABILITY/AUDIT/root_audit.py --verbose
  # Add missing root hash to RUN_ROOTS.json
  python -c "
  import json
  from pathlib import Path
  roots = json.loads(Path('CAPABILITY/RUNS/RUN_ROOTS.json').read_text())
  roots.append('MISSING_HASH_HERE')
  roots = sorted(list(set(roots)))
  Path('CAPABILITY/RUNS/RUN_ROOTS.json').write_text(json.dumps(roots, indent=2, sort_keys=True) + '\n')
  "
  ```

---
