# CMP-01: Catalytic Mutation Protocol

This document defines the operational protocol for catalytic execution in the Agent Governance System. It governs any agent run that temporarily mutates filesystem state while guaranteeing restoration.

## One-Line Rule

An agent MAY mutate only inside declared catalytic domains and MUST restore them byte-identical at end-of-run, while writing durable outputs only to allowed output roots.

## Scope

CMP-01 governs any agent run that needs temporary mutation of filesystem state for:
- Indexing and compilation passes (CORTEX indexes, section indexes, registries)
- Pack generation and verification (LITE/FULL/TEST)
- Large refactors that need intermediate transforms
- Migration scripts that must be reversible until finalized

CMP-01 is NOT required for pure read-only runs.

## Terms

| Term | Definition |
|------|------------|
| **Clean Workspace** | The small, authoritative state that MUST remain stable (Canon, contracts, skills, hand-authored docs) |
| **Catalyst** | A large writable area that MAY start in any state and is allowed to be temporarily mutated, provided it is restored exactly |
| **Catalytic Domain** | The set of paths permitted to be mutated under CMP-01 |
| **Durable Output Root** | The only locations where artifacts MAY remain after the run |
| **Restoration Proof** | A machine-verifiable record that the catalytic domain was restored exactly |
| **Run Ledger** | The per-run audit bundle storing restoration proof and run metadata |

## Invariants

These are hard requirements. Violation of any invariant makes a run invalid.

1. **INV-CMP-01**: No silent writes outside catalytic domains
2. **INV-CMP-02**: No durable artifacts outside allowed output roots
3. **INV-CMP-03**: Catalytic domains MUST be byte-identical after the run
4. **INV-CMP-04**: Restoration proof is mandatory; if proof fails, the run is invalid
5. **INV-CMP-05**: All mutations MUST be attributable to a run ID with a run ledger
6. **INV-CMP-06**: If non-determinism is unavoidable, it MUST be declared in the run ledger

## Canonical Path Constants

These constants are enforced by `CAPABILITY/MCP/server.py`, `CAPABILITY/TOOLS/agents/skill_runtime.py`, and `CAPABILITY/TOOLS/catalytic/catalytic_runtime.py`.

### Durable Output Roots

Files MAY persist here after a catalytic run:

```
LAW/CONTRACTS/_runs/
NAVIGATION/CORTEX/_generated/
MEMORY/LLM_PACKER/_packs/
```

### Catalytic Roots (Scratch Domains)

Temporary mutation is allowed here; MUST be restored byte-identical:

```
LAW/CONTRACTS/_runs/_tmp/
NAVIGATION/CORTEX/_generated/_tmp/
MEMORY/LLM_PACKER/_packs/_tmp/
CAPABILITY/PRIMITIVES/_scratch/
THOUGHT/LAB/_tmp/
```

### Forbidden Roots

These paths MUST NEVER be written to or overlapped by catalytic domains:

```
LAW/CANON/
AGENTS.md
BUILD/
.git
```

## The Catalytic Lifecycle

A catalytic run has six phases (0-5).

### Phase 0: Declare

The run MUST declare before any execution:

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | string | Unique identifier for this run |
| `job_id` | string | Stable job identifier (for memoization) |
| `catalytic_domains[]` | string[] | Paths to be mutated (MUST be under CATALYTIC_ROOTS) |
| `durable_output_roots[]` | string[] | Paths where outputs persist (MUST be under DURABLE_ROOTS) |
| `intent` | string | One-sentence description |
| `determinism` | enum | `deterministic`, `bounded_nondeterministic`, or `nondeterministic` |

Schema: `LAW/SCHEMAS/jobspec.schema.json`

### Phase 1: Snapshot

Before any writes:
- Compute a content-hash manifest of each catalytic domain
- Record file paths, sizes, and SHA-256 hashes
- Store snapshots in the run ledger as `PRE_MANIFEST`

### Phase 1.5: Memoization (Optional)

If memoization is enabled:
- Compute cache key from JobSpec + input domain roots + validator ID
- On cache HIT: restore outputs from cache, skip execution
- On cache MISS: proceed to Phase 2

### Phase 2: Execute (Mutate)

- Execute the wrapped command or pipeline
- Mutations are allowed ONLY inside declared catalytic domains
- Any write outside catalytic domains is a hard violation

### Phase 3: Commit Durable Outputs

- Write durable outputs ONLY inside durable output roots
- Durable outputs MUST be listed in the run ledger as `OUTPUTS`
- Hash all outputs for `OUTPUT_HASHES.json`

### Phase 4: Restore

- Restore catalytic domains to their original state
- Restoration is NOT "best effort"—it is exact
- Compute post-snapshot as `POST_MANIFEST`

### Phase 5: Prove

Generate and validate restoration proof:

1. Compare `PRE_MANIFEST` to `POST_MANIFEST`
2. Compute `RESTORE_DIFF` (MUST be empty for success)
3. Generate `PROOF.json` with verification result
4. If validation fails:
   - Mark run as `failed`
   - Attempt rollback again
   - If still failing, quarantine outputs and halt

## Canonical Artifact Set

Every successful catalytic run produces these artifacts in `LAW/CONTRACTS/_runs/<run_id>/`:

| Artifact | Description | Schema |
|----------|-------------|--------|
| `JOBSPEC.json` | Phase 0 declaration | `jobspec.schema.json` |
| `STATUS.json` | Run state machine | See below |
| `INPUT_HASHES.json` | Hashes of inputs from pre-snapshot | - |
| `OUTPUT_HASHES.json` | Hashes of durable outputs | - |
| `DOMAIN_ROOTS.json` | Merkle roots per catalytic domain | - |
| `LEDGER.jsonl` | Append-only receipt log | `ledger.schema.json` |
| `VALIDATOR_ID.json` | Validator version info | - |
| `PROOF.json` | Restoration proof (written LAST) | `proof.schema.json` |

### Legacy Artifacts (Backwards Compatibility)

| Artifact | Description |
|----------|-------------|
| `RUN_INFO.json` | Run metadata |
| `PRE_MANIFEST.json` | Pre-snapshot |
| `POST_MANIFEST.json` | Post-snapshot |
| `RESTORE_DIFF.json` | Diff (MUST be empty) |
| `OUTPUTS.json` | Durable output list |

## Run Ledger Schema

Schema: `LAW/SCHEMAS/ledger.schema.json`

Required fields:
- `RUN_INFO`: run_id, timestamp, intent, catalytic_domains, exit_code, restoration_verified
- `PRE_MANIFEST`: domain → {path → sha256}
- `POST_MANIFEST`: domain → {path → sha256}
- `RESTORE_DIFF`: domain → {added, removed, changed}
- `OUTPUTS`: [{path, type, sha256?}]
- `STATUS`: {status, restoration_verified, exit_code, validation_passed}

## Proof-Gated Acceptance

**Acceptance is STRICTLY proof-driven.**

A run is accepted if and only if `PROOF.json.restoration_result.verified == true`.

No heuristics. No logs. No RESTORE_DIFF inspection as fallback.

```python
# From CAPABILITY/TOOLS/catalytic/catalytic_validator.py
if restoration_result["verified"] is not True:
    return False  # HARD GATE
```

## Enforcement Layers

CMP-01 is enforced at three layers:

### Layer 1: Preflight Validation (Pre-execution)

Before execution, the JobSpec is validated:
- All `catalytic_domains` MUST be under `CATALYTIC_ROOTS`
- All `outputs.durable_paths` MUST be under `DURABLE_ROOTS`
- No path MAY overlap `FORBIDDEN_ROOTS`
- No path traversal (`..`) allowed
- No containment overlap within same list

Implementation: `CAPABILITY/PRIMITIVES/preflight.py`, `CAPABILITY/MCP/server.py:_validate_jobspec_paths()`

### Layer 2: Runtime Guard (During execution)

A filesystem guard tracks writes during the run:
- All writes are checked against allowed roots
- Writes to forbidden paths are blocked
- Out-of-domain writes fail immediately

Implementation: `CAPABILITY/PRIMITIVES/fs_guard.py`

### Layer 3: Post-Run Validation (CI Gate)

After execution, the run ledger is validated:
- All canonical artifacts MUST exist
- `PROOF.json.restoration_result.verified` MUST be `true`
- All outputs MUST be under durable roots
- All outputs MUST actually exist

Implementation: `CAPABILITY/TOOLS/catalytic/catalytic_validator.py`

## Failure Handling

- **HARD FAIL** on any write outside catalytic domains
- **HARD FAIL** on restoration proof mismatch
- **HARD FAIL** if canonical artifact writing fails
- If restore fails:
  - Quarantine outputs under `LAW/CONTRACTS/_runs/<run_id>/quarantine/`
  - Halt and require human arbitration

## Integration Points

### Cortex Indexing

```
catalytic_domains: ["NAVIGATION/CORTEX/_generated/_tmp/"]
durable_outputs: ["NAVIGATION/CORTEX/_generated/cortex.json"]
```

### LLM Packer

```
catalytic_domains: ["MEMORY/LLM_PACKER/_packs/_tmp/"]
durable_outputs: ["MEMORY/LLM_PACKER/_packs/<pack_id>/"]
```

### Skill Execution

Any skill that mutates the filesystem MUST declare one of:
- `read_only: true` — No CMP-01 required
- `catalytic: true` — Full CMP-01 enforcement
- `writer: true` — Non-catalytic writer (discouraged, rare)

Implementation: `CAPABILITY/TOOLS/agents/skill_runtime.py` (Z.1.6)

## Threat Model Coverage

CMP-01 protects against:
- Accidental repo pollution
- Incremental drift from repeated agent runs
- "Helpful" agents writing caches in random folders
- Partial runs leaving intermediate files behind

CMP-01 does NOT protect against:
- Malicious code that bypasses guards at the OS level
- External side effects (network calls, remote writes) unless separately sandboxed

## Implementation Files

| File | Purpose |
|------|---------|
| `CAPABILITY/TOOLS/catalytic/catalytic_runtime.py` | Six-phase lifecycle executor |
| `CAPABILITY/TOOLS/catalytic/catalytic_validator.py` | Run ledger validator |
| `CAPABILITY/TOOLS/agents/skill_runtime.py` | CMP-01 pre-validation for skills |
| `CAPABILITY/MCP/server.py` | CMP-01 path validation |
| `CAPABILITY/PRIMITIVES/preflight.py` | JobSpec preflight validation |
| `CAPABILITY/PRIMITIVES/fs_guard.py` | Runtime filesystem guard |
| `CAPABILITY/PRIMITIVES/restore_proof.py` | Restoration proof generator |

## Schemas

| Schema | Purpose |
|--------|---------|
| `LAW/SCHEMAS/jobspec.schema.json` | Phase 0 declaration |
| `LAW/SCHEMAS/ledger.schema.json` | Run ledger structure |
| `LAW/SCHEMAS/proof.schema.json` | Restoration proof |

## Tests

| Test File | Coverage |
|-----------|----------|
| `CAPABILITY/TESTBENCH/core/test_cmp01_validator.py` | Path validation (15 tests) |
| `CAPABILITY/TESTBENCH/core/test_skill_runtime_cmp01.py` | Skill execution enforcement (15 tests) |
| `CAPABILITY/TESTBENCH/integration/test_task_4_1_catalytic_snapshot_restore.py` | Snapshot/restore |
| `CAPABILITY/TESTBENCH/adversarial/test_adversarial_proof_tamper.py` | Proof tamper detection |
| `CAPABILITY/TESTBENCH/pipeline/test_runtime_guard.py` | Runtime guard |

## References

### Theory and Protocol
- [CATALYTIC_COMPUTING.md](CATALYTIC_COMPUTING.md) — Catalytic computing theory
- `LAW/CANON/INTEGRITY.md` — Integrity invariants
- `LAW/CONTEXT/decisions/ADR-018-catalytic-computing-canonical-note.md` — Formal definitions
- `NAVIGATION/maps/CATALYTIC_DOMAINS.md` — Domain inventory

### SPECTRUM Cryptographic Specifications
- [SPECTRUM-02_RESUME_BUNDLE.md](SPECTRUM-02_RESUME_BUNDLE.md) — Adversarial resume without execution history
- [SPECTRUM-03_CHAIN_VERIFICATION.md](SPECTRUM-03_CHAIN_VERIFICATION.md) — Chained temporal integrity
- [SPECTRUM-04_IDENTITY_SIGNING.md](SPECTRUM-04_IDENTITY_SIGNING.md) — Validator identity and Ed25519 signing
- [SPECTRUM-05_VERIFICATION_LAW.md](SPECTRUM-05_VERIFICATION_LAW.md) — 10-phase verification procedure
- [SPECTRUM-06_RESTORE_RUNNER.md](SPECTRUM-06_RESTORE_RUNNER.md) — Restore semantics with atomicity

---

*This document is canonical. Implementation MUST match this specification.*
