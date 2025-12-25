# Changelog

All notable changes to the CAT-DPT (Catalytic Distributed Processing Trust) system are documented in this file.

## [0.1.0] - 2025-12-25

### Phase 0 Complete: Contract Finalization and Enforcement

Phase 0 establishes the immutable contract that governs all future CAT-DPT work. All schemas, artifact specifications, and enforcement mechanisms are now finalized and frozen.

#### Schemas Finalized
- **jobspec.schema.json**: Canonical job specification format (Phase, TaskType, Intent, Inputs, Outputs, CatalyticDomains, Determinism)
- **ledger.schema.json**: Immutable ledger recording all job executions and artifacts
- **proof.schema.json**: Cryptographic proof of integrity (hash chains, restoration proofs, canonical artifact manifests)
- **validation_error.schema.json**: Deterministic error reporting with stable codes and paths
- **commonsense_entry.schema.json**: Knowledge base entries for commonsense reasoning
- **resolution_result.schema.json**: Results of symbolic resolution

#### Canonical Artifact Set (8 Required Files)
All runtime outputs must conform to the canonical artifact specification:
1. **JOBSPEC.json** - Job specification (immutable, validated at preflight)
2. **STATUS.json** - Run status (failed/completed)
3. **INPUT_HASHES.json** - SHA256 hashes of all inputs
4. **OUTPUT_HASHES.json** - SHA256 hashes of all outputs
5. **DOMAIN_ROOTS.json** - Catalytic domain state (required restoration targets)
6. **LEDGER.jsonl** - Immutable ledger of execution events
7. **VALIDATOR_ID.json** - Identity of accepting validator
8. **PROOF.json** - Cryptographic proof of execution integrity

#### 3-Layer Fail-Closed Enforcement

**Layer 1: Preflight Validation (Before Execution)**
- Validates JobSpec schema compliance
- Detects path traversal, absolute paths, forbidden paths
- Rejects overlapping input/output domains
- Returns deterministic validation_error objects with stable codes
- Implementation: `PRIMITIVES/preflight.py` (18/18 tests pass)

**Layer 2: Runtime Write Guard (During Execution)**
- Enforces allowed roots at write-time for all file operations
- Detects and rejects writes to forbidden paths (CANON, AGENTS.md, BUILD, .git)
- Blocks path traversal and absolute path escapes
- Fails closed immediately with RuntimeError on any violation
- Wraps all writes through FilesystemGuard class
- Implementation: `PRIMITIVES/fs_guard.py` (13/13 tests pass)

**Layer 3: CI Validation (After Execution)**
- Verifies all required canonical artifacts present
- Validates artifact schema compliance
- Checks hash integrity of outputs
- Verifies restoration proofs for catalytic domains
- Ensures ledger consistency
- Implementation: CI pipeline validation

#### Test Suite (40+ Total Tests)

- **test_preflight.py** (18 tests) - JobSpec validation, path safety, domain overlap detection
- **test_runtime_guard.py** (13 tests) - Write enforcement, forbidden paths, path escapes
- **test_commonsense_schema.py** (3 tests) - Knowledge base schema validation
- **test_resolver.py** (3 tests) - Symbolic resolution
- **test_symbols.py** (3 tests) - Symbol tables

All Phase 0 tests pass (37/37 Phase 0 tests) ✅

#### Exit Criteria Met

- [x] All schemas finalized and frozen (no breaking changes allowed)
- [x] Canonical artifact set fully specified
- [x] Layer 1 (Preflight) enforces contract before execution
- [x] Layer 2 (Runtime Guard) enforces contract during execution
- [x] Layer 3 (CI) enforces contract after execution
- [x] All enforcement layers fail-closed (RuntimeError/validation_error on violation)
- [x] Error codes deterministic and stable (JOBSPEC_*, WRITE_GUARD_*, etc.)
- [x] Comprehensive test coverage with all tests passing
- [x] Roadmap reflects completed work accurately

#### Key Implementation Files

- `CATALYTIC-DPT/SCHEMAS/` - Schema definitions
- `CATALYTIC-DPT/PRIMITIVES/preflight.py` - Preflight validator
- `CATALYTIC-DPT/PRIMITIVES/fs_guard.py` - Runtime write guard
- `TOOLS/catalytic_runtime.py` - Runtime integration
- `CATALYTIC-DPT/TESTBENCH/` - Test suite
- `CATALYTIC-DPT/ROADMAP_V2.1.md` - Updated roadmap

---

## Versioning

This project follows [Semantic Versioning](https://semver.org/). Phase 0 is marked as v0.1.0 because:
- Phase 0 establishes the immutable contract (major version 0 = pre-1.0)
- Schema versions are frozen and cannot break in minor versions
- All future work must comply with Phase 0 schemas
