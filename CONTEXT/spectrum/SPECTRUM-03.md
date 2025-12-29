# SPECTRUM-03: Chained Temporal Integrity

**Version:** 1.0.0  
**Status:** Draft  
**Created:** 2024-12-24

---

## 1. Purpose

SPECTRUM-03 defines the protocol for reasoning across a sequence of runs using ONLY SPECTRUM bundles as memory, with all other context deleted.

This establishes temporal integrity across time, not just per-run.

---

## 2. Definitions

### 2.1 Run Bundle

A SPECTRUM-02 bundle for a single run, consisting of:

| Artifact | Description |
|----------|-------------|
| `TASK_SPEC.json` | Immutable job specification |
| `STATUS.json` | Final execution status with `status` and `cmp01` fields |
| `OUTPUT_HASHES.json` | Output verification manifest with SHA-256 hashes |

### 2.2 Bundle Chain

An ordered list of run directories under `CONTRACTS/_runs/`.

Each run in the chain may reference outputs from:
- Itself
- Earlier runs in the chain

---

## 3. Chain Acceptance Rule

A bundle chain is **ACCEPTED** if and only if:

1. Every run bundle verifies via `verify_spectrum02_bundle`
2. Runs are ordered strictly by completion time or explicit chain order
3. Each run's `TASK_SPEC` references only durable outputs from:
   - Itself, or
   - Earlier runs in the chain

Otherwise, the chain **MUST** be rejected.

---

## 4. Chain Memory Model

### 4.1 Persisted Information (MEMORY)

Only the following information is allowed to persist across runs:

| Data | Type | Purpose |
|------|------|---------|
| `run_id` | String | Run identifier |
| Declared durable outputs | Paths | Output file locations |
| Output hashes | SHA-256 | Content verification |
| `validator_identity` | String | Semver + build_id |
| `status` | String | Success/failure state |

### 4.2 Non-Memory Artifacts (FORBIDDEN)

The following artifacts are explicitly NON-MEMORY and must not be required:

| Artifact Type | Reason |
|---------------|--------|
| `logs/` directory | Observability; not trust-critical |
| `tmp/` directory | Ephemeral workspace |
| Chat transcripts | UI artifact |
| Reasoning traces | Model internals; not verifiable |
| Intermediate state | Catalytic domain concern |

---

## 5. Chain Verification Procedure

```
VERIFY_CHAIN(run_dirs: List[Path]) -> Result:
    # Phase 1: Verify each bundle individually
    for run_dir in run_dirs:
        result = verify_spectrum02_bundle(run_dir)
        if not result.valid:
            return REJECT(run_dir.name, result.errors)
    
    # Phase 2: Build available output registry
    available_outputs = {}
    
    for run_dir in run_dirs (ordered):
        task_spec = load(run_dir / "TASK_SPEC.json")
        output_hashes = load(run_dir / "OUTPUT_HASHES.json")
        
        # Verify all input references exist in available_outputs
        for input_ref in task_spec.inputs:
            if input_ref not in available_outputs:
                return REJECT(run_dir.name, INVALID_CHAIN_REFERENCE)
        
        # Register this run's outputs
        for output_path in output_hashes.hashes:
            available_outputs[output_path] = run_dir.name
    
    # Phase 3: Verify no forbidden artifacts exist
    for run_dir in run_dirs:
        if (run_dir / "logs").exists():
            return REJECT(run_dir.name, FORBIDDEN_ARTIFACT)
        if (run_dir / "tmp").exists():
            return REJECT(run_dir.name, FORBIDDEN_ARTIFACT)
        if (run_dir / "transcript.json").exists():
            return REJECT(run_dir.name, FORBIDDEN_ARTIFACT)
    
    return ACCEPT
```

---

## 6. Failure Semantics

### 6.1 Fail Closed

Chain verification must fail closed. Any ambiguity or uncertainty results in rejection.

### 6.2 Error Format

All errors must follow the structured format:

```json
{
  "code": "ERROR_CODE",
  "message": "Human-readable description",
  "run_id": "affected-run-id",
  "path": "relative/path/if/applicable",
  "details": {}
}
```

### 6.3 Error Codes

| Code | Condition |
|------|-----------|
| `STATUS_NOT_SUCCESS` | `STATUS.status != "success"` |
| `CMP01_NOT_PASS` | `STATUS.cmp01 != "pass"` |
| `VALIDATOR_UNSUPPORTED` | `validator_semver` not in known set |
| `VALIDATOR_BUILD_ID_MISSING` | `validator_build_id` missing or empty |
| `OUTPUT_MISSING` | Declared output file does not exist |
| `HASH_MISMATCH` | Computed hash != declared hash |
| `BUNDLE_INCOMPLETE` | Required artifact missing from bundle |
| `INVALID_CHAIN_REFERENCE` | TASK_SPEC references output not produced by earlier run |
| `CHAIN_ORDER_VIOLATION` | Run timestamps not strictly ordered |

### 6.4 Chain Invalidation

Rejection of any run invalidates the entire chain. Partial acceptance is not allowed.

---

## 7. Security Properties

### 7.1 No History Dependency

Chain verification depends ONLY on:
- Bundle artifacts (TASK_SPEC, STATUS, OUTPUT_HASHES)
- Actual file hashes
- Chain ordering

It does NOT depend on:
- How runs were executed
- What tools were used
- Intermediate steps
- Agent reasoning

### 7.2 Tamper Evidence

Any modification to outputs after bundle generation is detected via hash mismatch.

### 7.3 Reference Integrity

A run cannot claim to depend on outputs that do not exist in the chain history.

### 7.4 Temporal Ordering

Runs must maintain strict temporal ordering. A later run cannot reference outputs from a future run.

---

## 8. Test Requirements

Tests MUST verify:

1. Chain accepts all verified bundles
2. Chain rejects on middle-run tamper (HASH_MISMATCH)
3. Chain rejects on invalid reference (INVALID_CHAIN_REFERENCE)
4. Chain rejects on missing bundle artifact (BUNDLE_INCOMPLETE)
5. Chain accepts without logs/tmp/transcripts (no history dependency)

Tests MUST NOT:
- Mock trust decisions
- Use heuristics
- Assume good faith

---

## 9. References

- SPECTRUM-01: Minimal Durable Execution Artifact
- SPECTRUM-02: Adversarial Resume Without Execution History
- CMP-01: Catalytic Minimal Primitive
