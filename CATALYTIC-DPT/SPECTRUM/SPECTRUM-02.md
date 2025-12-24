# SPECTRUM-02: Adversarial Resume Without Execution History

**Version:** 1.0.0
**Status:** Draft
**Created:** 2024-12-24

---

## 1. Purpose

SPECTRUM-02 defines the protocol for an agent to resume trust and continue work using ONLY a compressed durable bundle, when all prior execution context has been destroyed.

This is an adversarial scenario. The resuming agent has no memory of the prior run.

---

## 2. SPECTRUM-02 Resume Bundle

### 2.1 Required Artifacts

| Artifact | Type | Description |
|----------|------|-------------|
| `TASK_SPEC.json` | JSON | Immutable job specification |
| `STATUS.json` | JSON | Final execution status with `status` and `cmp01` fields |
| `OUTPUT_HASHES.json` | JSON | Output verification manifest with SHA-256 hashes |
| `VALIDATOR_VERSION` | String | Embedded in `OUTPUT_HASHES.json` |

### 2.2 Artifact Schemas

#### TASK_SPEC.json
```json
{
  "task_id": "string",
  "inputs": ["path/to/input1"],
  "expected_outputs": ["path/to/output1"],
  "constraints": {},
  "created_at": "ISO8601"
}
```

#### STATUS.json
```json
{
  "status": "success | failure | error",
  "cmp01": "pass | fail",
  "completed_at": "ISO8601",
  "error": null | { "code": "string", "message": "string" }
}
```

#### OUTPUT_HASHES.json
```json
{
  "validator_semver": "1.0.0",
  "validator_build_id": "git:abc1234 | file:sha256prefix",
  "generated_at": "ISO8601",
  "hashes": {
    "path/to/output1": "sha256:abc123..."
  }
}
```

**Fields:**
- `validator_semver`: Semantic version of the validator (must be in SUPPORTED_VALIDATOR_SEMVERS)
- `validator_build_id`: Deterministic build fingerprint (git commit SHA or file hash of validator code)
- `generated_at`: ISO8601 timestamp of bundle generation
- `hashes`: Map of posix-style relative paths to SHA-256 hashes

### 2.3 Explicitly Forbidden Artifacts

The following artifacts MUST NOT be present in a SPECTRUM-02 bundle:

| Artifact Type | Reason |
|---------------|--------|
| `logs/` directory | Execution observability; not trust-critical |
| `tmp/` directory | Ephemeral workspace; destroyed after run |
| Chat transcripts | UI artifact; no bearing on output validity |
| Reasoning traces | Model internals; not verifiable |
| Restoration snapshots | Intermediate state; catalytic domain concern |
| Execution order metadata | Process information; not output-relevant |
| Debug dumps | Development artifact |
| Checkpoint files | Intermediate state |

---

## 3. Resume Rule (Normative)

```
An agent MAY proceed as if the run is true IFF:
  - STATUS.status == "success"
  - STATUS.cmp01 == "pass"
  - OUTPUT_HASHES verify under VALIDATOR_VERSION

Otherwise the agent MUST reject the run.
```

### 3.1 Verification Procedure

1. Parse `STATUS.json`
   - If `status != "success"`, REJECT
   - If `cmp01 != "pass"`, REJECT

2. Parse `OUTPUT_HASHES.json`
   - Extract `validator_semver`
   - If `validator_semver` is unsupported, REJECT
   - Extract `validator_build_id`
   - If `validator_build_id` is missing or empty, REJECT

3. For each entry in `hashes`:
   - Verify file exists at declared path
   - Compute SHA-256 of file contents
   - If hash mismatch, REJECT

4. If all checks pass, ACCEPT

---

## 4. Agent Obligations on Resume

### 4.1 MUST

- Verify all bundle artifacts before proceeding
- Treat durable outputs as ground truth
- Fail closed if any verification fails
- Log rejection reason if rejecting

### 4.2 MUST NOT

- Infer intent from missing history
- Hallucinate process or reasoning
- Assume prior context exists
- Trust bundle without verification
- Proceed on partial verification

---

## 5. Rejection Codes

| Code | Condition |
|------|-----------|
| `STATUS_NOT_SUCCESS` | `STATUS.status != "success"` |
| `CMP01_NOT_PASS` | `STATUS.cmp01 != "pass"` |
| `VALIDATOR_UNSUPPORTED` | `validator_semver` not in known set |
| `VALIDATOR_BUILD_ID_MISSING` | `validator_build_id` missing or empty |
| `VALIDATOR_BUILD_MISMATCH` | `validator_build_id` != current (strict mode only) |
| `OUTPUT_MISSING` | Declared output file does not exist |
| `HASH_MISMATCH` | Computed hash != declared hash |
| `BUNDLE_INCOMPLETE` | Required artifact missing from bundle |

---

## 6. Security Properties

### 6.1 No History Dependency

The resume decision depends ONLY on:
- STATUS.json content
- OUTPUT_HASHES.json content
- Actual file hashes

It does NOT depend on:
- How the run was executed
- What tools were used
- What intermediate steps occurred
- What the agent was "thinking"

### 6.2 Tamper Evidence

Any modification to outputs after STATUS.json was written will be detected via hash mismatch.

### 6.3 Validator Binding

Hashes are bound to a specific validator version. Version mismatch forces re-validation.

---

## 7. Test Requirements

Tests MUST verify:

1. Valid bundle acceptance without history
2. Hash mismatch rejection
3. Missing output rejection
4. Validator version mismatch rejection
5. Independence from logs/tmp/transcripts

Tests MUST NOT:
- Mock trust decisions
- Use heuristics
- Assume good faith

---

## 8. References

- SPECTRUM-01: Minimal Durable Execution Artifact
- CMP-01: Catalytic Minimal Primitive
