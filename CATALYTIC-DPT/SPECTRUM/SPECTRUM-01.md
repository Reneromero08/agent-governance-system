# SPECTRUM-01: Minimal Durable Execution Artifact

**Version:** 1.0.0  
**Status:** Draft  
**Created:** 2024-12-24

---

## 1. Artifact Set Definition

SPECTRUM-01 defines the minimum artifact bundle required to resume trust and reasoning for a completed execution run. No other artifacts are required.

### 1.1 Required Artifacts

| Artifact | Type | Description |
|----------|------|-------------|
| `TASK_SPEC.json` | JSON | Immutable job specification. Defines inputs, expected outputs, and constraints. |
| `STATUS.json` | JSON | Final execution status. Contains `status` field (`success` \| `failure`), timestamp, and error details if applicable. |
| `OUTPUT_HASHES.json` | JSON | Output verification manifest. Maps each output path to its SHA-256 hash. |
| `validator_identity` | String/Object | Validator provenance. Includes semantic version (`validator_semver`) and MAY include a deterministic build fingerprint (`validator_build_id`). |

### 1.2 Artifact Schemas

#### TASK_SPEC.json
```json
{
  "task_id": "string",
  "inputs": ["path/to/input1", "path/to/input2"],
  "expected_outputs": ["path/to/output1"],
  "constraints": {},
  "created_at": "ISO8601"
}
```

#### STATUS.json
```json
{
  "status": "success | failure",
  "completed_at": "ISO8601",
  "error": null | { "code": "string", "message": "string" }
}
```

#### OUTPUT_HASHES.json
```json
{
  "validator_semver": "1.10.0",
  "validator_build_id": "git:abc1234",
  "generated_at": "ISO8601",
  "hashes": {
    "path/to/output1": "sha256:abc123...",
    "path/to/output2": "sha256:def456..."
  }
}
```

---

## 2. Excluded Artifacts

The following artifacts are explicitly **NOT REQUIRED** for trust resumption:

| Artifact Type | Reason for Exclusion |
|---------------|---------------------|
| Execution logs | Observability concern, not trust-critical |
| Intermediate state | Ephemeral by design; catalytic domain handles this |
| Catalytic domains | Restored/destroyed per CMP-01; not persisted |
| Chat history | UI/UX artifact; no bearing on output validity |
| Reasoning traces | Model internals; not verifiable post-hoc |
| Temporary files | Transient; excluded from output manifest |
| Debug dumps | Development artifact; not production-relevant |

---

## 3. Trust Rule

```
IF STATUS.status == "success"
   AND all paths in OUTPUT_HASHES.json exist
   AND sha256(file) == OUTPUT_HASHES.json.hashes[path] for each path
   AND validator_identity is verified
THEN
   the run is accepted as true.
ELSE
   the run is rejected.
```

### 3.1 Trust Verification Procedure

1. Parse `STATUS.json`. If `status != "success"`, reject.
2. Parse `OUTPUT_HASHES.json`. Extract `validator_semver` and `validator_build_id` (if present).
3. Validator identity clarification:
   - Validator identity MAY include both a semantic version (semver) and a deterministic build fingerprint (e.g., commit hash or source hash).
   - When both are present, verification MUST enforce both.
4. Validate `validator_identity` is supported/matched. If mismatch or unsupported, reject.
5. For each entry in `hashes`:
   - Verify file exists at path.
   - Compute SHA-256 of file contents.
   - Compare to stored hash. If mismatch, reject.
6. If all checks pass, accept.

---

## 4. Worked Example

### 4.1 Before: Full Execution Context

```
run_20241224_091500/
├── TASK_SPEC.json                    # 1 KB
├── STATUS.json                       # 0.2 KB
├── outputs/
│   └── report.pdf                    # 50 KB
├── logs/
│   ├── agent.log                     # 2 MB
│   ├── stderr.log                    # 500 KB
│   └── debug.trace                   # 10 MB
├── catalytic/
│   ├── tmp_workspace/                # 200 MB
│   ├── intermediate_state.json       # 5 MB
│   └── scratch/                      # 50 MB
├── chat_history.json                 # 100 KB
├── reasoning_trace.json              # 1 MB
└── checkpoints/
    ├── step_001.bin                  # 20 MB
    ├── step_002.bin                  # 20 MB
    └── step_003.bin                  # 20 MB

Total: ~328 MB
```

### 4.2 After: SPECTRUM-01 Bundle

```
run_20241224_091500/
├── TASK_SPEC.json                    # 1 KB
├── STATUS.json                       # 0.2 KB
├── outputs/
│   └── report.pdf                    # 50 KB
└── OUTPUT_HASHES.json                 # 0.3 KB

Total: ~51.5 KB
```

### 4.3 Reduction

| Metric | Before | After | Reduction |
|--------|--------|-------|-----------|
| Total Size | 328 MB | 51.5 KB | 99.98% |
| File Count | 14+ | 4 | 71% |
| Trust-Critical | 4 | 4 | 0% (preserved) |

---

## 5. Implementation Notes

1. **Hash Algorithm:** SHA-256 only. No alternatives.
2. **Path Format:** Relative to run directory. Forward slashes. No normalization.
3. **Validator Versioning:** SemVer. Breaking changes require major version bump.
4. **Atomicity:** `STATUS.json` must be written last. Presence of `STATUS.json` with `success` indicates complete bundle.
5. **Immutability:** Once written, SPECTRUM-01 artifacts must not be modified.

---

## 6. References

- CMP-01: Catalytic Minimal Primitive
- Validator: `MCP/server.py` (current implementation)
