# CATALYTIC-DPT Schemas (Phase 0)

**Phase**: 0 - Freeze the Contract
**Status**: Canonical definitions for Phase 1-7
**Purpose**: Three JSON schemas that define catalytic computing contracts

---

## The Three Schemas

### 1. JobSpec Schema (`jobspec.schema.json`)

**Definition**: Canonical specification for a catalytic computing task.

**Usage**: A 200M model or agent receives a JobSpec and knows exactly what to execute.

**Required Fields**:
- `job_id`: Unique identifier (kebab-case)
- `phase`: 0 (contract definition) or 1+ (execution)
- `task_type`: one of `schema_definition`, `primitive_implementation`, `validation`, `test_execution`
- `intent`: One-sentence human description
- `inputs`: Task-specific input parameters
- `outputs`: Expected durable outputs with validation criteria
- `catalytic_domains`: Scratch spaces that must be restored
- `determinism`: Level of determinism required (`deterministic`, `bounded_nondeterministic`, `nondeterministic`)

**Optional Fields**:
- `swarm_parallel`: Route to parallel execution (default: false)
- `metadata`: Priority, timeout, governance checks

**Example**:
```json
{
  "job_id": "phase1-catalytic-store",
  "phase": 1,
  "task_type": "primitive_implementation",
  "intent": "Implement content-addressable storage for catalytic kernel",
  "inputs": {"storage_path": "CATALYTIC-DPT/TESTBENCH/_store"},
  "outputs": {
    "durable_paths": ["TOOLS/catalytic_store.py"],
    "validation_criteria": {"all_tests_pass": true}
  },
  "catalytic_domains": ["CATALYTIC-DPT/TESTBENCH/_tmp"],
  "determinism": "deterministic",
  "swarm_parallel": false,
  "metadata": {"timeout_seconds": 300, "priority": 8}
}
```

---

### 2. Validation Error Vector Schema (`validation_error.schema.json`)

**Definition**: Canonical format for deterministic error reporting.

**Usage**: When a JobSpec fails validation, the error report follows this schema exactly. No ambiguous error messages.

**Required Fields**:
- `valid`: boolean (true if all checks pass)
- `errors`: array of error objects
- `warnings`: array of warning objects
- `timestamp`: ISO 8601 datetime
- `validator_version`: Version of validator that produced this report

**Error Object Format**:
```json
{
  "code": "SCHEMA_INVALID",  // UPPERCASE_SNAKE_CASE
  "message": "Field 'phase' is required",
  "path": "$.phase",  // JSONPath to the error location
  "details": {}  // Additional context
}
```

**Example Complete Report**:
```json
{
  "valid": false,
  "errors": [
    {
      "code": "MISSING_REQUIRED_FIELD",
      "message": "Field 'phase' is required but missing",
      "path": "$.phase",
      "details": {"required": ["phase"], "provided": ["job_id", "task_type"]}
    },
    {
      "code": "INVALID_ENUM_VALUE",
      "message": "Field 'determinism' has invalid value 'INVALID_VALUE'",
      "path": "$.determinism",
      "details": {"valid_values": ["deterministic", "bounded_nondeterministic", "nondeterministic"]}
    }
  ],
  "warnings": [
    {
      "code": "TIMEOUT_UNUSUALLY_HIGH",
      "message": "timeout_seconds is 10000, which is very high"
    }
  ],
  "timestamp": "2025-12-23T14:30:22.123456Z",
  "validator_version": "1.0.0"
}
```

**Error Codes** (deterministic, for parsing by models):
- `SCHEMA_INVALID`: The JSON itself is malformed
- `MISSING_REQUIRED_FIELD`: Required field not present
- `INVALID_FIELD_TYPE`: Field has wrong type (string instead of int, etc.)
- `INVALID_ENUM_VALUE`: Field value not in allowed set
- `INVALID_FORMAT`: Field fails format validation (e.g., regex)
- `INVALID_PATH`: Path does not exist or is not allowed
- `VALIDATION_LOGIC_ERROR`: Custom logic check failed
- `OUTPUT_DOES_NOT_EXIST`: Expected output file/dir not found
- `RESTORATION_FAILED`: Catalytic domain was not restored

---

### 3. Ledger Schema (`ledger.schema.json`)

**Definition**: Canonical format for run audit trails and restoration proofs.

**Usage**: After every catalytic execution, a ledger is generated in `CONTRACTS/_runs/<run_id>/`. This schema defines the structure.

**Core Files** (each becomes a JSON field):
- `RUN_INFO`: Metadata about the execution (run_id, timestamp, intent, catalytic_domains, exit_code)
- `PRE_MANIFEST`: Hash snapshot of catalytic domains before execution
- `POST_MANIFEST`: Hash snapshot of catalytic domains after execution
- `RESTORE_DIFF`: Diff between pre and post (must be empty for successful restore)
- `OUTPUTS`: List of durable outputs that were created/modified
- `STATUS`: Overall result (restored, dirty, error)
- `DECISION_LOG`: JSONL of timestamped decisions (OPTIONAL but recommended)

**Example RUN_INFO**:
```json
{
  "run_id": "phase1-catalytic-store-20251223-143022",
  "timestamp": "2025-12-23T14:30:22.123456Z",
  "intent": "Implement content-addressable storage",
  "catalytic_domains": ["CATALYTIC-DPT/TESTBENCH/_tmp"],
  "durable_output_roots": ["TOOLS/catalytic_store.py"],
  "exit_code": 0,
  "restoration_verified": true
}
```

**Example RESTORE_DIFF** (SUCCESS - empty):
```json
{
  "CATALYTIC-DPT/TESTBENCH/_tmp": {
    "added": {},
    "removed": {},
    "changed": {}
  }
}
```

**Example RESTORE_DIFF** (FAILURE - has changes):
```json
{
  "CATALYTIC-DPT/TESTBENCH/_tmp": {
    "added": {
      "leftover_file.txt": "abc123def456..."
    },
    "removed": {},
    "changed": {}
  }
}
```

**Example OUTPUTS**:
```json
[
  {
    "path": "TOOLS/catalytic_store.py",
    "type": "file",
    "sha256": "abc123def456abc123def456abc123def456abc123def456abc123def456abcd"
  },
  {
    "path": "CATALYTIC-DPT/TESTBENCH/_runs/phase1-catalytic-store-20251223-143022/",
    "type": "directory"
  }
]
```

**Example STATUS**:
```json
{
  "status": "restored",
  "restoration_verified": true,
  "exit_code": 0,
  "validation_passed": true
}
```

---

## Validation Rules (Executable)

### JobSpec Validation

1. **Required fields present**: job_id, phase, task_type, intent, outputs, catalytic_domains
2. **Field types correct**:
   - `job_id`: string, matches regex `^[a-z0-9-]+$`
   - `phase`: integer, value 0 or 1
   - `task_type`: string, one of enum
   - `catalytic_domains`: array of strings (directory paths)
   - `outputs.durable_paths`: array of strings under allowed roots
3. **Phase-specific rules**:
   - Phase 0: `task_type` must be `schema_definition`
   - Phase 1: `task_type` must be one of `primitive_implementation`, `validation`, `test_execution`
4. **Path validation**:
   - All `catalytic_domains` must be under `CATALYTIC-DPT/`
   - All `durable_paths` must be under allowed roots (CATALYTIC-DPT, CONTRACTS, TOOLS, etc.)
   - No paths outside the repo
   - No forbidden paths (CANON, AGENTS.md, BUILD)

### Ledger Validation

1. **File existence**: RUN_INFO, PRE_MANIFEST, POST_MANIFEST, RESTORE_DIFF, OUTPUTS, STATUS must exist
2. **JSON validity**: All files must be valid JSON
3. **Restoration proof**: RESTORE_DIFF must be empty (success) or documented (failure)
4. **Output existence**: All outputs listed in OUTPUTS.json must exist on disk
5. **Field presence**:
   - RUN_INFO must have: run_id, timestamp, intent, catalytic_domains, exit_code, restoration_verified
   - STATUS must have: status, restoration_verified

---

## Testing the Schemas

### Self-Validation

Each schema should validate its own structure:

```python
import json
import jsonschema

# Load jobspec schema
with open("CATALYTIC-DPT/SCHEMAS/jobspec.schema.json") as f:
    jobspec_schema = json.load(f)

# Validate that the schema itself is valid JSON Schema Draft 7
from jsonschema import Draft7Validator
Draft7Validator.check_schema(jobspec_schema)  # Raises if invalid

# Test with valid example
valid_jobspec = {
    "job_id": "test-valid",
    "phase": 0,
    "task_type": "schema_definition",
    "intent": "Test",
    "inputs": {},
    "outputs": {"durable_paths": [], "validation_criteria": {}},
    "catalytic_domains": []
}

jsonschema.validate(instance=valid_jobspec, schema=jobspec_schema)
# Passes if valid
```

### Test Fixtures

- `CATALYTIC-DPT/FIXTURES/phase0/valid/jobspec_*.json` - Examples that pass validation
- `CATALYTIC-DPT/FIXTURES/phase0/invalid/jobspec_*.json` - Examples that fail (with expected errors documented)

---

## Documentation for Codex (200M Model)

When a 200M model reads these schemas, it should understand:

1. **JobSpec**: "This is what I'm asked to do, broken down into fields"
2. **Validation Error Vector**: "If I produce errors, they follow this deterministic format"
3. **Ledger**: "After I execute, I produce this audit trail that proves restoration"

The schemas are the **contract** between human and small model. No ambiguity.

---

## Current Status

- [ ] jobspec.schema.json - TO DO
- [ ] validation_error.schema.json - TO DO
- [ ] ledger.schema.json - TO DO
- [ ] All schemas self-validating - TO DO
- [ ] Fixtures (valid + invalid) - TO DO
- [ ] Documentation complete - TO DO

**Next Step**: Implement Phase 0 contracts using this README as the specification.
