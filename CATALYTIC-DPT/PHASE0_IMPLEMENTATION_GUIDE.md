# Phase 0 Implementation Guide

**Executor**: Codex (200M parameter model)
**Task**: Define three canonical JSON schemas
**Budget**: Use minimal tokens, log everything
**Success Criteria**: All schemas valid, documented, tested

---

## The Three Schemas (Specification)

### 1. jobspec.schema.json - Complete Specification

This defines what a valid job looks like.

**Structure**:
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Catalytic JobSpec",
  "description": "Canonical specification for a catalytic computing job",
  "type": "object",
  "required": [
    "job_id",
    "phase",
    "task_type",
    "intent",
    "inputs",
    "outputs",
    "catalytic_domains"
  ],
  "properties": {
    "job_id": {
      "type": "string",
      "pattern": "^[a-z0-9-]+$",
      "description": "Unique identifier in kebab-case"
    },
    "phase": {
      "type": "integer",
      "enum": [0, 1],
      "description": "0 for contract definition, 1+ for execution"
    },
    "task_type": {
      "type": "string",
      "enum": [
        "schema_definition",
        "primitive_implementation",
        "validation",
        "test_execution"
      ],
      "description": "Category of work"
    },
    "intent": {
      "type": "string",
      "maxLength": 200,
      "description": "One-sentence purpose"
    },
    "inputs": {
      "type": "object",
      "description": "Input parameters (structure varies by task_type)"
    },
    "outputs": {
      "type": "object",
      "required": ["durable_paths", "validation_criteria"],
      "properties": {
        "durable_paths": {
          "type": "array",
          "items": {"type": "string"},
          "description": "Where outputs may persist (must be under allowed roots)"
        },
        "validation_criteria": {
          "type": "object",
          "description": "Success criteria for this job"
        }
      }
    },
    "catalytic_domains": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Scratch spaces that must be restored (must be under CATALYTIC-DPT/)"
    },
    "determinism": {
      "type": "string",
      "enum": ["deterministic", "bounded_nondeterministic", "nondeterministic"],
      "default": "deterministic",
      "description": "Determinism level"
    },
    "swarm_parallel": {
      "type": "boolean",
      "default": false,
      "description": "Whether to offload to parallel execution"
    },
    "metadata": {
      "type": "object",
      "properties": {
        "priority": {
          "type": "integer",
          "minimum": 0,
          "maximum": 10
        },
        "timeout_seconds": {
          "type": "integer",
          "minimum": 1
        },
        "requires_governance_check": {
          "type": "boolean",
          "default": true
        }
      }
    }
  },
  "additionalProperties": false
}
```

**Validation Rules**:
1. `job_id` must match regex `^[a-z0-9-]+$`
2. `phase` must be 0 or 1
3. `task_type` must be in the enum
4. If `phase` == 0, then `task_type` must be `schema_definition`
5. All paths in `catalytic_domains` must be under `CATALYTIC-DPT/`
6. All paths in `durable_paths` must be under allowed roots (CATALYTIC-DPT/, CONTRACTS/, TOOLS/)

---

### 2. validation_error.schema.json - Complete Specification

This defines what a deterministic error report looks like.

**Structure**:
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Validation Error Vector",
  "description": "Deterministic validation error format",
  "type": "object",
  "required": ["valid", "errors", "warnings", "timestamp"],
  "properties": {
    "valid": {
      "type": "boolean",
      "description": "True if validation passed"
    },
    "errors": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["code", "message", "path"],
        "properties": {
          "code": {
            "type": "string",
            "pattern": "^[A-Z][A-Z0-9_]*$",
            "description": "Machine-readable error code (UPPERCASE_SNAKE_CASE)"
          },
          "message": {
            "type": "string",
            "description": "Human-readable error description"
          },
          "path": {
            "type": "string",
            "description": "JSONPath to error location (e.g., $.phase or $.outputs[0])"
          },
          "details": {
            "type": "object",
            "description": "Additional context (e.g., expected values, actual value)"
          }
        }
      },
      "description": "List of validation failures"
    },
    "warnings": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["code", "message"],
        "properties": {
          "code": {
            "type": "string",
            "pattern": "^[A-Z][A-Z0-9_]*$"
          },
          "message": {
            "type": "string"
          }
        }
      },
      "description": "List of non-blocking issues"
    },
    "timestamp": {
      "type": "string",
      "format": "date-time",
      "description": "When validation occurred (ISO 8601)"
    },
    "validator_version": {
      "type": "string",
      "description": "Version of validator that produced this report"
    }
  },
  "additionalProperties": false
}
```

**Error Codes** (Canonical List - Must Include):
- `SCHEMA_INVALID` - JSON itself is malformed
- `MISSING_REQUIRED_FIELD` - Required field not present
- `INVALID_FIELD_TYPE` - Wrong data type
- `INVALID_ENUM_VALUE` - Value not in allowed set
- `INVALID_FORMAT` - Field fails format validation (regex, etc.)
- `INVALID_PATH` - Path doesn't exist or not allowed
- `VALIDATION_LOGIC_ERROR` - Custom logic check failed
- `OUTPUT_DOES_NOT_EXIST` - Expected output file not found
- `RESTORATION_FAILED` - Catalytic domain not restored

---

### 3. ledger.schema.json - Complete Specification

This defines what a complete run audit trail looks like.

**Structure**:
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Catalytic Run Ledger",
  "description": "Complete audit trail and restoration proof for a catalytic run",
  "type": "object",
  "required": [
    "run_id",
    "timestamp",
    "intent",
    "exit_code",
    "restoration_verified"
  ],
  "properties": {
    "run_id": {
      "type": "string",
      "pattern": "^[a-z0-9-]+$",
      "description": "Unique run identifier"
    },
    "timestamp": {
      "type": "string",
      "format": "date-time",
      "description": "When run started (ISO 8601)"
    },
    "intent": {
      "type": "string",
      "description": "Copy of job_spec.intent"
    },
    "job_spec": {
      "$ref": "jobspec.schema.json",
      "description": "The JobSpec that was executed (optional)"
    },
    "catalytic_domains": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Scratch spaces that should have been restored"
    },
    "durable_output_roots": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Paths where outputs may persist"
    },
    "exit_code": {
      "type": "integer",
      "description": "Exit code from executed command"
    },
    "restoration_verified": {
      "type": "boolean",
      "description": "True if catalytic domains were fully restored"
    },
    "pre_manifest": {
      "type": "object",
      "description": "Hash map of catalytic domain state before execution",
      "additionalProperties": {
        "type": "object",
        "description": "Path -> SHA256 hash mapping"
      }
    },
    "post_manifest": {
      "type": "object",
      "description": "Hash map of catalytic domain state after execution"
    },
    "restore_diff": {
      "type": "object",
      "description": "Difference between pre and post manifests",
      "additionalProperties": {
        "type": "object",
        "properties": {
          "added": {"type": "object"},
          "removed": {"type": "object"},
          "changed": {"type": "object"}
        }
      }
    },
    "outputs": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["path", "type"],
        "properties": {
          "path": {
            "type": "string",
            "description": "Path relative to repo root"
          },
          "type": {
            "enum": ["file", "directory"],
            "description": "File or directory"
          },
          "sha256": {
            "type": "string",
            "pattern": "^[a-f0-9]{64}$",
            "description": "SHA-256 hash of file content (for files only)"
          }
        }
      },
      "description": "List of durable outputs that were created"
    },
    "decision_log": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["timestamp", "decision", "reason"],
        "properties": {
          "timestamp": {
            "type": "string",
            "format": "date-time"
          },
          "decision": {
            "type": "string",
            "description": "What decision was made"
          },
          "reason": {
            "type": "string",
            "description": "Why that decision was made"
          },
          "tool_called": {
            "type": "string",
            "description": "MCP tool name (if applicable)"
          },
          "result": {
            "type": "object",
            "description": "Result from tool call"
          }
        }
      },
      "description": "Timestamped sequence of decisions (OPTIONAL but recommended)"
    }
  },
  "additionalProperties": false
}
```

---

## Testing Each Schema

### Test 1: Schema Self-Validation

Each schema must validate itself.

```python
import json
import jsonschema

# Test jobspec.schema.json
with open("CATALYTIC-DPT/SCHEMAS/jobspec.schema.json") as f:
    jobspec_schema = json.load(f)

# Check it's valid JSON Schema Draft 7
from jsonschema import Draft7Validator
Draft7Validator.check_schema(jobspec_schema)
print("✓ jobspec.schema.json is valid JSON Schema Draft 7")

# Repeat for other two schemas
```

### Test 2: Valid Examples Pass

```python
# Valid jobspec example
valid_spec = {
    "job_id": "test-valid",
    "phase": 0,
    "task_type": "schema_definition",
    "intent": "Test",
    "inputs": {},
    "outputs": {"durable_paths": ["CATALYTIC-DPT/TESTBENCH/"], "validation_criteria": {}},
    "catalytic_domains": ["CATALYTIC-DPT/TESTBENCH/_tmp"],
    "determinism": "deterministic"
}

jsonschema.validate(instance=valid_spec, schema=jobspec_schema)
print("✓ Valid example passes jobspec validation")
```

### Test 3: Invalid Examples Fail

```python
# Missing required field
invalid_spec = {
    "job_id": "test-invalid"
    # Missing phase, task_type, intent, outputs, catalytic_domains
}

try:
    jsonschema.validate(instance=invalid_spec, schema=jobspec_schema)
    print("✗ Should have failed but didn't")
except jsonschema.ValidationError as e:
    print(f"✓ Invalid example correctly rejected: {e.message}")
```

---

## Fixture Examples

### CATALYTIC-DPT/FIXTURES/phase0/valid/jobspec_phase0_basic.json

```json
{
  "job_id": "phase0-jobspec-schema",
  "phase": 0,
  "task_type": "schema_definition",
  "intent": "Create canonical JobSpec JSON schema for catalytic computing",
  "inputs": {
    "requirements": [
      "Must support Phase 0 and Phase 1 task types",
      "Must include catalytic_domains and durable_outputs",
      "Must be JSON Schema Draft 7 compliant"
    ]
  },
  "outputs": {
    "durable_paths": [
      "CATALYTIC-DPT/SCHEMAS/jobspec.schema.json"
    ],
    "validation_criteria": {
      "schema_is_valid_json": true,
      "schema_validates_self": true
    }
  },
  "catalytic_domains": [
    "CATALYTIC-DPT/TESTBENCH/_runs/phase0-jobspec-schema/_tmp"
  ],
  "determinism": "deterministic",
  "swarm_parallel": false,
  "metadata": {
    "priority": 10,
    "timeout_seconds": 300
  }
}
```

### CATALYTIC-DPT/FIXTURES/phase0/invalid/jobspec_missing_phase.json

```json
{
  "job_id": "test-invalid-no-phase",
  "task_type": "schema_definition",
  "intent": "Should fail: missing 'phase' field",
  "_error_expected": "MISSING_REQUIRED_FIELD",
  "_path_expected": "$.phase"
}
```

---

## Summary: What Codex Creates

**File 1**: `CATALYTIC-DPT/SCHEMAS/jobspec.schema.json`
- ~120 lines
- Defines job specification format
- Includes all properties above
- Valid JSON Schema Draft 7

**File 2**: `CATALYTIC-DPT/SCHEMAS/validation_error.schema.json`
- ~80 lines
- Defines error reporting format
- Includes error codes
- Valid JSON Schema Draft 7

**File 3**: `CATALYTIC-DPT/SCHEMAS/ledger.schema.json`
- ~140 lines
- Defines audit trail format
- References jobspec.schema.json
- Valid JSON Schema Draft 7

**Artifacts**:
- Tests proving self-validation
- Valid and invalid examples
- Complete documentation
- Run ledger in CONTRACTS/_runs/<run_id>/

**Estimated Time**: 1-2 hours

---

## The Checklist (Codex, use this)

Before reporting success:

- [ ] jobspec.schema.json exists
- [ ] jobspec.schema.json is valid JSON
- [ ] jobspec.schema.json is valid JSON Schema Draft 7
- [ ] jobspec.schema.json validates against example jobspecs
- [ ] validation_error.schema.json exists
- [ ] validation_error.schema.json is valid JSON
- [ ] validation_error.schema.json is valid JSON Schema Draft 7
- [ ] ledger.schema.json exists
- [ ] ledger.schema.json is valid JSON
- [ ] ledger.schema.json is valid JSON Schema Draft 7
- [ ] All schemas have descriptions
- [ ] Examples created in FIXTURES/
- [ ] critic_run passes (pre and post)
- [ ] Run ledger is in CONTRACTS/_runs/<run_id>/
- [ ] Restoration_verified == true
- [ ] Task log documents all decisions
- [ ] Report sent to Claude

**When all checked**: Phase 0 is complete. Ready for Phase 1.
