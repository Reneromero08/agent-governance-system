# GOV_IR_SPEC: Governance Intermediate Representation

**Canon ID:** SEMANTIC-GOV-IR-001
**Version:** 1.0.0
**Status:** NORMATIVE
**Created:** 2026-01-11
**Phase:** 5.3.2

---

## Abstract

This specification defines the Governance Intermediate Representation (GOV_IR) — a minimal typed IR for expressing governance meaning in machine-verifiable form. GOV_IR enables:

1. **Countable Semantics** — Every governance statement maps to countable `concept_units`
2. **Deterministic Comparison** — Byte-identical canonical JSON defines equality
3. **SPC Integration** — Pointers decode to GOV_IR subtrees (see SPC_SPEC.md)
4. **Schema Validation** — All IR nodes validate against JSON Schema

**Key insight:** Governance meaning must be typed and countable to enable measured compression. GOV_IR provides the semantic substrate for SPC's Concept Density Ratio (CDR).

---

## 1. Definitions

### 1.1 Core Terms

| Term | Definition |
|------|------------|
| **GOV_IR** | Governance Intermediate Representation — typed AST for governance meaning |
| **IRNode** | A single node in the GOV_IR tree |
| **concept_unit** | Atomic unit of governance meaning (see Section 6) |
| **Canonical Form** | Normalized JSON with stable ordering and explicit types |
| **CDR** | Concept Density Ratio = concept_units / tokens |

### 1.2 Design Principles

1. **Minimal** — Only primitives needed to express governance meaning
2. **Typed** — Every node has an explicit type, no inference
3. **Deterministic** — Same input always produces same canonical output
4. **Composable** — Complex expressions built from simple primitives
5. **Verifiable** — All nodes validate against JSON Schema

---

## 2. IR Primitives

### 2.1 Node Types

Every GOV_IR node has a `type` field from this enum:

```
IRNodeType =
  | "constraint"      // MUST/MUST NOT requirements
  | "permission"      // MAY allowances
  | "prohibition"     // MUST NOT denials
  | "reference"       // Pointer to canon/path/artifact
  | "gate"            // Verification checkpoint
  | "operation"       // Boolean/comparison operation
  | "literal"         // Atomic value (string, number, boolean)
  | "sequence"        // Ordered list of nodes
  | "record"          // Key-value mapping
```

### 2.2 Base Node Schema

All IR nodes share this base structure:

```json
{
  "type": "<IRNodeType>",
  "id": "<optional unique identifier>",
  "source": "<optional source location>",
  "metadata": {}
}
```

---

## 3. Primitive Definitions

### 3.1 Constraint Node

Expresses a requirement that MUST be satisfied.

```json
{
  "type": "constraint",
  "op": "requires" | "ensures" | "maintains",
  "subject": "<IRNode>",
  "predicate": "<IRNode>",
  "severity": "must" | "should" | "may"
}
```

**Examples:**
```json
{
  "type": "constraint",
  "op": "requires",
  "subject": {"type": "reference", "ref_type": "path", "value": "INBOX/"},
  "predicate": {"type": "literal", "value_type": "string", "value": "human-review documents"},
  "severity": "must"
}
```

**concept_units:** 1 per constraint node

### 3.2 Permission Node

Expresses an allowance (MAY).

```json
{
  "type": "permission",
  "op": "allows" | "grants" | "enables",
  "subject": "<IRNode>",
  "scope": "<IRNode>",
  "conditions": ["<IRNode>"]
}
```

**concept_units:** 1 per permission node

### 3.3 Prohibition Node

Expresses a denial (MUST NOT).

```json
{
  "type": "prohibition",
  "op": "forbids" | "denies" | "blocks",
  "subject": "<IRNode>",
  "target": "<IRNode>",
  "exceptions": ["<IRNode>"]
}
```

**concept_units:** 1 per prohibition node

### 3.4 Reference Node

Points to a canon path, version, tool, or artifact.

```json
{
  "type": "reference",
  "ref_type": "path" | "canon_version" | "tool_id" | "artifact_hash" | "rule_id" | "invariant_id",
  "value": "<string>",
  "anchor": "<optional hash or version>"
}
```

**Reference Types:**

| ref_type | Format | Example |
|----------|--------|---------|
| `path` | Relative path from repo root | `"LAW/CANON/CONSTITUTION/CONTRACT.md"` |
| `canon_version` | Semver string | `"1.8.0"` |
| `tool_id` | Tool identifier | `"scl_cli"` |
| `artifact_hash` | SHA-256 hex | `"7cfd0418e385f34a..."` |
| `rule_id` | Contract rule ID | `"C3"` |
| `invariant_id` | Invariant ID | `"INV-005"` |

**concept_units:** 1 per reference node

### 3.5 Gate Node

Verification checkpoint that must pass.

```json
{
  "type": "gate",
  "gate_type": "test" | "restore_proof" | "allowlist_check" | "hash_verify" | "schema_validate",
  "target": "<IRNode>",
  "pass_criteria": "<IRNode>",
  "fail_action": "reject" | "warn" | "log"
}
```

**Gate Types:**

| gate_type | Purpose | Example |
|-----------|---------|---------|
| `test` | Run test suite | Fixture tests must pass |
| `restore_proof` | Verify catalytic restore | Byte-identical after run |
| `allowlist_check` | Verify path in allowed roots | Output in `_runs/` |
| `hash_verify` | Content hash matches | Receipt hash valid |
| `schema_validate` | JSON Schema validation | JobSpec validates |

**concept_units:** 1 per gate node

### 3.6 Operation Node

Boolean and comparison operations.

```json
{
  "type": "operation",
  "op": "<OperationType>",
  "operands": ["<IRNode>"]
}
```

**Operation Types:**

| Category | Operators |
|----------|-----------|
| Boolean | `AND`, `OR`, `NOT`, `XOR`, `IMPLIES` |
| Comparison | `EQ`, `NE`, `LT`, `LE`, `GT`, `GE` |
| Set | `IN`, `NOT_IN`, `SUBSET`, `SUPERSET`, `INTERSECTS` |
| String | `MATCH`, `STARTS_WITH`, `ENDS_WITH`, `CONTAINS` |
| Existence | `EXISTS`, `NOT_EXISTS`, `IS_NULL`, `IS_NOT_NULL` |

**concept_units:**
- `AND`: sum of operand concept_units
- `OR`: max of operand concept_units
- `NOT`: operand concept_units
- Others: 1 + sum of operand concept_units

### 3.7 Literal Node

Atomic values with explicit types.

```json
{
  "type": "literal",
  "value_type": "string" | "integer" | "number" | "boolean" | "null",
  "value": "<typed value>"
}
```

**concept_units:** 0 (literals are structural, not semantic)

### 3.8 Sequence Node

Ordered list of nodes.

```json
{
  "type": "sequence",
  "elements": ["<IRNode>"]
}
```

**concept_units:** sum of element concept_units

### 3.9 Record Node

Key-value mapping.

```json
{
  "type": "record",
  "fields": {
    "<key>": "<IRNode>"
  }
}
```

**concept_units:** sum of field value concept_units

---

## 4. Side-Effects Flags

Operations that modify state must declare side-effects.

```json
{
  "type": "constraint",
  "side_effects": {
    "writes": ["<path>"],
    "deletes": ["<path>"],
    "creates": ["<path>"],
    "modifies_canon": false,
    "requires_ceremony": false,
    "emits_receipt": true
  }
}
```

**Side-Effect Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `writes` | string[] | Paths that may be written |
| `deletes` | string[] | Paths that may be deleted |
| `creates` | string[] | Paths that may be created |
| `modifies_canon` | boolean | Changes to LAW/CANON |
| `requires_ceremony` | boolean | ADR/version bump required |
| `emits_receipt` | boolean | TokenReceipt emitted |

---

## 5. Canonical JSON Schema

### 5.1 Normalization Rules

**N1: Stable Key Ordering**
All JSON object keys MUST be sorted alphabetically (Unicode code point order).

```python
def sort_keys(obj):
    if isinstance(obj, dict):
        return {k: sort_keys(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, list):
        return [sort_keys(v) for v in obj]
    return obj
```

**N2: Explicit Types**
- No implicit type coercion
- Numbers: Use JSON number type (no quotes)
- Booleans: Use JSON `true`/`false` (no quotes)
- Null: Use JSON `null` (no quotes)
- Strings: Always quoted

**N3: Canonical String Forms**
- Encoding: UTF-8
- Line endings: LF only (no CRLF)
- Trailing whitespace: None
- Unicode: NFC normalized
- Escaping: Minimal (only required escapes)

**N4: Compact Representation**
- No pretty-printing whitespace
- Separators: `,` and `:` with no spaces
- No trailing commas

**N5: Stability Property**
```
normalize(normalize(x)) == normalize(x)
```

### 5.2 Canonical JSON Function

```python
import json
import unicodedata

def canonical_json(obj: dict) -> bytes:
    """Convert IR node to canonical JSON bytes."""
    def normalize(o):
        if isinstance(o, dict):
            return {k: normalize(v) for k, v in sorted(o.items())}
        elif isinstance(o, list):
            return [normalize(v) for v in o]
        elif isinstance(o, str):
            return unicodedata.normalize('NFC', o)
        return o

    normalized = normalize(obj)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        separators=(',', ':'),
        sort_keys=True
    ).encode('utf-8')
```

### 5.3 Canonical Hash

```python
import hashlib

def canonical_hash(obj: dict) -> str:
    """Compute SHA-256 of canonical JSON."""
    return hashlib.sha256(canonical_json(obj)).hexdigest()
```

---

## 6. Equality Definition

### 6.1 Byte-Identical Equality

Two GOV_IR nodes are **equal** if and only if their canonical JSON representations are byte-identical.

```python
def ir_equal(a: dict, b: dict) -> bool:
    """Test IR node equality via canonical form."""
    return canonical_json(a) == canonical_json(b)
```

### 6.2 Hash Equality

Equivalent definition using hashes:

```python
def ir_hash_equal(a: dict, b: dict) -> bool:
    """Test IR node equality via canonical hash."""
    return canonical_hash(a) == canonical_hash(b)
```

### 6.3 Structural Equivalence

For debugging, structural equivalence ignores metadata:

```python
def ir_structural_equal(a: dict, b: dict) -> bool:
    """Test structural equivalence (ignores metadata)."""
    def strip_meta(o):
        if isinstance(o, dict):
            return {k: strip_meta(v) for k, v in o.items()
                    if k not in ('id', 'source', 'metadata')}
        elif isinstance(o, list):
            return [strip_meta(v) for v in o]
        return o
    return canonical_json(strip_meta(a)) == canonical_json(strip_meta(b))
```

**Note:** Structural equivalence is for debugging only. Normative equality is byte-identical.

---

## 7. concept_unit Definition

### 7.1 Definition

A **concept_unit** is the atomic unit of governance meaning in GOV_IR.

Each concept_unit represents one of:
- A single **constraint** (requirement that MUST be satisfied)
- A single **permission** (allowance that MAY be exercised)
- A single **prohibition** (action that MUST NOT occur)
- A single **reference** (pointer to governance artifact)
- A single **gate** (verification checkpoint)

### 7.2 Counting Rules

```python
def count_concept_units(node: dict) -> int:
    """Count concept_units in an IR node."""
    node_type = node.get('type')

    # Atomic semantic nodes: 1 concept_unit each
    if node_type in ('constraint', 'permission', 'prohibition', 'reference', 'gate'):
        return 1

    # Literals: 0 (structural, not semantic)
    if node_type == 'literal':
        return 0

    # Operations: depends on operator
    if node_type == 'operation':
        op = node.get('op')
        operands = node.get('operands', [])
        operand_units = [count_concept_units(o) for o in operands]

        if op == 'AND':
            return sum(operand_units)
        elif op == 'OR':
            return max(operand_units) if operand_units else 0
        elif op == 'NOT':
            return operand_units[0] if operand_units else 0
        else:
            return 1 + sum(operand_units)

    # Sequences: sum of elements
    if node_type == 'sequence':
        return sum(count_concept_units(e) for e in node.get('elements', []))

    # Records: sum of field values
    if node_type == 'record':
        return sum(count_concept_units(v) for v in node.get('fields', {}).values())

    return 0
```

### 7.3 CDR Calculation

Concept Density Ratio ties concept_units to token cost:

```
CDR = concept_units(IR) / tokens(pointer)
```

**Example:**
```
Pointer: C3 (2 tokens)
Expansion IR:
  {
    "type": "constraint",
    "op": "requires",
    "subject": {"type": "reference", "ref_type": "path", "value": "INBOX/"},
    "predicate": {"type": "literal", "value_type": "string", "value": "human-review"}
  }

concept_units = 1 (constraint) + 1 (reference) = 2
CDR = 2 / 2 = 1.0
```

**High-compression example:**
```
Pointer: 法 (1 token)
Expansion: ~50 constraints + ~100 references = 150 concept_units
CDR = 150 / 1 = 150.0
```

---

## 8. Schema Validation

### 8.1 JSON Schema Reference

GOV_IR nodes MUST validate against: `LAW/SCHEMAS/gov_ir.schema.json`

### 8.2 Validation Layers

| Layer | Validates |
|-------|-----------|
| L1 Syntax | JSON well-formed, required fields present |
| L2 Types | Field types match schema |
| L3 Semantics | Values in valid ranges, refs resolvable |
| L4 Integration | Node fits in larger IR tree |

### 8.3 Validation Function

```python
import jsonschema

def validate_ir_node(node: dict, schema: dict) -> tuple[bool, list[str]]:
    """Validate IR node against schema."""
    errors = []
    try:
        jsonschema.validate(node, schema)
    except jsonschema.ValidationError as e:
        errors.append(str(e.message))
    return len(errors) == 0, errors
```

---

## 9. Mapping from Governance Sources

### 9.1 Contract Rules → IR

Each contract rule (C1-C13) maps to IR:

```python
CONTRACT_RULE_IR = {
    "C1": {
        "type": "constraint",
        "op": "requires",
        "subject": {"type": "reference", "ref_type": "path", "value": "LAW/CANON"},
        "predicate": {"type": "literal", "value_type": "string", "value": "text outranks code"},
        "severity": "must"
    },
    "C3": {
        "type": "constraint",
        "op": "requires",
        "subject": {"type": "reference", "ref_type": "path", "value": "INBOX/"},
        "predicate": {"type": "literal", "value_type": "string", "value": "human-review documents"},
        "severity": "must"
    },
    "C7": {
        "type": "constraint",
        "op": "ensures",
        "subject": {"type": "literal", "value_type": "string", "value": "system"},
        "predicate": {"type": "literal", "value_type": "string", "value": "deterministic outputs"},
        "severity": "must"
    }
}
```

### 9.2 Invariants → IR

Each invariant (INV-001 to INV-020) maps to IR:

```python
INVARIANT_IR = {
    "INV-005": {
        "type": "constraint",
        "op": "maintains",
        "subject": {"type": "literal", "value_type": "string", "value": "system"},
        "predicate": {"type": "literal", "value_type": "string", "value": "determinism"},
        "severity": "must",
        "metadata": {"invariant_id": "INV-005", "summary": "Same inputs produce same outputs"}
    },
    "INV-006": {
        "type": "prohibition",
        "op": "forbids",
        "subject": {"type": "literal", "value_type": "string", "value": "artifacts"},
        "target": {
            "type": "operation",
            "op": "NOT_IN",
            "operands": [
                {"type": "reference", "ref_type": "path", "value": "_runs/"},
                {"type": "reference", "ref_type": "path", "value": "_generated/"},
                {"type": "reference", "ref_type": "path", "value": "_packs/"}
            ]
        },
        "exceptions": []
    }
}
```

### 9.3 Gates → IR

Verification gates from VERIFICATION.md:

```python
GATE_IR = {
    "fixture_gate": {
        "type": "gate",
        "gate_type": "test",
        "target": {"type": "reference", "ref_type": "path", "value": "LAW/CONTRACTS/fixtures/"},
        "pass_criteria": {"type": "literal", "value_type": "string", "value": "all tests pass"},
        "fail_action": "reject"
    },
    "restore_gate": {
        "type": "gate",
        "gate_type": "restore_proof",
        "target": {"type": "reference", "ref_type": "path", "value": "catalytic_domains"},
        "pass_criteria": {"type": "literal", "value_type": "string", "value": "byte-identical"},
        "fail_action": "reject"
    }
}
```

---

## 10. Integration with SPC

### 10.1 Decoder Output

SPC decode produces GOV_IR nodes:

```
decode(pointer, context, codebook) → GOV_IR | FAIL_CLOSED
```

### 10.2 JobSpec Wrapping

GOV_IR nodes are wrapped in JobSpec for execution:

```json
{
  "job_id": "ir-decode-c3",
  "phase": 5,
  "task_type": "validation",
  "intent": "Decoded from SPC pointer: C3",
  "inputs": {
    "pointer": "C3",
    "ir": {
      "type": "constraint",
      "op": "requires",
      "subject": {"type": "reference", "ref_type": "path", "value": "INBOX/"},
      "predicate": {"type": "literal", "value_type": "string", "value": "human-review documents"},
      "severity": "must"
    }
  },
  "outputs": {"durable_paths": [], "validation_criteria": {}},
  "catalytic_domains": [],
  "determinism": "deterministic"
}
```

### 10.3 Metrics Integration

GOV_IR enables SPC metrics:

```json
{
  "pointer": "C3",
  "tokens_pointer": 2,
  "ir_concept_units": 2,
  "cdr": 1.0,
  "ecr": 1.0
}
```

---

## 11. Examples

### 11.1 Simple Constraint

**Natural Language:** "All documents requiring human review must be in INBOX/"

**GOV_IR:**
```json
{
  "type": "constraint",
  "op": "requires",
  "subject": {
    "type": "reference",
    "ref_type": "path",
    "value": "INBOX/"
  },
  "predicate": {
    "type": "literal",
    "value_type": "string",
    "value": "human-review documents"
  },
  "severity": "must"
}
```

**concept_units:** 2 (1 constraint + 1 reference)

### 11.2 Compound Prohibition

**Natural Language:** "Artifacts must not be written outside allowed roots"

**GOV_IR:**
```json
{
  "type": "prohibition",
  "op": "forbids",
  "subject": {
    "type": "literal",
    "value_type": "string",
    "value": "system-generated artifacts"
  },
  "target": {
    "type": "operation",
    "op": "NOT_IN",
    "operands": [
      {"type": "reference", "ref_type": "path", "value": "_runs/"},
      {"type": "reference", "ref_type": "path", "value": "_generated/"},
      {"type": "reference", "ref_type": "path", "value": "_packs/"}
    ]
  },
  "exceptions": []
}
```

**concept_units:** 5 (1 prohibition + 1 operation + 3 references)

### 11.3 Verification Gate

**Natural Language:** "All fixtures must pass before merge"

**GOV_IR:**
```json
{
  "type": "gate",
  "gate_type": "test",
  "target": {
    "type": "reference",
    "ref_type": "path",
    "value": "LAW/CONTRACTS/fixtures/"
  },
  "pass_criteria": {
    "type": "operation",
    "op": "EQ",
    "operands": [
      {"type": "literal", "value_type": "string", "value": "exit_code"},
      {"type": "literal", "value_type": "integer", "value": 0}
    ]
  },
  "fail_action": "reject"
}
```

**concept_units:** 3 (1 gate + 1 reference + 1 operation)

---

## 12. References

### 12.1 Internal

- `LAW/CANON/SEMANTIC/SPC_SPEC.md` — Semantic Pointer Compression
- `LAW/CANON/SEMANTIC/TOKEN_RECEIPT_SPEC.md` — Token accountability
- `LAW/SCHEMAS/gov_ir.schema.json` — GOV_IR JSON Schema
- `LAW/SCHEMAS/jobspec.schema.json` — JobSpec schema
- `LAW/CANON/CONSTITUTION/CONTRACT.md` — Contract rules source
- `LAW/CANON/CONSTITUTION/INVARIANTS.md` — Invariants source

### 12.2 External

- JSON Schema Draft-07: https://json-schema.org/draft-07/schema
- Unicode NFC: https://unicode.org/reports/tr15/

---

## Appendix A: Complete Node Type Reference

| Type | concept_units | Fields |
|------|---------------|--------|
| `constraint` | 1 | op, subject, predicate, severity |
| `permission` | 1 | op, subject, scope, conditions |
| `prohibition` | 1 | op, subject, target, exceptions |
| `reference` | 1 | ref_type, value, anchor |
| `gate` | 1 | gate_type, target, pass_criteria, fail_action |
| `operation` | varies | op, operands |
| `literal` | 0 | value_type, value |
| `sequence` | sum | elements |
| `record` | sum | fields |

---

## Appendix B: Operation Precedence

When evaluating nested operations:

1. `NOT` (unary, highest)
2. `AND`
3. `OR`
4. `IMPLIES`
5. Comparison operators (lowest)

Parentheses (nesting) override precedence.

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-11 | Initial normative specification |

---

*GOV_IR: Making governance meaning typed, countable, and verifiable.*
