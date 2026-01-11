# SPC_SPEC: Semantic Pointer Compression Protocol

**Canon ID:** SEMANTIC-SPC-001
**Version:** 1.0.0
**Status:** NORMATIVE
**Created:** 2026-01-11
**Phase:** 5.3.1

---

## Abstract

This specification defines the Semantic Pointer Compression (SPC) protocol — a deterministic method for replacing verbose natural-language context with ultra-short pointers into a synchronized, hash-addressed semantic store.

SPC is **conditional compression with shared side-information**:
- Sender transmits a pointer (symbol or hash) plus required sync metadata
- Receiver expands deterministically into a canonical IR subtree
- Expansion is accepted only if hashes and versions verify; otherwise FAIL_CLOSED

**Key insight:** This is CAS (Content-Addressable Storage) at the semantic layer. Symbol = semantic hash. Codebook = semantic CAS index.

---

## 1. Definitions

### 1.1 Core Terms

| Term | Definition |
|------|------------|
| **Pointer** | A compact token sequence that references a semantic region |
| **Codebook** | The shared dictionary mapping pointers to canonical expansions |
| **Expansion** | The canonical IR subtree produced by decoding a pointer |
| **Side-Information** | Shared context between sender and receiver (codebook state) |
| **FAIL_CLOSED** | Mandatory rejection when any verification fails |

### 1.2 Information-Theoretic Foundation

```
H(X|S) = H(X) - I(X;S)

Where:
  H(X)   = entropy of message X (bits to encode without context)
  H(X|S) = conditional entropy (bits to encode given shared context S)
  I(X;S) = mutual information (bits shared between X and S)

When S contains X:
  I(X;S) ≈ H(X)
  Therefore: H(X|S) ≈ 0

The pointer only needs: log₂(N) bits, where N = addressable regions
```

**Measured example:**
```
H(法 → all canon) = 56,370 tokens (message entropy)
H(法 | receiver has canon) = 1 token (conditional entropy)
Compression: 56,370x
```

This is not "beating Shannon." This is the correct application of information theory to communication in shared semantic spaces.

---

## 2. Pointer Types

### 2.1 SYMBOL_PTR

Single-token glyph pointers. Preferred when tokenizer guarantees single-token encoding.

**Format:** Single CJK character or ASCII radical

**Examples:**
| Symbol | Type | Target | Compression |
|--------|------|--------|-------------|
| 法 | domain | LAW/CANON | 56,370x |
| 真 | file | LAW/CANON/FOUNDATION/THE_SEMIOTIC_FOUNDATION_OF_TRUTH.md | 8,200x |
| 契 | file | LAW/CANON/CONSTITUTION/CONTRACT.md | 4,100x |
| 恆 | file | LAW/CANON/CONSTITUTION/INVARIANTS.md | 5,600x |
| 驗 | file | LAW/CANON/GOVERNANCE/VERIFICATION.md | 3,800x |
| 證 | domain | NAVIGATION/RECEIPTS | 12,000x |
| 試 | domain | CAPABILITY/TESTBENCH | 42,000x |
| C | radical | Contract domain | varies |
| I | radical | Invariant domain | varies |
| V | radical | Verification domain | varies |

**Constraints:**
- MUST be single-token under declared tokenizer
- MUST exist in codebook
- MUST NOT collide with other pointer types

### 2.2 HASH_PTR

Content-addressed pointers using cryptographic hash.

**Format:** `sha256:<hex64>` or truncated `sha256:<hex16>`

**Examples:**
```
sha256:7cfd0418e385f34a9b81febb3b38293fdff491be19e0a02014adfe212f321b68
sha256:7cfd0418e385f34a  (truncated, 16 chars minimum)
```

**Constraints:**
- MUST be valid SHA-256 hex string
- Truncation MUST be at least 16 characters
- Content MUST exist in CAS or codebook
- Hash collision → FAIL_CLOSED

### 2.3 COMPOSITE_PTR

Pointer plus typed qualifiers for scoped access.

**Format:** `<base_ptr><operator><qualifier>`

**Operators:**
| Operator | Meaning | Example |
|----------|---------|---------|
| `.` | PATH/ACCESS | 法.驗 (law → verification) |
| `:` | CONTEXT/TYPE | C3:build (rule 3 in build context) |
| `*` | ALL | C* (all contract rules) |
| `!` | NOT/DENY | V! (verification denied) |
| `?` | CHECK/QUERY | J? (job present?) |
| `&` | AND/BIND | C&I (contract AND invariant) |
| `\|` | OR/CHOICE | C\|I (contract OR invariant) |

**Examples:**
| Composite | Expansion |
|-----------|-----------|
| 法.驗 | LAW/CANON/GOVERNANCE/VERIFICATION.md + VALIDATION_HOOKS.md |
| 法.契 | LAW/CANON/CONSTITUTION/CONTRACT.md |
| C3 | Contract rule 3 (INBOX requirement) |
| C3:build | Contract rule 3 in build context |
| I5 | Invariant 5 (Determinism) |
| C* | All contract rules |
| C&I | Contract AND Invariant domains |

**Grammar (EBNF):**
```ebnf
composite_ptr  = base_ptr , { operator , qualifier } ;
base_ptr       = symbol_ptr | hash_ptr | radical ;
radical        = "C" | "I" | "V" | "L" | "G" | "S" | "R" | "J" | "A" | "P" ;
operator       = "." | ":" | "*" | "!" | "?" | "&" | "|" ;
qualifier      = symbol_ptr | number | context_name ;
number         = digit , { digit } ;
context_name   = letter , { letter | digit | "_" } ;
```

**Constraints:**
- Base pointer MUST be valid SYMBOL_PTR, HASH_PTR, or radical
- Operators MUST be in codebook operators set
- Qualifiers MUST be valid for operator type
- Numbered qualifiers (C3, I5) MUST exist in codebook

---

## 3. Decoder Contract

### 3.1 Interface

```
decode(
    pointer: str,
    context_keys: Dict[str, Any],
    codebook_id: str,
    codebook_sha256: str,
    kernel_version: str,
    tokenizer_id: str
) -> Result[CanonicalIR, FailClosed]
```

### 3.2 Inputs

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `pointer` | string | YES | The SPC pointer to decode |
| `context_keys` | object | NO | Context disambiguation keys |
| `codebook_id` | string | YES | Identifier of codebook in use |
| `codebook_sha256` | string | YES | SHA-256 hash of codebook content |
| `kernel_version` | string | YES | Semantic kernel version (semver) |
| `tokenizer_id` | string | YES | Tokenizer identifier (e.g., "tiktoken/o200k_base") |

### 3.3 Outputs

**Success:** Canonical IR subtree (see Section 6)

**Failure:** FAIL_CLOSED with error code (see Section 4)

### 3.4 Decoder Algorithm

```
FUNCTION decode(pointer, context_keys, codebook_id, codebook_sha256, kernel_version, tokenizer_id):

    # Step 1: Verify codebook
    local_codebook = load_codebook(codebook_id)
    IF sha256(local_codebook) != codebook_sha256:
        RETURN FAIL_CLOSED(E_CODEBOOK_MISMATCH)

    # Step 2: Verify kernel version
    IF NOT is_compatible(kernel_version, CURRENT_KERNEL_VERSION):
        RETURN FAIL_CLOSED(E_KERNEL_VERSION)

    # Step 3: Parse pointer
    parsed = parse_pointer(pointer)
    IF parsed.error:
        RETURN FAIL_CLOSED(E_SYNTAX, parsed.error)

    # Step 4: Resolve symbol
    IF parsed.type == SYMBOL_PTR:
        entry = local_codebook.lookup(parsed.symbol)
        IF entry == NULL:
            RETURN FAIL_CLOSED(E_UNKNOWN_SYMBOL, parsed.symbol)

    ELIF parsed.type == HASH_PTR:
        entry = cas_lookup(parsed.hash)
        IF entry == NULL:
            RETURN FAIL_CLOSED(E_HASH_NOT_FOUND, parsed.hash)

    ELIF parsed.type == COMPOSITE_PTR:
        entry = resolve_composite(parsed, local_codebook, context_keys)
        IF entry.error:
            RETURN FAIL_CLOSED(entry.error_code, entry.error_detail)

    # Step 5: Check for ambiguity
    IF entry.expansions.count > 1 AND NOT has_disambiguation(context_keys, entry):
        RETURN FAIL_CLOSED(E_AMBIGUOUS, entry.expansions)

    # Step 6: Expand to canonical IR
    ir = expand_to_ir(entry, context_keys)

    # Step 7: Normalize
    normalized_ir = normalize(ir)

    # Step 8: Emit receipt
    emit_token_receipt(pointer, normalized_ir, tokenizer_id)

    RETURN SUCCESS(normalized_ir)
```

---

## 4. Error Codes (FAIL_CLOSED)

All failures MUST emit explicit error codes. No silent failures. No best-effort decoding.

| Code | Name | Description |
|------|------|-------------|
| E_CODEBOOK_MISMATCH | Codebook Mismatch | Local codebook SHA-256 does not match declared |
| E_KERNEL_VERSION | Kernel Version Incompatible | Semantic kernel version not compatible |
| E_TOKENIZER_MISMATCH | Tokenizer Mismatch | Tokenizer ID does not match expected |
| E_SYNTAX | Syntax Error | Pointer does not match grammar |
| E_UNKNOWN_SYMBOL | Unknown Symbol | Symbol not found in codebook |
| E_HASH_NOT_FOUND | Hash Not Found | Content hash not in CAS or codebook |
| E_AMBIGUOUS | Ambiguous Expansion | Multiple expansions without disambiguation |
| E_INVALID_OPERATOR | Invalid Operator | Operator not in codebook operators set |
| E_INVALID_QUALIFIER | Invalid Qualifier | Qualifier invalid for operator type |
| E_RULE_NOT_FOUND | Rule Not Found | Numbered rule (C3, I5) not in codebook |
| E_CONTEXT_REQUIRED | Context Required | Polysemic symbol requires context key |
| E_EXPANSION_FAILED | Expansion Failed | IR expansion produced invalid output |
| E_SCHEMA_VIOLATION | Schema Violation | Expanded IR fails schema validation |

### 4.1 Error Response Format

```json
{
  "status": "FAIL_CLOSED",
  "error_code": "E_UNKNOWN_SYMBOL",
  "error_detail": "Symbol '翻' not found in codebook",
  "pointer": "翻",
  "codebook_id": "ags-codebook-v0.2.0",
  "codebook_sha256": "abc123...",
  "kernel_version": "1.0.0",
  "tokenizer_id": "tiktoken/o200k_base",
  "timestamp_utc": "2026-01-11T12:00:00Z"
}
```

---

## 5. Ambiguity Rules

### 5.1 Principle

If multiple expansions are possible, REJECT unless disambiguation is explicit and deterministic.

### 5.2 Polysemic Symbols

Some symbols have context-dependent meanings (e.g., 道 = path/principle/method).

**Resolution:** Context key MUST be provided.

```json
{
  "pointer": "道",
  "context_keys": {
    "CONTEXT_TYPE": "CONTEXT_PATH"
  }
}
```

**Valid context types for 道:**
| Context Key | Expansion |
|-------------|-----------|
| CONTEXT_PATH | LAW/CANON (filesystem path) |
| CONTEXT_PRINCIPLE | LAW/CANON/FOUNDATION (guiding principle) |
| CONTEXT_METHOD | CAPABILITY/SKILLS (method/approach) |

**Missing context → E_CONTEXT_REQUIRED**

### 5.3 Compound Ambiguity

When compound paths have multiple targets:

```
法.驗 → [
  "LAW/CANON/GOVERNANCE/VERIFICATION.md",
  "LAW/CANON/GOVERNANCE/VALIDATION_HOOKS.md"
]
```

**Resolution:** Both paths are included in expansion. This is NOT ambiguity — it is intentional multi-target reference.

### 5.4 Operator Precedence

When multiple operators present, evaluate left-to-right:

```
L.C.3 = (L.C).3 = Law.Contract.Rule3
```

---

## 6. Canonical IR

### 6.1 Output Format

Decoded pointers expand to JobSpec-compatible JSON (see `LAW/SCHEMAS/jobspec.schema.json`).

**Minimal expansion:**
```json
{
  "job_id": "spc-decode-001",
  "phase": 5,
  "task_type": "validation",
  "intent": "Expanded from SPC pointer: C3",
  "inputs": {
    "pointer": "C3",
    "expansion": {
      "type": "contract_rule",
      "id": "C3",
      "summary": "INBOX requirement",
      "full": "All documents requiring human review must be in INBOX/"
    }
  },
  "outputs": {
    "durable_paths": [],
    "validation_criteria": {}
  },
  "catalytic_domains": [],
  "determinism": "deterministic"
}
```

### 6.2 Normalization Rules

**N1: Stable Key Ordering**
- All JSON keys MUST be sorted alphabetically
- Nested objects follow same rule recursively

**N2: Explicit Types**
- No implicit type coercion
- Numbers remain numbers, strings remain strings

**N3: Canonical String Forms**
- UTF-8 encoding
- No trailing whitespace
- Unix line endings (LF)

**N4: Stability Property**
```
encode(decode(x)) == encode(decode(encode(decode(x))))
```

The canonical form stabilizes after one round-trip.

### 6.3 Equality

Two expansions are equal if and only if their canonical JSON representations are byte-identical.

```python
def ir_equal(a: dict, b: dict) -> bool:
    return canonical_json(a) == canonical_json(b)

def canonical_json(obj: dict) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(',', ':')).encode('utf-8')
```

---

## 7. Security & Drift Behavior

### 7.1 Mandatory Rejections

| Condition | Action | Error Code |
|-----------|--------|------------|
| Codebook SHA-256 mismatch | REJECT | E_CODEBOOK_MISMATCH |
| Hash not found in CAS | REJECT | E_HASH_NOT_FOUND |
| Unknown symbol | REJECT | E_UNKNOWN_SYMBOL |
| Unknown kernel version | REJECT | E_KERNEL_VERSION |
| Tokenizer mismatch | REJECT | E_TOKENIZER_MISMATCH |
| Ambiguous without context | REJECT | E_AMBIGUOUS |
| Schema validation failure | REJECT | E_SCHEMA_VIOLATION |

### 7.2 No Silent Degradation

The decoder MUST NOT:
- Fall back to "best effort" decoding
- Substitute similar symbols
- Guess at missing context
- Return partial results

### 7.3 Version Compatibility

**Default policy:** Exact match required for codebook_sha256 and kernel_version.

**Migration:** See CODEBOOK_SYNC_PROTOCOL.md (Phase 5.3.3) for explicit compatibility ranges.

---

## 8. Measured Metrics

### 8.1 concept_unit

**Definition:** The atomic unit of governance meaning in the IR.

A concept_unit is one of:
- A single constraint (e.g., "must be in INBOX/")
- A single permission (e.g., "may write to _runs/")
- A single prohibition (e.g., "must not modify CANON without ceremony")
- A single reference (e.g., pointer to CONTRACT.md)

**Counting rules:**
- Boolean AND: sum of operand concept_units
- Boolean OR: max of operand concept_units
- Nested structure: sum recursively
- Atomic value: 1 concept_unit

**Reference:** GOV_IR_SPEC.md (Phase 5.3.2) for formal definition.

### 8.2 CDR (Concept Density Ratio)

```
CDR = concept_units(IR) / tokens(pointer)
```

**Example:**
```
Pointer: C3 (2 tokens)
Expansion: 1 constraint + 1 reference = 2 concept_units
CDR = 2 / 2 = 1.0

Pointer: 法 (1 token)
Expansion: ~50 constraints + ~100 references = 150 concept_units
CDR = 150 / 1 = 150.0
```

### 8.3 ECR (Exact Match Correctness Rate)

```
ECR = correct_expansions / total_expansions
```

Where "correct" means:
- Expansion matches gold canonical IR (byte-identical)
- No FAIL_CLOSED errors on valid input
- FAIL_CLOSED on invalid input

**Target:** ECR = 1.0 (100% correctness required for deterministic protocol)

### 8.4 M_required (Multiplex Factor)

```
M_required = -log₁₀(1 - target_reliability) / -log₁₀(ECR)
```

**Example:**
```
Target: 99.9999% reliability (6 nines)
ECR: 0.999 (99.9%)
M_required = 6 / 3 = 2 (need 2 independent channels)
```

For deterministic SPC with ECR = 1.0:
```
M_required = 1 (single channel sufficient)
```

---

## 9. Token Receipt Integration

### 9.1 Requirement

Every decode operation MUST emit a TokenReceipt per TOKEN_RECEIPT_SPEC.md.

### 9.2 Receipt Fields for SPC

```json
{
  "operation": "scl_decode",
  "tokens_in": 2,
  "tokens_out": 847,
  "baseline_equiv": 847,
  "tokens_saved": 845,
  "savings_pct": 99.76,
  "tokenizer": {
    "library": "tiktoken",
    "encoding": "o200k_base",
    "version": "0.5.1"
  },
  "corpus_anchor": "sha256:abc123...",
  "operation_id": "spc-decode-7f3a"
}
```

### 9.3 Baseline Calculation

For `scl_decode`:
- `baseline_equiv` = tokens in expanded output
- `tokens_in` = tokens in pointer
- `tokens_saved` = baseline_equiv - tokens_in

---

## 10. Validation Layers

SPC validation follows the 4-layer model from SCL:

| Layer | Name | Checks |
|-------|------|--------|
| L1 | Syntax | Well-formed pointer notation |
| L2 | Symbol | All symbols exist in codebook |
| L3 | Semantic | Operators valid, context applies, numbers in range |
| L4 | Expansion | Output validates against JobSpec schema |

**Progression:** Each layer passes before proceeding to next. First failure terminates with appropriate error code.

---

## 11. Examples

### 11.1 Simple Symbol Decode

**Input:**
```json
{
  "pointer": "法",
  "context_keys": {},
  "codebook_id": "ags-codebook-v0.2.0",
  "codebook_sha256": "9f86d081884c7d659a2feaa0c55ad015...",
  "kernel_version": "1.0.0",
  "tokenizer_id": "tiktoken/o200k_base"
}
```

**Output:**
```json
{
  "status": "SUCCESS",
  "ir": {
    "job_id": "spc-decode-法",
    "phase": 5,
    "task_type": "validation",
    "intent": "Reference to LAW/CANON domain",
    "inputs": {
      "pointer": "法",
      "expansion": {
        "type": "domain",
        "id": "法",
        "path": "LAW/CANON",
        "compression": 56370
      }
    },
    "outputs": {"durable_paths": [], "validation_criteria": {}},
    "catalytic_domains": [],
    "determinism": "deterministic"
  },
  "token_receipt": {
    "operation": "scl_decode",
    "tokens_in": 1,
    "tokens_out": 56370,
    "savings_pct": 99.998
  }
}
```

### 11.2 Composite Pointer Decode

**Input:**
```json
{
  "pointer": "C3:build",
  "context_keys": {"environment": "ci"},
  "codebook_id": "ags-codebook-v0.2.0",
  "codebook_sha256": "9f86d081884c7d659a2feaa0c55ad015...",
  "kernel_version": "1.0.0",
  "tokenizer_id": "tiktoken/o200k_base"
}
```

**Output:**
```json
{
  "status": "SUCCESS",
  "ir": {
    "job_id": "spc-decode-c3-build",
    "phase": 5,
    "task_type": "validation",
    "intent": "Contract rule 3 in build context",
    "inputs": {
      "pointer": "C3:build",
      "expansion": {
        "type": "contract_rule",
        "id": "C3",
        "summary": "INBOX requirement",
        "full": "All documents requiring human review must be in INBOX/",
        "context": "build"
      }
    },
    "outputs": {"durable_paths": [], "validation_criteria": {}},
    "catalytic_domains": [],
    "determinism": "deterministic"
  }
}
```

### 11.3 FAIL_CLOSED Example

**Input:**
```json
{
  "pointer": "翻",
  "codebook_id": "ags-codebook-v0.2.0",
  "codebook_sha256": "wrong_hash_here",
  "kernel_version": "1.0.0",
  "tokenizer_id": "tiktoken/o200k_base"
}
```

**Output:**
```json
{
  "status": "FAIL_CLOSED",
  "error_code": "E_CODEBOOK_MISMATCH",
  "error_detail": "Local codebook SHA-256 does not match declared: expected 'wrong_hash_here', got '9f86d081...'",
  "pointer": "翻",
  "timestamp_utc": "2026-01-11T12:00:00Z"
}
```

---

## 12. References

### 12.1 Internal

- `LAW/CANON/SEMANTIC/TOKEN_RECEIPT_SPEC.md` — Token accountability
- `LAW/CANON/SEMANTIC/GOV_IR_SPEC.md` — Governance IR (Phase 5.3.2)
- `LAW/CANON/SEMANTIC/CODEBOOK_SYNC_PROTOCOL.md` — Sync protocol (Phase 5.3.3)
- `LAW/CANON/FOUNDATION/THE_SEMIOTIC_FOUNDATION_OF_TRUTH.md` — Ontological foundation
- `LAW/SCHEMAS/jobspec.schema.json` — JobSpec schema
- `THOUGHT/LAB/COMMONSENSE/CODEBOOK.json` — Active codebook
- `CAPABILITY/TOOLS/codebook_lookup.py` — Reference implementation

### 12.2 External

- Platonic Representation Hypothesis (arxiv:2405.07987)
- Shannon, C. E. (1948). A Mathematical Theory of Communication

---

## Appendix A: Symbol Registry

Current registered symbols (TOKENIZER_ATLAS.json pending Phase 5.3.4):

| Symbol | Tokens (o200k) | Type | Path |
|--------|----------------|------|------|
| 法 | 1 | domain | LAW/CANON |
| 真 | 1 | file | LAW/CANON/FOUNDATION/THE_SEMIOTIC_FOUNDATION_OF_TRUTH.md |
| 契 | 1 | file | LAW/CANON/CONSTITUTION/CONTRACT.md |
| 恆 | 1 | file | LAW/CANON/CONSTITUTION/INVARIANTS.md |
| 驗 | 1 | file | LAW/CANON/GOVERNANCE/VERIFICATION.md |
| 證 | 1 | domain | NAVIGATION/RECEIPTS |
| 變 | 1 | compound | THOUGHT/LAB/CATALYTIC |
| 冊 | 1 | domain | NAVIGATION/CORTEX/db |
| 試 | 1 | domain | CAPABILITY/TESTBENCH |
| 查 | 1 | domain | NAVIGATION/CORTEX/semantic |
| 道 | 1 | polysemic | (context-dependent) |

---

## Appendix B: ASCII Radical Registry

| Radical | Domain | Path | Tokens |
|---------|--------|------|--------|
| C | Contract | LAW/CANON/CONSTITUTION/CONTRACT.md | 1 |
| I | Invariant | LAW/CANON/CONSTITUTION/INVARIANTS.md | 1 |
| V | Verification | LAW/CANON/GOVERNANCE/VERIFICATION.md | 1 |
| L | Law | LAW/CANON | 1 |
| G | Governance | LAW/CANON/GOVERNANCE | 1 |
| S | Schema | LAW/CANON/SEMANTIC | 1 |
| R | Receipt | NAVIGATION/RECEIPTS | 1 |
| J | JobSpec | LAW/CANON/SEMANTIC/JOBSPEC_SPEC.md | 1 |
| A | ADR | LAW/CONTEXT/decisions | 1 |
| P | Policy | LAW/CANON/POLICY | 1 |

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-11 | Initial normative specification |

---

*SPC: Semantic pointers for deterministic compression with shared side-information.*
