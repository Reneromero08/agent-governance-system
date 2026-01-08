# TOKEN_RECEIPT_SPEC

> Canon Law: Every operation that consumes or saves tokens MUST emit a TokenReceipt.

**Version:** 1.0.0
**Status:** ACTIVE
**Phase:** 5.2.7 Token Accountability Layer

---

## Purpose

TokenReceipt is a mandatory primitive that accompanies every token-consuming operation. It provides:

1. **Accountability** - Every operation reports its token cost
2. **Compression Proof** - Savings are measured, not claimed
3. **Enforcement** - Operations without receipts are rejected
4. **Aggregation** - Session totals computed from individual receipts

---

## Schema

### TokenReceipt (JSON)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": ["operation", "tokens_out", "tokenizer"],
  "properties": {
    "operation": {
      "type": "string",
      "description": "Operation type that generated this receipt",
      "enum": [
        "semantic_query",
        "scl_decode",
        "scl_encode",
        "cas_get",
        "cas_put",
        "skill_invoke",
        "session_load",
        "expand_hash"
      ]
    },
    "tokens_in": {
      "type": "integer",
      "minimum": 0,
      "description": "Input tokens (query, command, etc.)"
    },
    "tokens_out": {
      "type": "integer",
      "minimum": 0,
      "description": "Output tokens (result, expansion, etc.)"
    },
    "baseline_equiv": {
      "type": "integer",
      "minimum": 0,
      "description": "Tokens that would be consumed by paste-scan baseline"
    },
    "tokens_saved": {
      "type": "integer",
      "description": "baseline_equiv - tokens_out (can be negative if expansion > baseline)"
    },
    "savings_pct": {
      "type": "number",
      "minimum": -100,
      "maximum": 100,
      "description": "Percentage savings: (tokens_saved / baseline_equiv) * 100"
    },
    "tokenizer": {
      "type": "object",
      "required": ["library", "encoding"],
      "properties": {
        "library": {
          "type": "string",
          "description": "Tokenizer library (tiktoken, word-count-proxy)"
        },
        "version": {
          "type": "string",
          "description": "Library version"
        },
        "encoding": {
          "type": "string",
          "description": "Encoding used (o200k_base, cl100k_base)"
        }
      }
    },
    "timestamp_utc": {
      "type": "string",
      "format": "date-time",
      "description": "ISO 8601 timestamp"
    },
    "corpus_anchor": {
      "type": "string",
      "description": "SHA256 hash of corpus state (for reproducibility)"
    },
    "operation_id": {
      "type": "string",
      "description": "Unique identifier for this operation"
    }
  }
}
```

### Python Dataclass

```python
from dataclasses import dataclass
from typing import Optional
import hashlib
import datetime as _dt

@dataclass
class TokenizerInfo:
    library: str        # "tiktoken" or "word-count-proxy"
    encoding: str       # "o200k_base", "cl100k_base"
    version: Optional[str] = None

@dataclass
class TokenReceipt:
    operation: str              # semantic_query, scl_decode, etc.
    tokens_out: int             # Output tokens
    tokenizer: TokenizerInfo    # Tokenizer used
    tokens_in: int = 0          # Input tokens
    baseline_equiv: int = 0     # Paste-scan baseline
    tokens_saved: int = 0       # Savings (computed)
    savings_pct: float = 0.0    # Savings percentage
    timestamp_utc: Optional[str] = None
    corpus_anchor: Optional[str] = None
    operation_id: Optional[str] = None

    def __post_init__(self):
        if self.timestamp_utc is None:
            self.timestamp_utc = _dt.datetime.utcnow().isoformat() + "Z"
        if self.baseline_equiv > 0:
            self.tokens_saved = self.baseline_equiv - self.tokens_out
            self.savings_pct = (self.tokens_saved / self.baseline_equiv) * 100
        if self.operation_id is None:
            self.operation_id = hashlib.sha256(
                f"{self.timestamp_utc}:{self.operation}:{self.tokens_out}".encode()
            ).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "operation": self.operation,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "baseline_equiv": self.baseline_equiv,
            "tokens_saved": self.tokens_saved,
            "savings_pct": round(self.savings_pct, 2),
            "tokenizer": {
                "library": self.tokenizer.library,
                "encoding": self.tokenizer.encoding,
                "version": self.tokenizer.version,
            },
            "timestamp_utc": self.timestamp_utc,
            "corpus_anchor": self.corpus_anchor,
            "operation_id": self.operation_id,
        }
```

---

## Required Surfaces

TokenReceipt MUST be emitted by:

| Surface | Operation Type | Baseline Calculation |
|---------|---------------|---------------------|
| `semantic_search.py` | `semantic_query` | Sum of all indexed file tokens |
| `scl/decode.py` | `scl_decode` | Expanded output tokens |
| `scl/encode.py` | `scl_encode` | Input natural language tokens |
| `cas.py` | `cas_get` / `cas_put` | Content size in tokens |
| Skill executor | `skill_invoke` | Full context window estimate |
| Session loader | `session_load` | Raw session history tokens |
| Hash expander | `expand_hash` | Full content tokens |

---

## Aggregation

### SessionTokenSummary

```json
{
  "session_id": "abc123",
  "receipts_count": 47,
  "total_tokens_in": 1234,
  "total_tokens_out": 5678,
  "total_baseline_equiv": 1250000,
  "total_tokens_saved": 1244322,
  "aggregate_savings_pct": 99.54,
  "tokenizer": {
    "library": "tiktoken",
    "encoding": "o200k_base"
  }
}
```

---

## Enforcement

### Firewall Rules

1. **REJECT** any tool output > 1000 tokens without TokenReceipt
2. **WARN** if savings_pct < 50% for semantic_query operations
3. **LOG** all receipts to session ledger for audit

### Validation

```python
def validate_receipt(receipt: dict) -> bool:
    """Validate TokenReceipt is present and well-formed."""
    required = ["operation", "tokens_out", "tokenizer"]
    if not all(k in receipt for k in required):
        return False
    if receipt.get("baseline_equiv", 0) > 0:
        expected_saved = receipt["baseline_equiv"] - receipt["tokens_out"]
        if receipt.get("tokens_saved") != expected_saved:
            return False
    return True
```

---

## Display Format

### Compact (CLI)

```
[TOKEN] semantic_query: 834 tokens (saved 623,336 / 99.87%)
```

### Verbose (Reports)

```
TOKEN RECEIPT
─────────────
Operation:     semantic_query
Tokens Out:    834
Baseline:      624,170
Saved:         623,336 (99.87%)
Tokenizer:     tiktoken/o200k_base
Corpus:        c4d4bcd66a7b...
```

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-08 | Initial specification |

---

*TokenReceipt: Making compression accountable.*
