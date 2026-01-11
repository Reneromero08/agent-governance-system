# CODEBOOK_SYNC_PROTOCOL: Semantic Side-Information Synchronization

**Canon ID:** SEMANTIC-SYNC-001
**Version:** 1.0.0
**Status:** NORMATIVE
**Created:** 2026-01-11
**Phase:** 5.3.3

---

## Abstract

This specification defines the Codebook Sync Protocol — the mechanism by which sender and receiver establish shared side-information for Semantic Pointer Compression (SPC).

**Core insight:** The sync protocol defines a **Markov blanket** between communicating parties. The codebook state represents the boundary through which semantic pointers can flow. When boundaries align (hashes match), information transfers deterministically. When they diverge, the system fails closed until resync.

**Information-theoretic foundation:**
```
H(X|S) ≈ 0  when S contains the expansion of X

Where:
  S = shared codebook state (side-information)
  X = semantic pointer
  H(X|S) = bits needed to decode X given S
```

The sync protocol establishes S. Without sync, H(X|S) = H(X) — no compression benefit.

---

## 1. Definitions

### 1.1 Core Terms

| Term | Definition |
|------|------------|
| **Codebook** | The shared dictionary mapping pointers to canonical expansions |
| **Side-Information** | Shared state between sender and receiver (S in H(X\|S)) |
| **Sync Handshake** | Protocol exchange establishing shared codebook state |
| **Markov Blanket** | Boundary separating systems while permitting information exchange |
| **Blanket Alignment** | State where sender and receiver share identical codebook |
| **Blanket Dissolution** | State where codebook mismatch prevents semantic transfer |

### 1.2 Sync Tuple

Every sync handshake exchanges a **Sync Tuple**:

```
SyncTuple = (codebook_id, codebook_sha256, codebook_semver,
             kernel_version, tokenizer_id)
```

| Field | Type | Description |
|-------|------|-------------|
| `codebook_id` | string | Unique identifier for codebook (e.g., "ags-codebook") |
| `codebook_sha256` | string | SHA-256 hash of canonical codebook JSON (64 hex chars) |
| `codebook_semver` | string | Semantic version of codebook (e.g., "0.2.0") |
| `kernel_version` | string | Semantic kernel version (e.g., "1.0.0") |
| `tokenizer_id` | string | Tokenizer identifier (e.g., "tiktoken/o200k_base") |

### 1.3 Design Principles

1. **Fail-Closed** — Any mismatch rejects; no silent degradation
2. **Explicit Sync** — Shared state must be verified before symbol expansion
3. **Deterministic** — Same sync tuple always produces same result
4. **Boundary Preservation** — Markov blanket maintained through continuous verification

---

## 2. Sync Handshake

### 2.1 Handshake Flow

```
┌─────────┐                              ┌──────────┐
│ Sender  │                              │ Receiver │
└────┬────┘                              └────┬─────┘
     │                                        │
     │  ───── SyncRequest ─────────────────>  │
     │        (sender's SyncTuple)            │
     │                                        │
     │  <──── SyncResponse ────────────────   │
     │        (match status + receiver's      │
     │         SyncTuple if mismatch)         │
     │                                        │
     │  IF MATCHED:                           │
     │  ───── SemanticPointer ────────────>   │
     │        (pointer + context_keys)        │
     │                                        │
     │  <──── Expansion ───────────────────   │
     │        (canonical IR or FAIL_CLOSED)   │
     │                                        │
```

### 2.2 Handshake States

| State | Description | Next Action |
|-------|-------------|-------------|
| `UNSYNCED` | No handshake performed | Send SyncRequest |
| `PENDING` | Awaiting SyncResponse | Wait or timeout |
| `SYNCED` | Blankets aligned | May send pointers |
| `MISMATCHED` | Blankets divergent | Resync required |
| `FAILED` | Handshake error | Retry or escalate |

### 2.3 State Transitions

```
UNSYNCED ──[SyncRequest]──> PENDING
PENDING  ──[SyncResponse:MATCH]──> SYNCED
PENDING  ──[SyncResponse:MISMATCH]──> MISMATCHED
PENDING  ──[Timeout]──> FAILED
SYNCED   ──[HashChange]──> MISMATCHED
SYNCED   ──[SessionEnd]──> UNSYNCED
MISMATCHED ──[Resync]──> PENDING
```

---

## 3. Handshake Message Shapes

### 3.1 SyncRequest

Sender initiates handshake with their codebook state.

```json
{
  "message_type": "SYNC_REQUEST",
  "protocol_version": "1.0.0",
  "sender_id": "agent-001",
  "timestamp_utc": "2026-01-11T12:00:00.000Z",
  "sync_tuple": {
    "codebook_id": "ags-codebook",
    "codebook_sha256": "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
    "codebook_semver": "0.2.0",
    "kernel_version": "1.0.0",
    "tokenizer_id": "tiktoken/o200k_base"
  },
  "capabilities": ["symbol_ptr", "hash_ptr", "composite_ptr"],
  "request_id": "sync-7f3a8b2c"
}
```

**Required Fields:**

| Field | Type | Constraints |
|-------|------|-------------|
| `message_type` | string | Must be "SYNC_REQUEST" |
| `protocol_version` | string | Semver, must be compatible with receiver |
| `sender_id` | string | Non-empty, unique sender identifier |
| `timestamp_utc` | string | ISO 8601 format, UTC timezone |
| `sync_tuple` | object | All 5 fields required |
| `request_id` | string | Unique request identifier for correlation |

**Optional Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `capabilities` | string[] | Pointer types sender supports |
| `session_id` | string | For session-scoped sync |

### 3.2 SyncResponse

Receiver responds with match status and optional correction data.

**Match Response:**
```json
{
  "message_type": "SYNC_RESPONSE",
  "protocol_version": "1.0.0",
  "receiver_id": "cortex-main",
  "timestamp_utc": "2026-01-11T12:00:00.050Z",
  "request_id": "sync-7f3a8b2c",
  "status": "MATCHED",
  "sync_tuple": {
    "codebook_id": "ags-codebook",
    "codebook_sha256": "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
    "codebook_semver": "0.2.0",
    "kernel_version": "1.0.0",
    "tokenizer_id": "tiktoken/o200k_base"
  },
  "blanket_status": "ALIGNED",
  "session_token": "sess-abc123",
  "ttl_seconds": 3600
}
```

**Mismatch Response:**
```json
{
  "message_type": "SYNC_RESPONSE",
  "protocol_version": "1.0.0",
  "receiver_id": "cortex-main",
  "timestamp_utc": "2026-01-11T12:00:00.050Z",
  "request_id": "sync-7f3a8b2c",
  "status": "MISMATCHED",
  "sync_tuple": {
    "codebook_id": "ags-codebook",
    "codebook_sha256": "different_hash_here_abc123...",
    "codebook_semver": "0.3.0",
    "kernel_version": "1.0.0",
    "tokenizer_id": "tiktoken/o200k_base"
  },
  "blanket_status": "DISSOLVED",
  "mismatch_fields": ["codebook_sha256", "codebook_semver"],
  "migration_available": true,
  "migration_path": {
    "from_version": "0.2.0",
    "to_version": "0.3.0",
    "migration_id": "migrate-0.2.0-to-0.3.0",
    "migration_sha256": "migration_hash_here..."
  }
}
```

**Response Status Values:**

| Status | Description | Action Required |
|--------|-------------|-----------------|
| `MATCHED` | All sync tuple fields match | Proceed with pointers |
| `MISMATCHED` | One or more fields differ | Resync or migrate |
| `INCOMPATIBLE` | Protocol versions incompatible | Upgrade required |
| `ERROR` | Handshake processing error | Retry or escalate |

### 3.3 SyncError

Error during handshake processing.

```json
{
  "message_type": "SYNC_ERROR",
  "protocol_version": "1.0.0",
  "receiver_id": "cortex-main",
  "timestamp_utc": "2026-01-11T12:00:00.050Z",
  "request_id": "sync-7f3a8b2c",
  "error_code": "E_PROTOCOL_VERSION",
  "error_detail": "Protocol version 2.0.0 not supported. Supported: 1.0.x",
  "retry_after_seconds": 0
}
```

---

## 4. Failure Codes

### 4.1 Sync-Specific Failures

| Code | Name | Description | Recovery |
|------|------|-------------|----------|
| `E_SYNC_REQUIRED` | Sync Required | Pointer sent without handshake | Perform handshake |
| `E_SYNC_EXPIRED` | Sync Expired | Session token expired | Re-handshake |
| `E_SYNC_TIMEOUT` | Sync Timeout | Handshake timed out | Retry |
| `E_PROTOCOL_VERSION` | Protocol Incompatible | Sync protocol version mismatch | Upgrade |
| `E_BLANKET_DISSOLVED` | Blanket Dissolved | Codebook changed mid-session | Re-handshake |

### 4.2 Codebook Failures (from SPC_SPEC)

| Code | Name | Description | Recovery |
|------|------|-------------|----------|
| `E_CODEBOOK_MISMATCH` | Codebook Mismatch | SHA-256 hash mismatch | Resync codebook |
| `E_KERNEL_VERSION` | Kernel Incompatible | Kernel version mismatch | Update kernel |
| `E_TOKENIZER_MISMATCH` | Tokenizer Mismatch | Tokenizer ID differs | Align tokenizer |
| `E_CODEBOOK_NOT_FOUND` | Codebook Not Found | codebook_id unknown | Obtain codebook |

### 4.3 Migration Failures

| Code | Name | Description | Recovery |
|------|------|-------------|----------|
| `E_MIGRATION_NOT_FOUND` | Migration Not Found | No migration path exists | Manual update |
| `E_MIGRATION_FAILED` | Migration Failed | Migration script error | Report bug |
| `E_MIGRATION_HASH_MISMATCH` | Migration Hash Invalid | Migration artifact corrupted | Re-fetch |
| `E_MIGRATION_NOT_ALLOWED` | Migration Not Allowed | Policy forbids auto-migration | Manual approval |

---

## 5. Compatibility Policy

### 5.1 Default Policy: Exact Match

By default, ALL sync tuple fields must match exactly:

```
MATCH iff:
  sender.codebook_sha256 == receiver.codebook_sha256
  AND sender.kernel_version == receiver.kernel_version
  AND sender.tokenizer_id == receiver.tokenizer_id
```

**Rationale:** Determinism requires identical side-information. Any deviation introduces potential for divergent expansions.

### 5.2 Semver Compatibility Ranges

Optional explicit compatibility ranges may be declared. These are NEVER silent — migration step is always explicit.

**Codebook Semver:**
```
Compatible if:
  sender.codebook_semver.major == receiver.codebook_semver.major
  AND sender.codebook_semver.minor <= receiver.codebook_semver.minor
  AND migration_path_exists(sender.codebook_semver, receiver.codebook_semver)
```

**Kernel Semver:**
```
Compatible if:
  sender.kernel_version.major == receiver.kernel_version.major
```

### 5.3 Migration Protocol

When compatibility ranges allow migration:

1. **Receiver detects mismatch** with compatible versions
2. **Receiver returns `migration_path`** in SyncResponse
3. **Sender fetches migration** from declared location
4. **Sender verifies migration** hash matches `migration_sha256`
5. **Sender applies migration** to local codebook
6. **Sender re-handshakes** with updated sync tuple

**Migration Artifact:**
```json
{
  "migration_id": "migrate-0.2.0-to-0.3.0",
  "from_version": "0.2.0",
  "to_version": "0.3.0",
  "migration_sha256": "abc123...",
  "operations": [
    {"type": "add_symbol", "symbol": "新", "path": "LAW/CANON/NEW"},
    {"type": "rename_radical", "from": "X", "to": "Y"},
    {"type": "deprecate", "symbol": "@OLD", "replacement": "N"}
  ],
  "created_utc": "2026-01-10T00:00:00Z",
  "author": "system"
}
```

### 5.4 No Silent Migration

**CRITICAL:** Migration is NEVER silent.

- Receiver MUST indicate mismatch even if compatible range allows migration
- Sender MUST explicitly apply migration before retry
- Audit log MUST record migration event
- Token receipts MUST include migration metadata if migration occurred

---

## 6. Cassette Network Integration

### 6.1 Cassette Sync State

Each cassette carries codebook state as part of its handshake:

```python
class SemanticCassette(DatabaseCassette):
    def handshake(self) -> Dict:
        base = super().handshake()
        base["sync_tuple"] = {
            "codebook_id": self.codebook_id,
            "codebook_sha256": self._compute_codebook_hash(),
            "codebook_semver": self.codebook_version,
            "kernel_version": KERNEL_VERSION,
            "tokenizer_id": self.tokenizer_id
        }
        base["blanket_status"] = self._check_blanket()
        return base
```

### 6.2 Verification Before Expansion

Before any symbol expansion, cassette MUST verify sync:

```python
def expand_pointer(self, pointer: str, context_keys: dict) -> Result:
    # Step 1: Check blanket alignment
    if self.blanket_status != "ALIGNED":
        return FailClosed(E_BLANKET_DISSOLVED)

    # Step 2: Verify codebook hash hasn't changed
    current_hash = self._compute_codebook_hash()
    if current_hash != self.sync_tuple["codebook_sha256"]:
        self.blanket_status = "DISSOLVED"
        return FailClosed(E_CODEBOOK_MISMATCH)

    # Step 3: Proceed with expansion
    return self._decode(pointer, context_keys)
```

### 6.3 Network-Wide Sync

When multiple cassettes form a network:

1. **Hub cassette** maintains authoritative codebook
2. **Spoke cassettes** sync with hub on initialization
3. **Periodic heartbeat** verifies continued alignment
4. **Any mismatch** triggers network-wide resync

```
Hub ←──── SyncRequest ──── Spoke-1
Hub ────> SyncResponse ───> Spoke-1
Hub ←──── SyncRequest ──── Spoke-2
Hub ────> SyncResponse ───> Spoke-2
...
Hub ────> BroadcastSync ──> All Spokes (on codebook update)
```

### 6.4 Cassette Handshake Extension

Extended handshake for cassette network:

```json
{
  "cassette_id": "canon-index",
  "db_path": "NAVIGATION/CORTEX/db/canon_index.db",
  "db_hash": "7cfd0418e385f34a",
  "capabilities": ["fts5", "vector_search"],
  "schema_version": "1.0",
  "sync_tuple": {
    "codebook_id": "ags-codebook",
    "codebook_sha256": "9f86d081884c7d659a2feaa0c55ad015...",
    "codebook_semver": "0.2.0",
    "kernel_version": "1.0.0",
    "tokenizer_id": "tiktoken/o200k_base"
  },
  "blanket_status": "ALIGNED",
  "last_sync_utc": "2026-01-11T12:00:00Z",
  "stats": {
    "total_chunks": 1247,
    "symbol_coverage": 0.95
  }
}
```

---

## 7. Markov Blanket Semantics

### 7.1 Theoretical Foundation

The sync protocol formalizes a **Markov blanket** in the semiotic space:

```
┌─────────────────────────────────────────────────────┐
│                  External World                      │
│   (unknown symbols, drift, unshared context)        │
└─────────────────────┬───────────────────────────────┘
                      │
              ┌───────┴───────┐
              │ Markov Blanket │  ← Sync Protocol
              │ (sync_tuple)   │
              └───────┬───────┘
                      │
┌─────────────────────┴───────────────────────────────┐
│                 Shared Semantic Space               │
│   (codebook, kernel, tokenizer — deterministic)     │
└─────────────────────────────────────────────────────┘
```

### 7.2 Blanket Properties

**P1: Conditional Independence**
Given the blanket (sync_tuple), internal expansion is independent of external variation:
```
P(expansion | pointer, blanket) = P(expansion | pointer)
```
The blanket screens off external uncertainty.

**P2: Minimal Surprise**
When blankets align, expansion has zero surprise:
```
H(expansion | pointer, aligned_blanket) = 0
```
Perfect determinism within the blanket.

**P3: Blanket Dissolution = High Surprise**
When blankets diverge, expansion is undefined:
```
H(expansion | pointer, misaligned_blanket) = undefined → FAIL_CLOSED
```

### 7.3 Active Inference Interpretation

The sync protocol implements **Active Inference** at the protocol level:

1. **Prediction:** Sender predicts receiver has matching codebook
2. **Verification:** Handshake tests prediction
3. **Error Signal:** Mismatch = prediction error
4. **Action:** Resync to minimize prediction error

This is R-gating (per Q35): R > τ permits semantic transfer; R < τ requires resync.

### 7.4 Blanket Status Semantics

| Status | R-value | Interpretation |
|--------|---------|----------------|
| `ALIGNED` | R > τ | Stable blanket, semantic transfer permitted |
| `DISSOLVED` | R < τ | Blanket broken, resync required |
| `PENDING` | R ≈ τ | Boundary forming, awaiting confirmation |

---

## 8. Session Management

### 8.1 Session Lifecycle

```
SessionInit ──> Handshake ──> Synced ──> [Operations] ──> SessionEnd
                   │                          │
                   └──── Mismatch ────────────┘
                           │
                        Resync
```

### 8.2 Session Token

Successful handshake returns a session token for subsequent operations:

```json
{
  "session_token": "sess-abc123",
  "ttl_seconds": 3600,
  "sync_tuple_hash": "sha256:abc123..."
}
```

**Token Properties:**
- Bound to specific sync_tuple
- Expires after TTL
- Invalidated on any sync_tuple change
- Must be included in all pointer operations

### 8.3 Heartbeat

Long-lived sessions should send periodic heartbeats:

```json
{
  "message_type": "SYNC_HEARTBEAT",
  "session_token": "sess-abc123",
  "timestamp_utc": "2026-01-11T12:30:00Z",
  "local_codebook_sha256": "9f86d081884c7d659a2feaa0c55ad015..."
}
```

Response:
```json
{
  "message_type": "HEARTBEAT_ACK",
  "session_token": "sess-abc123",
  "blanket_status": "ALIGNED",
  "ttl_remaining_seconds": 1800
}
```

---

## 9. Implementation Notes

### 9.1 Hash Computation

Codebook hash MUST be computed from canonical JSON:

```python
import hashlib
import json

def compute_codebook_hash(codebook: dict) -> str:
    """Compute SHA-256 of canonical codebook JSON."""
    # Sort keys recursively
    def sort_recursive(obj):
        if isinstance(obj, dict):
            return {k: sort_recursive(v) for k, v in sorted(obj.items())}
        elif isinstance(obj, list):
            return [sort_recursive(v) for v in obj]
        return obj

    canonical = json.dumps(
        sort_recursive(codebook),
        ensure_ascii=False,
        separators=(',', ':')
    ).encode('utf-8')

    return hashlib.sha256(canonical).hexdigest()
```

### 9.2 Sync Tuple Comparison

```python
def sync_tuples_match(sender: dict, receiver: dict) -> tuple[bool, list[str]]:
    """Compare sync tuples, return (match, mismatched_fields)."""
    fields = ['codebook_sha256', 'kernel_version', 'tokenizer_id']
    mismatches = []

    for field in fields:
        if sender.get(field) != receiver.get(field):
            mismatches.append(field)

    return len(mismatches) == 0, mismatches
```

### 9.3 Blanket Status Check

```python
def check_blanket_status(local_tuple: dict, remote_tuple: dict) -> str:
    """Determine Markov blanket alignment status."""
    match, mismatches = sync_tuples_match(local_tuple, remote_tuple)

    if match:
        return "ALIGNED"

    # Check if compatible for migration
    if can_migrate(local_tuple, remote_tuple):
        return "DISSOLVED"  # But recoverable

    return "DISSOLVED"  # Not recoverable without manual intervention
```

---

## 10. Information-Theoretic Semantics

### 10.1 Conditional Entropy and Sync

The sync protocol's value is quantifiable via conditional entropy:

```
Without sync:  H(meaning) = H(X)           ← must transmit full expansion
With sync:     H(meaning|S) = H(X|S) ≈ 0   ← pointer suffices
Compression:   I(X;S) = H(X) - H(X|S)      ← mutual information with codebook
```

**Measured example:**
```
X = "All documents requiring human review must be in INBOX/"
S = codebook containing C3 → X mapping

H(X) ≈ 12 tokens (full statement)
H(X|S) ≈ 2 tokens (pointer "C3")
I(X;S) ≈ 10 tokens saved
Compression ratio: 6x
```

### 10.2 Semantic Density Connection

Per Q33 (conditional entropy vs semantic density):

The sync protocol establishes the conditions under which semantic density (σ^Df) becomes measurable:

```
CDR = concept_units / tokens = σ^Df (empirical)
```

Where:
- `concept_units` = atomic governance meaning (from GOV_IR_SPEC)
- `tokens` = pointer token count
- `σ` = semantic density (meaning per symbol)
- `Df` = fractal dimension (complexity of meaning structure)

**Key insight:** Sync enables CDR measurement. Without aligned blankets, CDR is undefined — there's no shared basis for counting concept_units.

### 10.3 When Density Helps vs Hurts

| Scenario | σ^Df Effect | Sync State |
|----------|-------------|------------|
| Aligned blankets + known symbol | High CDR, low H(X\|S) | SYNCED |
| Aligned blankets + unknown symbol | FAIL_CLOSED | SYNCED |
| Misaligned blankets | Undefined CDR | DISSOLVED |
| Polysemic symbol + context | Disambiguation via context_keys | SYNCED |
| Polysemic symbol + no context | E_AMBIGUOUS | SYNCED |

Higher semantic density **lowers** uncertainty when:
1. Blankets are aligned (S is shared)
2. Symbol is unambiguous (single expansion)
3. Context keys resolve polysemy

Higher semantic density **increases** ambiguity when:
1. Multiple valid expansions exist
2. Context is missing
3. Codebook has drifted

### 10.4 Measurement Procedure

To measure H(X|S) empirically:

1. **Establish S:** Perform sync handshake, confirm ALIGNED
2. **Encode X:** Map governance statement to pointer
3. **Count tokens:** tokens(pointer) = empirical H(X|S)
4. **Count baseline:** tokens(full expansion) = empirical H(X)
5. **Compute mutual information:** I(X;S) = H(X) - H(X|S)

**Example measurement:**
```python
def measure_compression(pointer: str, expansion: str, tokenizer: str) -> dict:
    """Measure conditional entropy and mutual information."""
    import tiktoken
    enc = tiktoken.get_encoding(tokenizer)

    h_x = len(enc.encode(expansion))        # H(X)
    h_x_given_s = len(enc.encode(pointer))  # H(X|S)
    i_x_s = h_x - h_x_given_s               # I(X;S)

    return {
        "H_X": h_x,
        "H_X_given_S": h_x_given_s,
        "I_X_S": i_x_s,
        "compression_ratio": h_x / h_x_given_s if h_x_given_s > 0 else float('inf')
    }
```

---

## 11. Security Considerations

### 11.1 Hash Collision Resistance

SHA-256 provides 128-bit collision resistance. For additional security:
- Full 64-char hash SHOULD be used in production
- Truncation to 16 chars acceptable for display only
- Any collision detection → FAIL_CLOSED

### 11.2 Replay Protection

Sync requests include:
- `timestamp_utc` — reject if too old (> 5 minutes)
- `request_id` — unique per request, track for replay detection
- `session_token` — bound to specific handshake

### 11.3 Man-in-the-Middle

In untrusted networks:
- Sync messages SHOULD be signed
- Codebook SHOULD be fetched from trusted source
- Migration artifacts MUST be hash-verified

---

## 12. References

### 12.1 Internal

- `LAW/CANON/SEMANTIC/SPC_SPEC.md` — Semantic Pointer Compression (uses this protocol)
- `LAW/CANON/SEMANTIC/GOV_IR_SPEC.md` — Governance IR (expansion target)
- `LAW/CANON/SEMANTIC/TOKEN_RECEIPT_SPEC.md` — Token accountability
- `NAVIGATION/CORTEX/network/cassette_protocol.py` — Implementation target
- `THOUGHT/LAB/COMMONSENSE/CODEBOOK.json` — Codebook artifact
- `THOUGHT/LAB/FORMULA/research/questions/high_priority/q35_markov_blankets.md` — Markov blanket foundation
- `THOUGHT/LAB/FORMULA/research/questions/medium_priority/q33_conditional_entropy_semantic_density.md` — Conditional entropy foundation

### 12.2 External

- Friston, K. (2019). A Free Energy Principle for a Particular Physics
- Pearl, J. (1988). Probabilistic Reasoning in Intelligent Systems (Markov blankets)
- Shannon, C. E. (1948). A Mathematical Theory of Communication

---

## Appendix A: Message Type Registry

| Message Type | Direction | Purpose |
|--------------|-----------|---------|
| `SYNC_REQUEST` | Sender → Receiver | Initiate handshake |
| `SYNC_RESPONSE` | Receiver → Sender | Handshake result |
| `SYNC_ERROR` | Receiver → Sender | Handshake failure |
| `SYNC_HEARTBEAT` | Sender → Receiver | Verify continued alignment |
| `HEARTBEAT_ACK` | Receiver → Sender | Confirm alignment |
| `BLANKET_DISSOLVED` | Either → Either | Signal mismatch detected |
| `RESYNC_REQUEST` | Either → Either | Request re-handshake |

---

## Appendix B: Canonical Sync Tuple Example

```json
{
  "codebook_id": "ags-codebook",
  "codebook_sha256": "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
  "codebook_semver": "0.2.0",
  "kernel_version": "1.0.0",
  "tokenizer_id": "tiktoken/o200k_base"
}
```

Canonical hash of this tuple:
```
sha256(canonical_json(sync_tuple)) = "7a8b9c..."
```

---

## Appendix C: State Machine Diagram

```
                    ┌─────────────┐
                    │  UNSYNCED   │
                    └──────┬──────┘
                           │ SyncRequest
                           ▼
                    ┌─────────────┐
        ┌───────────│   PENDING   │───────────┐
        │           └──────┬──────┘           │
        │ Timeout          │ Response         │ Error
        ▼                  ▼                  ▼
 ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
 │   FAILED    │    │   SYNCED    │    │ MISMATCHED  │
 └─────────────┘    └──────┬──────┘    └──────┬──────┘
        │                  │                  │
        │ Retry            │ HashChange       │ Resync
        │                  ▼                  │
        │           ┌─────────────┐           │
        └──────────>│   PENDING   │<──────────┘
                    └─────────────┘
```

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-11 | Initial normative specification |

---

*CODEBOOK_SYNC_PROTOCOL: Establishing shared side-information through Markov blanket alignment.*
