# CODEBOOK_SYNC_PROTOCOL: Handshake and Message Shapes

**Parent Document:** [CODEBOOK_SYNC_PROTOCOL.md](CODEBOOK_SYNC_PROTOCOL.md)
**Sections:** 2, 3

---

## 2. Sync Handshake

### 2.1 Handshake Flow

```
+----------+                              +-----------+
| Sender   |                              | Receiver  |
+----+-----+                              +----+------+
     |                                        |
     |  ----- SyncRequest ----------------->  |
     |        (sender's SyncTuple)            |
     |                                        |
     |  <---- SyncResponse ----------------   |
     |        (match status + receiver's      |
     |         SyncTuple if mismatch)         |
     |                                        |
     |  IF MATCHED:                           |
     |  ----- SemanticPointer ------------>   |
     |        (pointer + context_keys)        |
     |                                        |
     |  <---- Expansion -------------------   |
     |        (canonical IR or FAIL_CLOSED)   |
     |                                        |
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
UNSYNCED --[SyncRequest]--> PENDING
PENDING  --[SyncResponse:MATCH]--> SYNCED
PENDING  --[SyncResponse:MISMATCH]--> MISMATCHED
PENDING  --[Timeout]--> FAILED
SYNCED   --[HashChange]--> MISMATCHED
SYNCED   --[SessionEnd]--> UNSYNCED
MISMATCHED --[Resync]--> PENDING
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

*Back to [CODEBOOK_SYNC_PROTOCOL.md](CODEBOOK_SYNC_PROTOCOL.md)*
