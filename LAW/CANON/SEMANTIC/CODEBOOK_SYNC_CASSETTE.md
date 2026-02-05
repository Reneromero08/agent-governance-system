# CODEBOOK_SYNC_PROTOCOL: Cassette Network Integration

**Parent Document:** [CODEBOOK_SYNC_PROTOCOL.md](CODEBOOK_SYNC_PROTOCOL.md)
**Section:** 6

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
Hub <---- SyncRequest ---- Spoke-1
Hub ----> SyncResponse ---> Spoke-1
Hub <---- SyncRequest ---- Spoke-2
Hub ----> SyncResponse ---> Spoke-2
...
Hub ----> BroadcastSync --> All Spokes (on codebook update)
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

*Back to [CODEBOOK_SYNC_PROTOCOL.md](CODEBOOK_SYNC_PROTOCOL.md)*
