# CODEBOOK_SYNC_PROTOCOL: Failure Codes and Compatibility Policy

**Parent Document:** [CODEBOOK_SYNC_PROTOCOL.md](CODEBOOK_SYNC_PROTOCOL.md)
**Sections:** 4, 5

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

Optional explicit compatibility ranges may be declared. These are NEVER silent -- migration step is always explicit.

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
    {"type": "add_symbol", "symbol": "new", "path": "LAW/CANON/NEW"},
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

*Back to [CODEBOOK_SYNC_PROTOCOL.md](CODEBOOK_SYNC_PROTOCOL.md)*
