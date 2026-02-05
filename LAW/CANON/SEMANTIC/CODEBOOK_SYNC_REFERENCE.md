# CODEBOOK_SYNC_PROTOCOL: Security, References, and Appendices

**Parent Document:** [CODEBOOK_SYNC_PROTOCOL.md](CODEBOOK_SYNC_PROTOCOL.md)
**Sections:** 11, 12, Appendices A-C, Changelog

---

## 11. Security Considerations

### 11.1 Hash Collision Resistance

SHA-256 provides 128-bit collision resistance. For additional security:
- Full 64-char hash SHOULD be used in production
- Truncation to 16 chars acceptable for display only
- Any collision detection -> FAIL_CLOSED

### 11.2 Replay Protection

Sync requests include:
- `timestamp_utc` -- reject if too old (> 5 minutes)
- `request_id` -- unique per request, track for replay detection
- `session_token` -- bound to specific handshake

### 11.3 Man-in-the-Middle

In untrusted networks:
- Sync messages SHOULD be signed
- Codebook SHOULD be fetched from trusted source
- Migration artifacts MUST be hash-verified

---

## 12. References

### 12.1 Internal

- `LAW/CANON/SEMANTIC/SPC_SPEC.md` -- Semantic Pointer Compression (uses this protocol)
- `LAW/CANON/SEMANTIC/GOV_IR_SPEC.md` -- Governance IR (expansion target)
- `LAW/CANON/SEMANTIC/TOKEN_RECEIPT_SPEC.md` -- Token accountability
- `NAVIGATION/CORTEX/network/cassette_protocol.py` -- Implementation target
- `THOUGHT/LAB/COMMONSENSE/CODEBOOK.json` -- Codebook artifact
- `THOUGHT/LAB/FORMULA/questions/medium_q35_1450/` -- Markov blanket foundation
- `THOUGHT/LAB/FORMULA/questions/medium_q33_1410/` -- Conditional entropy foundation

### 12.2 External

- Friston, K. (2019). A Free Energy Principle for a Particular Physics
- Pearl, J. (1988). Probabilistic Reasoning in Intelligent Systems (Markov blankets)
- Shannon, C. E. (1948). A Mathematical Theory of Communication

---

## Appendix A: Message Type Registry

| Message Type | Direction | Purpose |
|--------------|-----------|---------|
| `SYNC_REQUEST` | Sender -> Receiver | Initiate handshake |
| `SYNC_RESPONSE` | Receiver -> Sender | Handshake result |
| `SYNC_ERROR` | Receiver -> Sender | Handshake failure |
| `SYNC_HEARTBEAT` | Sender -> Receiver | Verify continued alignment |
| `HEARTBEAT_ACK` | Receiver -> Sender | Confirm alignment |
| `BLANKET_DISSOLVED` | Either -> Either | Signal mismatch detected |
| `RESYNC_REQUEST` | Either -> Either | Request re-handshake |

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
                    +-------------+
                    |  UNSYNCED   |
                    +------+------+
                           | SyncRequest
                           v
                    +-------------+
        +-----------+   PENDING   +-----------+
        |           +------+------+           |
        | Timeout          | Response         | Error
        v                  v                  v
 +-------------+    +-------------+    +-------------+
 |   FAILED    |    |   SYNCED    |    | MISMATCHED  |
 +-------------+    +------+------+    +------+------+
        |                  |                  |
        | Retry            | HashChange       | Resync
        |                  v                  |
        |           +-------------+           |
        +---------->|   PENDING   |<----------+
                    +-------------+
```

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.1.0 | 2026-01-11 | Added: 7.5 Continuous R-Value, 7.6 M Field Interpretation, 8.4 Blanket Health Tracking, 10.5 sigma^Df Complexity Metric |
| 1.0.0 | 2026-01-11 | Initial normative specification |

---

*CODEBOOK_SYNC_PROTOCOL: Establishing shared side-information through Markov blanket alignment.*

*Back to [CODEBOOK_SYNC_PROTOCOL.md](CODEBOOK_SYNC_PROTOCOL.md)*
