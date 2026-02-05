# CODEBOOK_SYNC_PROTOCOL: Semantic Side-Information Synchronization

**Canon ID:** SEMANTIC-SYNC-001
**Version:** 1.1.0
**Status:** NORMATIVE
**Created:** 2026-01-11
**Phase:** 5.3.3

---

## Abstract

This specification defines the Codebook Sync Protocol -- the mechanism by which sender and receiver establish shared side-information for Semantic Pointer Compression (SPC).

**Core insight:** The sync protocol defines a **Markov blanket** between communicating parties. The codebook state represents the boundary through which semantic pointers can flow. When boundaries align (hashes match), information transfers deterministically. When they diverge, the system fails closed until resync.

**Information-theoretic foundation:**
```
H(X|S) ~ 0  when S contains the expansion of X

Where:
  S = shared codebook state (side-information)
  X = semantic pointer
  H(X|S) = bits needed to decode X given S
```

The sync protocol establishes S. Without sync, H(X|S) = H(X) -- no compression benefit.

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

1. **Fail-Closed** -- Any mismatch rejects; no silent degradation
2. **Explicit Sync** -- Shared state must be verified before symbol expansion
3. **Deterministic** -- Same sync tuple always produces same result
4. **Boundary Preservation** -- Markov blanket maintained through continuous verification

---

## Document Parts

This specification is organized into the following sub-documents:

| Part | Document | Sections |
|------|----------|----------|
| Handshake | [CODEBOOK_SYNC_HANDSHAKE.md](CODEBOOK_SYNC_HANDSHAKE.md) | 2. Sync Handshake, 3. Message Shapes |
| Failures | [CODEBOOK_SYNC_FAILURES.md](CODEBOOK_SYNC_FAILURES.md) | 4. Failure Codes, 5. Compatibility Policy |
| Cassette | [CODEBOOK_SYNC_CASSETTE.md](CODEBOOK_SYNC_CASSETTE.md) | 6. Cassette Network Integration |
| Blanket | [CODEBOOK_SYNC_BLANKET.md](CODEBOOK_SYNC_BLANKET.md) | 7. Markov Blanket Semantics |
| Session | [CODEBOOK_SYNC_SESSION.md](CODEBOOK_SYNC_SESSION.md) | 8. Session Management |
| Implementation | [CODEBOOK_SYNC_IMPLEMENTATION.md](CODEBOOK_SYNC_IMPLEMENTATION.md) | 9. Implementation Notes, 10. Info-Theoretic Semantics |
| Reference | [CODEBOOK_SYNC_REFERENCE.md](CODEBOOK_SYNC_REFERENCE.md) | 11. Security, 12. References, Appendices A-C, Changelog |

---

*CODEBOOK_SYNC_PROTOCOL: Establishing shared side-information through Markov blanket alignment.*
