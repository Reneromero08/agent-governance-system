# CODEBOOK_SYNC_PROTOCOL: Markov Blanket Semantics

**Parent Document:** [CODEBOOK_SYNC_PROTOCOL.md](CODEBOOK_SYNC_PROTOCOL.md)
**Section:** 7

---

## 7. Markov Blanket Semantics

### 7.1 Theoretical Foundation

The sync protocol formalizes a **Markov blanket** in the semiotic space:

```
+-----------------------------------------------------+
|                  External World                      |
|   (unknown symbols, drift, unshared context)         |
+------------------------+----------------------------+
                         |
                 +-------+--------+
                 | Markov Blanket  |  <-- Sync Protocol
                 | (sync_tuple)    |
                 +-------+--------+
                         |
+------------------------+----------------------------+
|                 Shared Semantic Space                |
|   (codebook, kernel, tokenizer -- deterministic)    |
+-----------------------------------------------------+
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
H(expansion | pointer, misaligned_blanket) = undefined -> FAIL_CLOSED
```

### 7.3 Active Inference Interpretation

The sync protocol implements **Active Inference** at the protocol level:

1. **Prediction:** Sender predicts receiver has matching codebook
2. **Verification:** Handshake tests prediction
3. **Error Signal:** Mismatch = prediction error
4. **Action:** Resync to minimize prediction error

This is R-gating (per Q35): R > tau permits semantic transfer; R < tau requires resync.

### 7.4 Blanket Status Semantics

| Status | R-value | Interpretation |
|--------|---------|----------------|
| `ALIGNED` | R > tau | Stable blanket, semantic transfer permitted |
| `DISSOLVED` | R < tau | Blanket broken, resync required |
| `PENDING` | R ~ tau | Boundary forming, awaiting confirmation |

### 7.5 Continuous R-Value (Extended)

The binary ALIGNED/DISSOLVED status provides a hard gate, but a continuous R-value enables gradient-based diagnostics and predictive health tracking.

**Formula:**
```
R = gate(codebook_sha256) x (Sum_i w_i * score(field_i)) / (Sum_i w_i)

Where:
  gate(codebook_sha256) = 1 if exact match, 0 otherwise (hard requirement)
  score(field) = compatibility_score(sender.field, receiver.field) in [0, 1]
```

**Default Weights:**

| Field | Weight | Rationale |
|-------|--------|-----------|
| `kernel_version` | 1.0 | Processing compatibility (major must match) |
| `codebook_semver` | 0.7 | Version compatibility (migration path) |
| `tokenizer_id` | 0.5 | H(X\|S) measurement (not semantic content) |

**Compatibility Scoring:**
```python
def score_kernel_version(sender: str, receiver: str) -> float:
    """Score kernel version compatibility (semver)."""
    s_major, s_minor, s_patch = parse_semver(sender)
    r_major, r_minor, r_patch = parse_semver(receiver)

    if s_major != r_major:
        return 0.0  # Major mismatch = incompatible
    if s_minor != r_minor:
        return 0.7  # Minor mismatch = compatible but warn
    if s_patch != r_patch:
        return 0.9  # Patch mismatch = essentially compatible
    return 1.0  # Exact match

def score_tokenizer_id(sender: str, receiver: str) -> float:
    """Score tokenizer compatibility."""
    if sender == receiver:
        return 1.0
    # Known compatible families
    if same_tokenizer_family(sender, receiver):
        return 0.8
    return 0.0  # Unknown compatibility = fail-closed

def compute_continuous_r(sender_tuple: dict, receiver_tuple: dict) -> float:
    """Compute continuous R-value with weighted fields."""
    # Hard gate: codebook hash must match
    if sender_tuple['codebook_sha256'] != receiver_tuple['codebook_sha256']:
        return 0.0

    weights = {'kernel_version': 1.0, 'codebook_semver': 0.7, 'tokenizer_id': 0.5}
    score_funcs = {
        'kernel_version': score_kernel_version,
        'codebook_semver': score_kernel_version,  # Reuses semver comparison logic
        'tokenizer_id': score_tokenizer_id
    }

    weighted_sum = sum(
        weights[f] * score_funcs[f](sender_tuple[f], receiver_tuple[f])
        for f in weights
    )
    return weighted_sum / sum(weights.values())
```

**Threshold Interpretation:**

| R Range | Status | Interpretation |
|---------|--------|----------------|
| R = 1.0 | ALIGNED | Perfect match, full semantic transfer |
| 0.8 <= R < 1.0 | ALIGNED (warn) | Compatible but minor differences |
| 0.5 <= R < 0.8 | PENDING | Marginal alignment, consider resync |
| R < 0.5 | DISSOLVED | Insufficient alignment, resync required |

### 7.6 M Field Interpretation (Theoretical)

The sync protocol has a natural interpretation in terms of the M field (per Q32):

```
dB = Markov blanket boundary (where grad(M) is discontinuous)
S = M|dB (shared side-information is M restricted to boundary)
```

The sync_tuple is a discrete approximation to M|dB -- it captures the meaning field at the boundary between sender and receiver semantic spaces.

**Correspondence:**
| Protocol Concept | Field-Theoretic Analog |
|------------------|------------------------|
| sync_tuple | M\|dB (field at boundary) |
| codebook_sha256 | Hash of M configuration |
| ALIGNED | grad(M) continuous across dB |
| DISSOLVED | grad(M) discontinuous (barrier) |
| Handshake | Probing M at boundary |

**Future Direction (Q32):** Continuous M field dynamics would allow gradient flow across blankets, smooth alignment transitions, and field-theoretic formalization of semantic drift.

---

*Back to [CODEBOOK_SYNC_PROTOCOL.md](CODEBOOK_SYNC_PROTOCOL.md)*
