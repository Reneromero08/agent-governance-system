# Question 35: Markov Blankets & System Boundaries (R: 1450)

**STATUS: ✅ ANSWERED (2026-01-11)**

## Question
How do Markov blankets (boundaries that separate systems while allowing information exchange) relate to R-gating? Do R-gates define Markov blankets in the semiosphere?

**Concretely:**
- ✅ Does R > threshold define a Markov blanket boundary?
- ✅ How does Active Inference (minimizing surprise through action) connect to R-gating?
- ⏳ Can we formalize "meaning boundaries" as Markov blankets on the M field? (→ Q32)

---

## Resolution: CODEBOOK_SYNC_PROTOCOL.md (Phase 5.3.3)

**Reference:** `LAW/CANON/SEMANTIC/CODEBOOK_SYNC_PROTOCOL.md`

### R-gating = Blanket Status (Section 7.4)

| Status | R-value | Interpretation |
|--------|---------|----------------|
| `ALIGNED` | R > τ | Stable blanket, semantic transfer permitted |
| `DISSOLVED` | R < τ | Blanket broken, resync required |
| `PENDING` | R ≈ τ | Boundary forming, awaiting confirmation |

**Key insight:** τ is operationalized as sync_tuple match:
```
sync_tuple = (codebook_sha256, kernel_version, tokenizer_id)
match → ALIGNED;  mismatch → DISSOLVED
```

### Active Inference Implementation (Section 7.3)

The handshake **IS** Active Inference at protocol level:
```
1. PREDICTION:    Sender predicts receiver has matching codebook
2. VERIFICATION:  Handshake tests prediction (SYNC_REQUEST → SYNC_RESPONSE)
3. ERROR SIGNAL:  Mismatch = prediction error (SYNC_ERROR)
4. ACTION:        Resync to minimize prediction error
```

Agents maintain blanket by: heartbeats, resync on mismatch, refuse ops when DISSOLVED.

### Information-Theoretic Value (Section 10)

```
Without alignment:  H(X) = full expansion required
With alignment:     H(X|S) ≈ 0 (pointer suffices)
Gain:               I(X;S) = H(X) - H(X|S)
```

---

## Why This Matters

**Connection to Free Energy Principle (Q9):**
- ✅ R-gating IS the decision boundary for Markov blanket formation
- ✅ Active Inference = acting to keep R high (heartbeat + resync)
- Note: R↔F derivation (`R ∝ exp(-F)`) is Q9's domain, not Q35

**Connection to Q32 (Meaning Field):**
- ⏳ M field boundary conditions still theoretical
- Markov blankets define where meaning basins begin/end
- This remains open for Q32

**Connection to Q33 (Conditional Entropy):**
- ✅ CDR = σ^Df measurable only with aligned blankets
- ✅ Without alignment, CDR undefined

**Connection to Q34 (Platonic Convergence):**
- ✅ Sync protocol enforces convergence (exact match required)

---

## Hypotheses (CONFIRMED)

- ✅ R > τ defines a "stable Markov blanket" (ALIGNED state)
- ✅ R < τ means blanket is dissolving (DISSOLVED state)
- ⏳ M field dynamics = evolution of blanket structure (→ Q32)

---

## Tests

1. **Blanket Formation Test:** ✅ IMPLEMENTABLE via handshake sequence
2. **Active Inference Test:** ✅ IMPLEMENTED via heartbeat + resync
3. **Boundary Stability Test:** ✅ IMPLEMENTABLE via TTL + drift detection

---

## Original Open Questions — Status

| Question | Status | Notes |
|----------|--------|-------|
| Is R-gating equivalent to blanket maintenance? | ✅ YES | Section 7.3-7.4 |
| How does σ^Df relate to blanket complexity? | ✅ Partial | CDR requires alignment; complexity measure → Q33 |
| Can we derive R from FEP + Markov blankets? | → Q9 | Already answered: `R ∝ exp(-F)` |

---

## Related Work
- Karl Friston: Free Energy Principle, Active Inference
- Markov Blankets in biology (Michael Levin)
- Predictive Processing frameworks
- **AGS:** `LAW/CANON/SEMANTIC/CODEBOOK_SYNC_PROTOCOL.md`
