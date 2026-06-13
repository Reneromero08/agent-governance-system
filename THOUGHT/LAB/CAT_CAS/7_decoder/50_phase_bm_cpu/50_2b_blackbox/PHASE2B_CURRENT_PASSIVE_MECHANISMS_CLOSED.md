# Phase 2B Current Passive Mechanisms Closed

**Date:** 2026-06-05

## Summary

No tested passive mechanism produced a shared-substrate advantage that survived anti-ferromagnetic or mixed-sign controls. Phase 2B passive hidden-attractor computation is closed for current mechanisms. This does not close firmware/AGESA Phase 2 route or future new passive mechanisms.

## Evidence Table

| Test | Worker | Condition | Ferro Hits | Anti-Ferro Hits | Mixed Hits | Verdict |
|------|--------|-----------|------------|-----------------|------------|---------|
| 2B.2 | Random flip | Shared | 0/200 (-0.83) | — | — | NEGATIVE |
| 2B.2 | Random flip | Null | 0/200 (-3.00) | — | — | — |
| 2B.3B P1 | Ferro-bias | Shared | 200/200 (-7.00) | 0/200 (+7.00) | — | FALSIFIED |
| 2B.3B P1 | Ferro-bias | Null | 200/200 (-7.00) | 0/200 (+7.00) | — | — |
| 2B.3B P2 | Sign-aware | Shared | 200/200 (-7.00) | 200/200 (-7.00) | — | ACTIVE |
| 2B.3B P2 | Sign-aware | Null | 200/200 (-7.00) | 200/200 (-7.00) | — | — |
| 2B.4 C1 | QR subspace | Shared | 0/300 (+7.00) | 300/300 (-7.00) | 0/300 (+1.00) | FAILED |
| 2B.4 C1 | QR subspace | Null | 0/300 (+3.90) | 38/300 (-3.90) | 0/300 (+4.10) | — |
| 2B.4 C2 | Retrocausal | Shared | 0/300 (+7.00) | 300/300 (-7.00) | 0/300 (+1.00) | FAILED |
| 2B.4 C3 | Fingerprint | Shared | 0/300 (+5.72) | 169/300 (-5.52) | 0/300 (+0.11) | FAILED |
| 2B.4 C4 | DID detune | Shared | 0/300 (+7.00) | 300/300 (-7.00) | 0/300 (+0.99) | FAILED |

## Why P1 Was a False Positive

P1's "flip to align" rule converged to 200/200 on ferromagnetic because the ground state IS all-aligned. The anti-ferro acid test (J=-1, ground = anti-aligned) exposed it: the same rule produced 0/200 with worst energy (+7.00). The rule doesn't solve the Ising problem — it just aligns spins, which happens to match one specific configuration.

## Why P2 Is Active, Not Passive

P2's sign-aware edge rule works on all problem types. But shared-substrate (2 workers) performs identically to single-worker null in ALL cases (Δ=0). The shared hardware provides no advantage. P2 is a valid active local constraint solver, useful as a Phase 3 catalytic primitive, not passive hidden-attractor evidence.

## Why Shared Substrate Advantage Is Not Supported

Across all channel matrix tests (C1-C4), shared-substrate conditions either performed WORSE than single-worker nulls (ferro, mixed) or tied (anti-ferro). The apparent anti-ferro shared advantage was caused by the base rule matching the problem, not by hardware-mediated coupling. No channel created genuine shared-substrate advantage across problem types.

## What Remains Open

- Firmware/AGESA Phase 2 route (BIOS P4 VID manipulation)
- Future genuinely new passive mechanisms
- Active catalytic Ising solver (promoted to Phase 3.13)
- Physical GOE eigenvalue measurement (Phase 4.4B, pending Phase 2 physical phase channel)

## Contamination Checklist

- [x] All workers verified: no J_ij access, no local field, no energy computation, no Metropolis, no hidden solver
- [x] Energy scored only after fork'd workers exit
- [x] Nulls constructed with matched total operation counts
- [x] Anti-ferro acid test applied to all mechanisms
- [x] Mixed-sign test applied to all channel matrix mechanisms

## Status Labels

```
PHASE2B_2_PASSIVE_RANDOM_NEGATIVE
PHASE2B_3B_P1_FERRO_BIAS_FALSIFIED
PHASE2B_3B_P2_ACTIVE_EDGE_SOLVER_WORKING
PHASE2B_3B_SHARED_SUBSTRATE_NO_ADVANTAGE
PHASE2B_4_CHANNEL_MATRIX_PASSIVE_NULLS_FAILED
PHASE2B_CURRENT_PASSIVE_MECHANISMS_CLOSED
PHASE3_ACTIVE_CATALYTIC_ISING_PROMOTED
```
