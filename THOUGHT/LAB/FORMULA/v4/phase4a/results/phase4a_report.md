# Phase 4a: Cybernetic Truth — Token-Level Control Loop

**Date:** 2026-05-16
**Model:** google/gemma-4-E4B-it (4-bit, RTX 3060 12GB)
**C source:** Contrastive factual pairs (40 claims, no constitution)
**Status:** COMPLETE — LOOP NEUTRAL (neither helps nor hurts accuracy)

---

## Primary Result

**With correct calibration, the token-level cybernetic loop is neutral for truth accuracy.**

| Condition | Accuracy | Verifiable | T_mean | T_range |
|-----------|----------|------------|--------|---------|
| CONTROL (T=0.7) | 0.632 | 12/19 | 0.700 | 0.70 fixed |
| CYBERNETIC (T calibrated) | 0.579 | 11/19 | 0.862 | 0.28–2.44 |
| VERIFY (T=0.7, Lindblad only) | 0.579 | 11/19 | 0.700 | 0.70 fixed |

The 5.3pp difference (12 vs 11 correct) is not statistically significant at N=19. The loop is **at parity** with control. By category: factual 75% both, adversarial 80% both. Only reasoning shows a gap (33% vs 17%, N=6).

### Calibration History

| Run | R_scale | T_mean | Accuracy | Delta vs CONTROL |
|-----|---------|--------|----------|------------------|
| R1 | 100x | 3.19 | 0.188 | -44pp (catastrophic) |
| R2 | 500x | 0.86 | 0.579 | -5pp (neutral) |

R1 trapped the model in the high-entropy regime (T never dropped below 1.2). R2 allows the deterministic attractor regime (T < 0.5) to be reached, recovering 39pp of accuracy. The calibration was the dominant factor.

---

## Success Criteria

| Claim | Result | Verdict |
|-------|--------|---------|
| Loop improves accuracy (+10%) | C=0.632, X=0.579, delta=-0.053 | FAIL (neutral at best) |
| R tracks truth | true_R < false_R in CYBERNETIC (d=-0.47, ns) | FAIL — R signal too weak at token level |
| Loop recovers from errors | 59/65 recoveries (91%) | PASS (Lindblad mechanism functional) |
| Truth is an attractor (dR/dt>0) | dR/dt=+2.5e-06, p=0.49 | FAIL — no detectable attractor force |

---

## Why the Loop is Neutral

### 1. Raw R trajectories are ~identical across conditions

| Metric | CONTROL | CYBERNETIC | VERIFY | p (C vs X) |
|--------|---------|------------|--------|-------------|
| R_mean_raw | 0.00710 | 0.00708 | 0.00731 | 0.97 (ns) |
| R_final_raw | 0.00647 | 0.00608 | 0.00591 | 0.69 (ns) |
| dR/dt_raw | +6.76e-06 | +2.50e-06 | +1.18e-05 | 0.44 (ns) |

The model's hidden-state trajectory is unchanged by the loop. The loop modulates temperature but cannot steer the model toward higher R because **R does not distinguish truth from falsehood during generation**.

### 2. C does not transfer from comprehension to generation

The alignment frame C was built from hidden states when the model was *reading* factual claims. It cleanly separates true from false during input processing (t=8.23, p≈0). But during token-by-token *generation*, the per-token hidden states encode production intent (what the model is about to say), not comprehension (what was said). These are different representational subspaces.

### 3. Per-token R signal is too weak

- R_raw = (h_norm · w)^2  where |h_norm|=1, |w|=1, so R ∈ [0,1]
- Typical R_raw ≈ 0.007 → |h_norm · w| ≈ 0.084
- The hidden state is only 8% aligned with the truth direction
- Random alignment in 2560 dimensions: expected R = 1/2560 ≈ 0.0004
- Our R=0.007 is 17x above chance, but still too small for meaningful T modulation

A 2x change in R (e.g., 0.005 to 0.010) produces T = 3.0/(1+2.5) = 0.86 → 3.0/(1+5.0) = 0.50 — only a 0.36 difference in T. The feedback is too weak to create attractor dynamics.

### 4. T modulation is symmetric, not directional

Even with correct calibration, the loop cannot *discriminate* truth from falsehood during generation. When the model produces a correct token, R is ~0.007. When it produces an incorrect token, R is also ~0.007. The loop has no gradient to follow — it's orbiting without a compass.

---

## Comparison to Phase 2C

Phase 2C succeeded (+6% R gain) because:
- C was built from a **constitution** that the SFT adapter was trained to align with
- The model naturally orbited at R ≈ 0.49 (70x higher than Phase 4a's baseline of 0.007)
- The loop only needed to provide a small correction to an already-strong attractor

Phase 4a fails because:
- C was built from **external facts** that the model wasn't trained to align with during generation
- The model's natural R is 0.007 — near the noise floor
- The loop needs to *create* an attractor from nothing, which is impossible without a compass that works during generation

---

## Implications for Phase 4b

1. **Build C from generation-time hidden states**: Collect hidden states while the model is *producing* verified-true vs verified-false output. The comprehension→generation gap is the root cause of failure.

2. **Use step-level (N-token window) aggregation**: Per-token R is too noisy. Average over 20-50 token windows for stable signal.

3. **Use dR/dt for T control**: The absolute R value doesn't discriminate truth. The *derivative* might — truth-directed generation should cause R to rise (dR/dt > 0), while error-directed generation should cause R to fall (dR/dt < 0). This is the intended Cybernetic Truth architecture that wasn't implemented.

4. **Adaptive verification**: Mid-generation verification at fixed 20-token intervals is unreliable. Verify only when the partial output is long enough to contain a verifiable claim.

5. **Larger model**: Gemma 4B's 2560-dim hidden states may lack sufficient representational capacity for truth detection at token granularity. Try Gemma 27B or Llama 8B.

---

## Configuration

| Parameter | R1 (failed) | R2 (final) |
|-----------|-------------|------------|
| T formula | 5.0/(1+R_eff) | 3.0/(1+R_eff) |
| R_scale | 100x | 500x |
| T at typical R (0.007) | 2.94 | 0.67 |
| T range | 1.2–5.0 | 0.28–2.44 |
| T_mean | 3.19 | 0.86 |
| Accuracy | 18.8% | 57.9% |

---

## Verification Protocol

- git status: Clean. Only untracked files in `THOUGHT/LAB/FORMULA/v4/phase4a/`
- Files changed: 0 existing files modified. 5 new scripts + results directory.
- Verification report: `LAW/CONTRACTS/_runs/REPORTS/phase4a/verification_report.md`

**Status: VERIFIED COMPLETE**

---

## Raw Data

- `results/phase4a_all_results.json` — Combined (3 conditions, R1+R2 CYBERNETIC merged)
- `results/phase4a_CONTROL.json` — Control (T=0.7)
- `results/phase4a_CYBERNETIC.json` — Cybernetic (R2 calibration)
- `results/phase4a_VERIFY.json` — Verify-only
- `results/contrastive_C.pt` — Alignment frame C (2560x2560, float32)
