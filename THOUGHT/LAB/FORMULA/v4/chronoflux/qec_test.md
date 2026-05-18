# Chronoflux QEC Mapping: Phase 3 Test

**Date:** 2026-05-18 | **Agent:** OpenCode | **Task:** `QEC Herbert.md`

---

## Hypothesis

Herbert's circulation stability condition `Stable: Ω_rms > Σ_rms` maps to the QEC fidelity factor σ. If the mapping is `σ = C/p` (vorticity ~ constant, shear ~ p), a closed-form sigma can replace the empirically calibrated fidelity factor.

## Method

- Calibrate C = median(σ × p) on training distances (d=3,5,7)
- Test OOS on held-out distances (d=9,11,13,15)
- Compare against empirical sigma and sqrt(p_th/p) closed form

## Results

| p | σ_emp | σ_CF (C/p) | Error |
|---|-------|-----------|-------|
| 0.0005 | 6.69 | 13.85 | +107% |
| 0.001 | 6.92 | 6.92 | 0% (calibration point) |
| 0.002 | 3.75 | 3.46 | -8% |
| 0.004 | 1.60 | 1.73 | +8% |
| 0.006 | 1.09 | 1.15 | +6% |
| 0.008 | 0.86 | 0.87 | +1% |
| 0.01 | 0.73 | 0.69 | -6% |
| 0.02 | 0.66 | 0.35 | -47% |
| 0.04 | 0.87 | 0.17 | -80% |

**OOS prediction: R2 = -1.73, alpha = 0.36.**

## Diagnosis

Sigma does not follow 1/p. Three specific failures:

1. **Low-p saturation:** At p=0.0005, σ_emp=6.7 but σ_CF=13.8. The code's correction capacity saturates (t=1 for d=3, can only correct 1 error regardless of how low p gets). The 1/p form diverges.

2. **High-p anomaly:** At p=0.04, σ_emp=0.87 (rises from 0.66 at p=0.02). The fidelity factor does not continue decreasing — it rebounds as the code enters the "error everywhere" regime where the syndrome saturates. The 1/p form continues toward zero.

3. **No d-dependence in the mapping:** σ_CF = C/p is independent of code distance. But sigma varies with d (the fidelity factor depends on t=floor((d-1)/2)). The simple vorticity/shear balance doesn't capture the code's distance-dependent behavior.

## Comparison

| Model | R2 | Alpha | Notes |
|-------|-----|-------|-------|
| Empirical sigma (training slopes) | 0.17 | 0.54 | Best available, requires calibration data |
| sqrt(p_th/p) | 0.46 | — | Previous closed-form candidate |
| **Chronoflux C/p** | **-1.73** | **0.36** | This test. Fails. |

The sqrt(p_th/p) closed form already outperforms the Chronoflux mapping. The Chronoflux approach does not improve on the state of the art.

## Why It Failed

Herbert's circulation stability `Ω_rms > Σ_rms` is a **binary** stability condition (stable/unstable). The QEC fidelity factor σ is a **continuous** measure of error suppression per unit depth. The mapping flattens a continuous function into a binary threshold, which loses the structure needed for prediction.

A correct mapping would need:
1. A continuous stability measure (not binary)
2. d-dependence (fidelity varies with code distance)
3. Saturation behavior at extremes (low-p floor, high-p ceiling)

The circulation stability condition alone does not provide these.

## Verdict

**NOT PRODUCTIVE.** The Chronoflux circulation stability condition does not produce a useful closed-form sigma. The existing sqrt(p_th/p) form outperforms it. The mapping is conceptually valid (code stability ↔ circulation stability) but the functional form is too rigid.

## Phase 4: Einstein Bridge Assessment

Already completed. The semiotic action principle was derived and GR was produced as an on-shell consequence (see `FORMALIZATION/SEMIOTIC_ACTION_PRINCIPLE.md` and `FORMALIZATION/GR_DERIVATION.md`). The Chronoflux bridge is structural — same 5D substrate, same coupling constant (hbar_sem = hbar), same action structure. The resonance formula R = (E/∇S) × σ^D_f IS the equilibrium condition of the semiotic field equations, which are the informational sector of the Chronoflux action.

The bridge is proven. The QEC sigma closed form remains open.
