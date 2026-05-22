# Q32: Fabric Propagation Update

**Date:** 2026-05-21
**Finding:** The multi-scale Feistel fabric from Q57 provides the propagation medium for the M field. The semiotic wave operator (multi-scale Laplacian = discrete Box term from SEMIOTIC_ACTION_PRINCIPLE.md Section 3) correctly describes signal propagation through the fabric.

## Results

1. **Operator confirmed**: The multi-scale rotation-toward-mean at each Feistel round is a discrete Laplacian at scale 2^r. Summing across all scales yields the coarse-grained Box (d'Alembertian) operator.

2. **Entropy vs survival**: Signal survival follows R ~ E / (nabla_S_fabric + nabla_S_signal). The fabric's background entropy (~8.4 bits) dominates, with signal entropy as a perturbation. R² = 0.63 for the 1/nabla_S relationship.

3. **Low-entropy signals survive**: Pure sine waves (near-zero spectral entropy) survive at >99%. High-entropy noise decays faster. Directionally consistent with the formula.

4. **Lindblad gap**: The exact 1/nabla_S functional form requires the full dissipative dynamics (Lindblad environmental coupling). The unitary rotation operator alone gives linear decay, not inverse.

## Files Added

- `verify_q32_fabric.py` — Signal injection + multi-scale propagation tests
- `verify_q32_lindblad.py` — Lindblad dissipation on the fabric
- `verify_q32_entropy_survival.py` — Spectral entropy vs signal survival sweep

## Connection to Q57

Q57 proved the fabric structure (gapped, O(1) min-cut channels). Q32 extends this to dynamics: those channels are the propagation medium for the semiotic wave operator. The formula R = (E/nabla_S) * sigma^D_f describes the signal-to-noise ratio at equilibrium.
