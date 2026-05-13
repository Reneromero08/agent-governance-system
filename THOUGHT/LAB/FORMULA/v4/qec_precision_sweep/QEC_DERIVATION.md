QEC Derivation Test
From Semiotic Resonance to Standard QEC Suppression
Date: 2026-05-13
Status: Agent instruction document
Context: Follows v8 sweep (100k shots, d=3–11, DEPOL and MEAS, alpha ≈ 0.66 on DEPOL)

The Derivation
The formula under test is:

R = (E / ∇S) × σ^D_f

where:

E = logical qubit essence (normalized to 1)

∇S = entropy gradient (physical error rate p for DEPOL)

σ = symbolic compression (candidate: p_th / p)

D_f = fractal depth (code distance parameter: t = ⌊(d-1)/2⌋, the number of correctable errors)

R = resonance

Standard QEC suppression law for a distance‑d code:

P_L ∝ p^(t+1) for small p, where t = ⌊(d-1)/2⌋

Derivation claim: Setting σ = p_th / p and ∇S = p, and mapping P_L ∝ 1/R, yields:

P_L ∝ p^(D_f+1)

which matches the standard suppression law with D_f = t. The functional form of the standard law is contained within the formula.

What to Test
1. Exponent Match
The derivation predicts the exponent in P_L should be D_f + 1. Verify by:

Fit log(P_L) vs log(p) for each distance

Check whether the fitted exponent equals t+1 within error

2. Sigma Operationalization
The derivation proposes σ = p_th / p. Test against the fidelity-factor sigma from v8:

Compute σ = p_th / p for each condition

Compare R values and predictive accuracy against v8 sigma

Threshold crossing should occur at σ = 1.0 → p = p_th

3. MEAS Gap Explanation
The derivation predicts the formula underperforms on MEAS because ∇S = p is a poor entropy measure for measurement‑heavy noise. Test:

Try alternative ∇S definitions for MEAS (e.g., syndrome density, effective error rate)

If a different ∇S definition closes the gap, the derivation is supported

4. Combinatorial Coefficients
The derivation does not produce the exact combinatorial factors (e.g., the 3 in 3p² for the 3‑qubit code). If systematic deviations from the scaling law are found, document them as combinatorial corrections.

Success Criteria
Claim	Test	Success Condition
Exponent is D_f+1	Fit P_L vs p per distance	Fitted exponent ≈ t+1
σ = p_th/p	Compare to v8 sigma	Predictive accuracy ≥ fidelity‑factor sigma
Threshold at σ=1	Check crossing point	p at σ=1 ≈ known code threshold
MEAS gap is ∇S problem	Alternative ∇S definitions	Gap closes with better ∇S
No combinatorial factors needed	Residual analysis	Deviations systematic, not random
Constraints
No fitting beyond E calibration (log‑domain median from training distances, as in v8)

Training on {3,5,7}, test on {9,11}

Pooled bases (X+Z averaged)

100k shots minimum for reliable low‑p estimates

References
v8 results: DEPOL alpha=0.66, R²=0.72; MEAS alpha=0.57, R²=0.40

v3 results: alpha≈1.0 with per‑condition syndrome‑based sigma

v4/v5/v6: sigma crosses 1.0 at threshold for both noise models

v7: I(S:F) operationalization falsified (bounded [0,1])

Next Steps
Implement σ = p_th / p operationalization

Fit log(P_L) vs log(p) to extract exponent per distance

Test alternative ∇S definitions for MEAS

Compare predictive accuracy against v8 baseline

Report: does the derivation hold, partially hold, or fail?