# 01 Scientific-Law Adversary - Feynman

Agent/thread ID: `019f5dfd-060f-7012-9400-ea090c99b287`

Model: `gpt-5.6-sol`

Reasoning effort: `max`

Role: Subagent 1, Scientific-Law Adversary

SUBAGENT_ROLE  
Scientific-Law Adversary. Read-only exact-object review of local `main` at `f6ef90374de424723e0edba34786778e8e3f1a29`; source locations below are relative to the package root.

VERDICT: MATERIAL_BLOCKER

CONFIDENCE: HIGH

MATERIAL_FINDINGS:

SLAW-01  
SOURCE: `gain_covariant_confirmation_public.py:704-710` (`rel_error`, `complex_rel_error`), `:977-992` (`estimate_control_gain`), `:1022-1037`.  
MECHANISM: The count-valued floor `152` is reused for dimensionless gain agreement. Therefore `|g_post-g_equal|` is divided by `152` for realistic gains, making the nominal 25% agreement gate nearly vacuous. Arithmetic averaging then centers mutually incompatible controls.  
CONSEQUENCE: Executed counterexample: `g_post=1.000234973`, `g_equal=39`, reported error `0.249998454`, and target/fold/polarity gain `20` produced `CONFIRMED`. The one-count nonlinear response `F(q)=sign(q)` also produced `CONFIRMED` with `g_control=0.000710819`. The existing disagreement mock itself reports `control_gain_agreement=true`; it rejects only because its target was not changed to the mean.  
MINIMAL_REPAIR: Define domain-specific errors. Gain agreement must use a dimensionless denominator, with no count floor. Estimate one predeclared common slope only after both control residuals pass; specify weighting prospectively.  
REQUIRED_REGRESSION: The 39:1 construction and one-count `sign(q)` construction must fail specifically at named gain/geometry gates, not merely produce any non-confirmed class.  
MUST_REPAIR_BEFORE_LIVE: true

SLAW-02  
SOURCE: `FINAL_CONFIRMATION_CONTRACT.md:115,141-168`; `gain_covariant_confirmation_public.py:31-32,1028-1047`.  
MECHANISM: The target odd floor scales as `456*g_control`, while the null ceiling remains an absolute `152`. For `g_control<1/3`, the required target odd signal can be smaller than an accepted null. No minimum effect-size or SNR law closes this.  
CONSEQUENCE: Executed integer-count counterexample: rounded `q/10` gave `g_control=0.100219`, target odd amplitude `82`, scaled floor `45.70`, and a source-off decoded imaginary null of `100`. Every law passed and the result was `CONFIRMED`, although the accepted null exceeded the target odd signal.  
MINIMAL_REPAIR: Require a prospectively justified absolute/SNR floor in addition to normalized geometry, for example `odd > max(456*g_control, 3*152)`, or freeze an equivalent minimum-gain and realized-null separation law.  
REQUIRED_REGRESSION: The `g~0.1`, target-odd `82`, source-off-null `100` packet must fail specifically at the absolute/SNR gate.  
MUST_REPAIR_BEFORE_LIVE: true

SLAW-03  
SOURCE: `gain_covariant_confirmation_public.py:240-268,709-710,967-974,1022-1036`.  
MECHANISM: Both gain controls are collinear real-axis probes at amplitudes approximately `1298` and `1536`; they do not probe the target's `822` quadrature. The symmetric componentwise metric permits an observed component to be `4/3` of prediction while calling the error 25%, and has no explicit phase gate.  
CONSEQUENCE: Analytically, a `1.32` imaginary gain is reported as `0.32/1.32=0.2424` and shifts phase by `7.55 deg`. Executed stronger counterexample with real/imaginary factors `0.76/1.32` produced `CONFIRMED`, phase `47.722 deg` versus `32.344 deg`, and Euclidean relative vector error `0.265377`.  
MINIMAL_REPAIR: Define complex error against the predicted complex norm, add explicit magnitude and phase-error gates, and add a prospectively frozen control/linearity probe covering `|q|=822`.  
REQUIRED_REGRESSION: Both anisotropic constructions must fail on a named phase, norm, or linearity gate while exact multiplicative gains continue to pass.  
MUST_REPAIR_BEFORE_LIVE: true

SLAW-04  
SOURCE: `gain_covariant_confirmation_public.py:569-589,672-689,822-827`; `gain_covariant_confirmation_runtime.c:732-751,801-825`.  
MECHANISM: The extractor defines `P0=L0` for `map0` and `P1=-L1` for `map1`. Consequently `relative_error(P0,-P1)` is identically `relative_error(L0,L1)`; "physical reversal" is not independent evidence. `measured_bank` is emitted but neither validated nor used.  
CONSEQUENCE: Changing every raw `measured_bank` label to `A` left all laws green and produced `CONFIRMED`. A failed physical crossover can therefore be represented as passing physical reversal whenever its logical numbers agree.  
MINIMAL_REPAIR: Validate the exact `(mapping, component) -> measured_bank` assignment and replace the duplicate comparison with independently observed bank-resolved evidence.  
REQUIRED_REGRESSION: All-`A` bank labels must hard-fail extraction, and a packet where logical mapping passes but bank crossover fails must fail only the independent physical gate.  
MUST_REPAIR_BEFORE_LIVE: true

SLAW-05  
SOURCE: `gain_covariant_confirmation_public.py:146-168,258-264,902-925,1199-1202,1399-1468`.  
MECHANISM: `declaration_sham` is DC across decoder phase and `query_scramble` is a second-harmonic pattern, so each has ideal first harmonic zero. Their decodes are computed but excluded from every decoded-null gate. Existing leakage mutations are dormant because the self-test list omits them.  
CONSEQUENCE: The built-in sham mutation produced real first harmonic `400` and the scramble mutation imaginary first harmonic `400`; both had no failed laws and returned `CONFIRMED`. Equivalent post/equal leakages were correctly rejected.  
MINIMAL_REPAIR: Freeze per-replicate real and imaginary first-harmonic null gates for both active controls and include them in final classification.  
REQUIRED_REGRESSION: `sham_first_harmonic_leakage` and `query_scramble_leakage` must fail their specific null gates; the regression must inspect those gates directly.  
MUST_REPAIR_BEFORE_LIVE: true

NONBLOCKING_CONCERNS  
- The `|q|=256` partition is internally consistent: actual absolute values are only `{0,822,1298,1536}`, with no mixed mapping pair. The `255/256` tests only exercise the comparison expression, but the frozen schedule has a wide boundary margin.  
- Aggregate geometry is genuinely diagnostic-only and cannot rescue either replicate.  
- The named near-zero statistics have coherent count scaling but are not independent: raw-leg bounds imply both pair averages and the decoded component bounds by the triangle inequality. The current "physical reversal average" mock never actually violates that named statistic.

ATTACKS_ATTEMPTED  
- Pure phase-independent additive offset: canceled by the four-phase decoder and did not independently forge geometry. Source-sign and active-source harmonic offsets did forge confirmation as described above.  
- Tested nonlinear saturation, incompatible control gains, low-gain/null inversion, quadrature distortion, mapping-label corruption, sham/scramble leakage, aggregate rescue, and threshold-boundary behavior.  
- Dynamic checks copied only the adjudicator into system temp; all temp directories were deleted afterward.

CLAIMS_SUPPORTED  
- Local branch/head remained `main` at `f6ef90374de424723e0edba34786778e8e3f1a29`; final status was only `main...origin/main [ahead 1]`, with no repository diff.  
- Every supplied hash matched exactly, including contract, both manifest hashes, schedule JSON/TSV, both private-map hashes, source bundle, runtime binary, implementation audit, and Sol audit.  
- Every source-bundle entry matched its exact commit blob.  
- Gain estimation excludes target/fold/polarity inputs; both replicates are hard-gated; aggregate values do not classify.  
- No live authority, hardware operation, target contact, Git write, repository write, or cross-agent discussion occurred.

CLAIMS_NOT_SUPPORTED  
- That the two controls establish one common carrier gain or are independent against nonlinear/phase-dependent response.  
- That "25% relative error" has dimensionally valid, contract-self-contained semantics.  
- That physical reversal is independently measured.  
- That target odd signal is separated from accepted nulls, or that all zero-first-harmonic controls are gated.

RECOMMENDATION  
Do not authorize a live run. Repair SLAW-01 through SLAW-05, explicitly define every error metric and SNR law in the contract, then re-freeze the implementation, manifest, bundle, binary, and direct gate-specific regressions before another read-only adversarial review.

