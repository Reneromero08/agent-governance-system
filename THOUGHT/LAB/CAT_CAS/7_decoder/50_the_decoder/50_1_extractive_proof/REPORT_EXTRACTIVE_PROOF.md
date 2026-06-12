# Exp 50.1 — The Extractive Proof

**Verdict:** `EXTRACTIVE_CONFIRMED` (5/5 gates). Claim level **4-5**.

## Overview

- **Mechanism:** a global coherent spectral readout (FFT-peak / covariance-eigenmode / prime-resonance sweep) vs. a class of *lookup* decoders restricted to a bounded receptive field or bounded statistical order.
- **Invariant:** a global frequency / character (synthetic tone) and the non-trivial Riemann zeros (primes).
- **Isomorphism:** this is the abelian-HSP / Fourier-sampling advantage rendered as a measurement — the answer is a global character, recoverable only by coherent integration over the full domain.
- **Claim (level 4-5):** the holographic decoder is **extractive** — it reads a global invariant that no lookup-class decoder recovers, and it survives a statistics-matched wrong-answer control. NOT claimed: any domain/ontology conclusion (L6-8).

## The lookup-null class (formal)

A decoder is **lookup-class** iff it factors as `g(Φ(E))` where `Φ` is (a) a value scan/table over `E`, (b) low-order shift-invariant statistics (moments / histogram), or (c) a windowed feature map with receptive field `w << M`. **Compute is unbounded; only the functional form is constrained.** The defining restriction is bounded receptive field OR bounded statistical order — never a global phase-coherent sum over the full domain.

**Why each fails (proved, not just observed):** the coherent SNR of a frequency estimate scales with integration length `L`: `SNR ~ A^2 L / sigma^2`. We set the amplitude so `A^2 w / sigma^2 < 1 < A^2 M / sigma^2` (per-window SNR = 1.0, global SNR = 64). Any decoder limited to receptive field `w < M` therefore cannot reach detection; the global operator can. The histogram null is f-blind because the value histogram of `A·e^{iθ}+noise` is rotation-invariant — certified empirically by the wrong-answer control.

## Method

Two testbeds. **Note on deviation from plan:** the mod-exp *period* grating was dropped from the null-separation — its sequence returns to 1 at the period, so a trivial value-scan finds `r` (Exp 20's own iteration fallback). That is a weak separation. The rigorous version uses cases where the answer is genuinely a global coherent-integration property:

1. **synth** — a controllable weak tone in noise (provable locality barrier; primary).
2. **zeta** — the real lab decoder: primes → explicit-formula grating `S(w)=Σ Λ(n) n^{-1/2-iw}` (von Mangoldt weight `ln p/√p`) → global resonance sweep → zeros emerge as power peaks (poles of −ζ′/ζ on the critical line). Reuses the in-lab working `analyze_spectrum` / sweep machinery.

Extractive decoders: `fft_peak`, `eigenmode` (covariance-eigenbasis, FFT-independent cross-check), `freq_sweep` (zeta). Lookup-nulls: `windowed_fft`, `windowed_kay` (lag-1 phase), `windowed_autocorr`, `histogram_regressor`.

## Results (synth: M=4096, 60 signals, global SNR=64, per-window SNR=1.0, w=M/64)

| decoder | class | success | 95% CI | vs extractive |
|---|---|---|---|---|
| fft_peak | EXTRACTIVE | **1.000** | [0.940, 1.000] | — |
| eigenmode | EXTRACTIVE | **1.000** | [0.940, 1.000] | — |
| windowed_fft | lookup-null | 0.117 | [0.058, 0.222] | p=2e-4, h=2.44 |
| windowed_kay | lookup-null | 0.050 | [0.017, 0.137] | p=2e-4, h=2.69 |
| windowed_autocorr | lookup-null | 0.067 | [0.026, 0.159] | p=2e-4, h=2.62 |
| histogram_regressor | lookup-null | 0.050 | [0.017, 0.137] | p=2e-4, h=2.69 |

Random-guess chance ≈ 0.040. Extractive ≈ 1.00; every lookup-null ≤ 0.12, defeated at p=2×10⁻⁴, Cohen h > 2.4.

## Wrong-answer control (anti-circularity)

Matched pair (same noise, tones at k=137 vs 613): statistical identity **min KS p = 0.754**, max |moment diff| = 3.4×10⁻³. Extractive **tracks truth** (137, 613 exact). Statistics-null **cannot** (returns 565, 308 — both wrong). This constructively rules out "the spectral decoder keys on a stored statistic."

## Zeta (real lab decoder)

6000 primes, sweep [10,50], 6000 bins. **All 10 first zeros recovered (score 1.00)**; phase-scrambled control (identical amplitude distribution, coherence destroyed) scores 0.40 — a 0.60 differential. *Honest caveat:* the absolute coverage is inflated by peak density; the real signal is the real-vs-scrambled differential, which is clean.

## Catalytic integrity

Grating XOR-encoded into a `CatalyticTape`; decode reads the grating back **out of the mutated tape** (`E = current ⊕ dirty`), recovering k=551 (matches direct). `uncompute()` + `verify()`: SHA-256 `013e874ef8bb…` initial == final, `was_modified=True`. 0 bits erased.

## Gates

| Gate | Result |
|---|---|
| G1 extractive(synth) ≥ 0.8 | PASS (1.000) |
| G2 extractive(zeta) recovers zeros | PASS (1.00 vs scrambled 0.40) |
| G3 null separation (p<1e-3, rate<½ extractive, h>0.8) | PASS (all 4 nulls) |
| G4 wrong-answer control | PASS |
| G5 catalytic restoration | PASS |

## Conclusion

On both a controllable synthetic case and the real lab zeta decoder, the holographic readout extracts a global invariant that no bounded-receptive-field or statistical-order decoder recovers, and it tracks the true answer where statistics-matched nulls cannot. The decoder is **extractive, not lookup** — at claim level 4-5. The barrier separating extractive from lookup is *integration length (locality)*, a structural property, not a compute handicap. This licenses Brick 2: map *where* along a problem continuum this extractive power survives vs. collapses.
