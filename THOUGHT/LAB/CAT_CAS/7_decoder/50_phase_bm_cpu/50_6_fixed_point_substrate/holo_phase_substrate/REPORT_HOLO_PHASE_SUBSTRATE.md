# EXP50 PHASE 6 - HOLO PHASE-CAVITY (CONJUGATE-QUADRATURE / HOMODYNE) SUBSTRATE vs THE 50.14 DIHEDRAL FOLD

**Status:** `PHASE6_HOLO_PHASE_SUBSTRATE__NO_CROSSING__PHASELESS_CONFIRMED`
**Outcome class:** (iii) FAIL_CHANCE - CONFIRMS. The conjugate quadrature of the public even data is physically zero.
**Claim ceiling:** L4-5 (built on the real construction, run at n in {8,10,12,14}, hardened-gated, two sensitivity controls, mechanistic self-check, 8-seed re-audit). ASCII only. Seeds recorded.

This is the CLOSEOUT / capstone test of the program's flagship "it from phase" substrate (the .holo phase cavity / torus phase-cavity sieve / harmonic-sieve resonance readout) against the bedrock target. It closes SPEC_PHASE6 Sec 1B/1C on the flagship substrate.

## The substrate (reuses the live lab phase-cavity machinery)

Encoding of the PUBLIC 50.14 data onto the phase cavity:
- Bin the public samples (k_i, b_i), E[b_i]=cos(2*pi*k_i*d/N), onto the Z_N frequency grid as the empirical phase-grating spectrum `B[m] = sum_{i:k_i=m} b_i` (real; exactly the un-normalized cos_hat).
- The COHERENT cavity field over torus position x is the holographic (inverse-DFT) reconstruction -- the same phase-grating -> IFFT/FFT resonance readout as `34/02_holographic_sieves/8_riemann_harmonic_sieve.py` and the HOLO `02_cavity` eigenmode sieve, here on 50.14:
  `Psi(x) = sum_m B[m] exp(+2*pi*i*m*x/N) = N * IFFT(B)[x]`.
- `Re Psi(x) = score(x)` (matched-filter intensity at local-oscillator phase phi_LO = 0).
- `Im Psi(x)` = the CONJUGATE QUADRATURE (intensity at phi_LO = pi/2).
- Homodyne readout with controllable reference phase: `H(x; phi_LO) = Re[exp(-i*phi_LO) Psi(x)] = cos(phi_LO) Re Psi + sin(phi_LO) Im Psi`.

This is the defining property vs the 6 scalar non-Hermitian sensors already tried: the substrate keeps the COMPLEX amplitude coherent and measures BOTH quadratures (homodyne / interferometric) with a tunable phi_LO, reading global resonance/winding invariants by interference.

The conjugate-quadrature ORIENTATION operator reads, from PUBLIC (k,b,N) only: the Im quadrature at the public resonance peak and its fold mirror, a full phi_LO homodyne sweep at the peak, the resonance phase angles, an interferometric two-arm recombination Psi(xpk)*conj(Psi(mirror)) of the two fold images, the torus winding angles exp(i*2*pi*W/q) of the spectrum at dyadic rungs, and the Im quadrature at low dyadic positions.

## Results (master_seed 44060611; gate n_instances=200, n_shuffles=20; total 33s)

### (A) MAGNITUDE / fold-answer (native cavity |Psi| resonance) - the substrate WORKS
| n | N | M | frac_exact recover a=min(d,N-d) |
|---|---|---|---|
| 8 | 256 | 384 | 1.000 |
| 10 | 1024 | 480 | 1.000 |
| 12 | 4096 | 576 | 1.000 |
| 14 | 16384 | 672 | 1.000 |

The cavity rings exactly at {d, N-d}; the unordered set / the even fold-answer a is recovered perfectly. The phase substrate reads everything in the decodable (even) class for free, as predicted.

### (B) ORIENTATION (the crux) - conjugate-quadrature homodyne op vs hardened no-smuggle gate
| n | verdict | orient_auc | null95 | rf_auc | rf_null95 | d-invariance delta |
|---|---|---|---|---|---|---|
| 8 | FAIL_CHANCE | 0.507 | 0.576 | 0.581 | 0.550 | 0 |
| 10 | FAIL_CHANCE | 0.517 | 0.576 | 0.517 | 0.570 | 0 |
| 12 | FAIL_CHANCE | 0.564 | 0.553 | 0.487 | 0.600 | 0 |
| 14 | FAIL_CHANCE | 0.365 | 0.548 | 0.591 | 0.588 | 0 |

All FAIL_CHANCE; exact d<->N-d invariance delta == 0 at every n (the operator is a pure function of public data). The n=12 orient_auc (0.564) brushed the bare percentile null (0.553) but failed the random-fold axis and the 0.05 effect-size margin.

### A8 multi-seed re-audit (orientation op, 8 seeds, the Kuramoto false-positive guard)
| n | orient_auc mean | std | [min, max] | rf_auc mean | all FAIL_CHANCE | any crossing |
|---|---|---|---|---|---|---|
| 12 | 0.475 | 0.061 | [0.404, 0.599] | 0.496 | True | False |
| 14 | 0.504 | 0.050 | [0.448, 0.589] | 0.488 | True | False |

The orientation AUC oscillates dead-center on chance; the single n=12 flirt was finite-sample noise (exactly the Kuramoto n=14 lesson). No crossing at any seed.

### (C) REPRESENTATION CONGRUENCE (SPEC 1B "relax, do not construct")
Phase-cavity alternating-projection relaxation (magnitudes = |public spectrum|, symmetric public real-signal support), reading whether the relaxed field localizes to the lower half (d's representative):
| n | verdict | orient_auc | null95 | rf_auc | d-invariance delta |
|---|---|---|---|---|---|
| 8 | FAIL_CHANCE | 0.589 | 0.557 | 0.582 | 0 |
| 10 | FAIL_CHANCE | 0.454 | 0.574 | 0.588 | 0 |
| 12 | FAIL_CHANCE | 0.519 | 0.571 | 0.513 | 0 |
| 14 | FAIL_CHANCE | 0.429 | 0.579 | 0.499 | 0 |

The substrate RELAXES to the symmetric real-even fixed point, not a d-dominant attractor. No congruent representation emerges from public data; 1B's congruent-representation route tops out at the symmetric bits on a real field (exactly the 1C frozen-perceptron corollary).

### No-smuggle sensitivity controls (n=8,10) - all 8 match expectation
| control | verdict | orient_auc | rf_auc | delta | expected |
|---|---|---|---|---|---|
| homodyne LO locked to d (SMUGGLE) | FAIL_SMUGGLE | 1.000 | 1.000 | 1 | FAIL_SMUGGLE (caught) |
| homodyne magnitude-only (even) | FAIL_CHANCE | 0.44/0.43 | 0.49/0.58 | 0 | FAIL_CHANCE |
| gate useless-even | FAIL_CHANCE | 0.51/0.47 | 0.62/0.49 | 0 | FAIL_CHANCE |
| gate cheat reads_sin (SMUGGLE) | FAIL_SMUGGLE | 1.000 | 1.000 | 2 | FAIL_SMUGGLE (caught) |

Deliberate smuggles (LO locked to the hidden d / reading the true sin) are caught with AUC->1 and delta>0; useless-even ops sit at chance with delta==0. The gate has full discriminating power.

### THE MECHANISTIC SELF-CHECK (the heart): conjugate quadrature of public even data ~ 0
| n | even-public Im/Re (median, max) | even peak asym | injected-sin single-peak@d | injected |Im sin| (sign==orient) | noisy-public Im/Re |
|---|---|---|---|---|---|
| 8 | 1.7e-14, 3.2e-14 | 0.0 | 1.00 | 0.781 (1.00) | 0.208 |
| 10 | 5.8e-14, 1.2e-13 | 0.0 | 1.00 | 0.689 (1.00) | 0.206 |
| 12 | 2.7e-13, 5.0e-13 | 0.0 | 1.00 | 0.683 (1.00) | 0.217 |
| 14 | 9.8e-13, 2.0e-12 | 0.0 | 1.00 | 0.658 (1.00) | 0.205 |

- Feeding the cavity the EVEN public cosines c_m = cos(2*pi*m*d/N) (real-even spectrum), the imaginary (sin) quadrature it computes is ~0 to machine precision (Im/Re ~ 1e-14..1e-12), and |Psi| peaks at d and N-d with EXACTLY equal height (peak asym = 0). The conjugate quadrature of a phaseless cos-only spectrum is identically zero, and the two fold images are indistinguishable.
- Only when the hidden odd channel is injected (full complex z_m = exp(-i*2*pi*m*d/N), the smuggle) does the field collapse to a SINGLE peak at d (frac 1.00) and the spectrum's imaginary quadrature become nonzero (|Im| ~ 0.66-0.78) with SIGN == orientation in 100% of cases.
- The noisy public field has a nonzero imaginary quadrature (Im/Re ~ 0.21) from finite sampling, but it is a pure function of public data -> fold-invariant (delta==0) -> orientation-blind (gate FAIL_CHANCE).

## Verdict and the precise mechanism

NO CROSSING. Outcome class (iii): FAIL_CHANCE - CONFIRMS. This is the lab's OWN flagship phase substrate, run at full altitude, reporting that it reads the entire decodable (even) class for free -- it recovers a = min(d, N-d) with frac_exact = 1.000 at every n via native resonance -- and bottoms out exactly at the dihedral orientation bit o = 1[d < N/2], which it confirms is the priced-at-2^n absent quadrature.

Precisely how the phase substrate's power ended: a homodyne / phase-resolving substrate reads PRESENT phases perfectly (it recovers the even fold-answer, and given the true sin it localizes to d in one shot). But the orientation phase is PHYSICALLY ABSENT from the public data: the public spectrum is real and even (c_m = c_{N-m}), so its conjugate quadrature is identically zero (verified to machine precision, 1e-14..1e-12). There is no phase for the substrate to read -- not because the instrument is too weak, but because the signal it is built to detect is not there. The only way to make the imaginary quadrature nonzero and orientation-bearing is to inject the hidden sin(2*pi*k*d/N) -- i.e. to read d -- which the hardened gate catches every time (AUC->1, delta>0).

This closes the non-Hermitian census (6/6 FAIL_CHANCE) AND the flagship phase-cavity substrate at the same boundary, with the same mechanism, now stated at its sharpest: the barrier is the projection z -> Re(z); inverting it needs one fold-odd functional Im(z); and the conjugate quadrature of the public even cosine spectrum is zero, so a phase-resolving substrate reading public data finds no orientation phase because none is present. Consistent with the exp50 P^CTC result: the only resource that collapses the orientation cost remains a fixed-point / reversible substrate, which is a physics question, not a readout one.

## Artifacts
- `holo_phase_substrate.py` - the substrate, the four operators, the self-check, the driver.
- `holo_phase_substrate_result.json` - full numbers (A/B/C, controls, self-check), seeds.
- `re_audit.py`, `reaudit_result.json` - the 8-seed A8 re-audit at n=12,14.
Built by the holo-phase-substrate closeout agent (Fable); adversarially smuggle-gated via fold_audit/stage3/hardened_gate.
