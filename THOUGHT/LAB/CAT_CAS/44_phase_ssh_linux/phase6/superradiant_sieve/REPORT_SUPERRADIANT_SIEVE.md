# EXP44 PHASE 6 - CHIRAL SUPERRADIANT combination of the 50.14 coset states

**Status:** PHASE6_CHIRAL_SUPERRADIANT__LOOPHOLE_REAL__COLLECTIVE_NO_GAIN__SUBEXP_KUPERBERG
**Claim ceiling:** L4-5 (built on the real 50.14 coset state + the lab's validated non-Hermitian
superradiance engine; run at n in {4..12}, multi-seed, no-smuggle gated, cost-scaled). ASCII only.
master_seed 50140611; all seeds in superradiant_sieve_result.json.

This is the convergence test of the program: superradiance is the physical form of "combine the
coset states as one collective resonance" (one diagonalization, NOT Kuperberg's iterative sieve),
it is non-Hermitian, and the orientation is a CHIRALITY (handedness). The 50.14 black-hole result
left exactly one loophole: the wall is "any FIXED operator that COMMUTES WITH the mirror R is
blind," and a CHIRAL geometry is a fixed, d-independent operator that does NOT commute with R. We
build that chiral handedness reference physically and MEASURE the scaling against the 2^{O(sqrt n)}
Kuperberg bar.

## 0. The engine is faithful (Dicke + sum rule + bright/dark)

The lab's validated radiative Hamiltonian H = Omega - i Upsilon/2 (Spano-Mukamel dipole coupling)
reproduces, with no tuning:

- Dicke test: two parallel near-field dipoles give collective decay Gamma/gamma = [2, 0] exactly
  (one superradiant, one dark).
- Sum rule: sum_j Gamma_j = M gamma to 1.4e-14 (the anti-Hermitian trace is conserved).
- Chiral channel (traveling wave, Gamma_G = gamma w w^dagger, w_m = e^{i k0 z_m}): ONE super-bright
  mode Gamma = M, ||Im(Gamma_G)|| = 5.6, ||[Gamma_G, P_reflect]|| = 11.2 != 0.
- Achiral channel (standing wave, Gamma_G = gamma cos(k0(z_m-z_n))): two bright modes,
  ||Im(Gamma_G)|| = 0, ||[Gamma_G, P_reflect]|| ~ 3e-15 = 0 (MIRROR-SYMMETRIC).

The chiral array is a fixed, d-independent operator that breaks the mirror; the achiral array is
the mirror-symmetric control that, by the 50.14 wall, must be orientation-blind.

## 1. The encoding (no-smuggle)

The oracle hands us M single-qubit coset states |c_{k_j}> = (|0> + omega^{k_j d}|1>)/sqrt2 for
random KNOWN labels k_j (omega = exp(-2 pi i / N), N = 2^n). Emitter j IS this state: its
transition-dipole phase is p_j = omega^{k_j d} (the secret lives in the phase, the labels are
public). The fixed chiral array provides the collective decay operator Gamma_G; the physical
collective emission rate of the phased array is the quadratic form R(d) = v(d)^dagger Gamma_G v(d),
v_j(d) = p_j / sqrt(M).

The wall, restated at the array level. With D(d) = diag(p_j) unitary, the phased dynamics is
Gamma(d) = D(d)^dagger Gamma_G D(d) -- a SIMILARITY transform. So the collective decay RATES
(eigenvalues / the bright-mode decay rate) are d-INVARIANT (measured ~1e-14). The orientation does
NOT ring in any eigenvalue (the array-level B3). What CAN ring is the chiral EMISSION asymmetry of
the specific phased input:

  R(d) - R(N-d) = 2i v^dagger Im(Gamma_G) v  =>  T(d) = Im( v^dagger Im(Gamma_G) v ),
    ACHIRAL: Im(Gamma_G) = 0  => T == 0  => AUC = 0.5 EXACTLY  (mirror-blind).
    CHIRAL:  T(N-d) = -T(d)               => sign(T) = orientation (a handedness read).

Closed form (O(M), exact to 1e-15 vs the dense matrix): T(d) = (2 gamma/M) Im(conj(S_s) S_c),
S_s = sum_j sin(k0 k_j) p_j, S_c = sum_j cos(k0 k_j) p_j.

## 2. The loophole is REAL (chiral reads what every mirror-symmetric operator cannot)

| n | chiral AUC | achiral AUC | eig-rate d-invariance |
|---|---|---|---|
| 4 | 0.609 | 0.500 (EXACT) | 1.8e-14 |
| 6 | 0.507 | 0.500 (EXACT) | 2.1e-14 |
| 8 | 0.582 | 0.500 (EXACT) | 3.6e-14 |
| 10 | 0.508 | 0.500 (EXACT) | 4.3e-14 |

The achiral (mirror-symmetric) array is blind to machine precision -- AUC = 0.5 EXACTLY, because
Im(Gamma_G) = 0 makes the chiral statistic identically zero (not merely small: zero). The chiral
array's statistic is genuinely orientation-odd (T(N-d) = -T(d), nonzero) so it lifts above chance.
On the contiguous matched set the lift is unambiguous (chiral AUC 0.94 at n=8,10 vs achiral 0.500).
The chirality loophole is real: a fixed, d-independent chiral operator reads the orientation that no
mirror-symmetric operator can. The open question was never "does it read it" but "at what scaling"
-- and that is where the result lands.

## 3. The scaling (THE DELIVERABLE)

### 3a. Chiral COLLECTIVE vs independent single-copy (B1), random labels (the honest oracle)

| n | N | chiral M* (AUC>=0.75) | chiral peak AUC (M=16N) | indep-B1 M* | indep M*/N |
|---|---|---|---|---|---|
| 4 | 16 | none | 0.729 | 8 | 0.50 |
| 6 | 64 | none | 0.735 | 16 | 0.25 |
| 8 | 256 | none | 0.700 | 64 | 0.25 |
| 10 | 1024 | none | 0.534 | 256 | 0.25 |
| 12 | 4096 | none | 0.550 | 1024 | 0.25 |

The chiral COLLECTIVE read never resolves at poly M (M up to 16N), and its peak AUC DECAYS toward
chance as n grows. The independent single-copy conjugate read (B1: sum_j Im(p_j) cot(pi k_j/N), the
per-copy matched filter) reaches AUC ~1.0 at M* = Theta(N) = 2^n (M*/N constant ~0.25, fit
log2 M* vs n r2 = 0.988, slope 0.90 -> EXP). The one-shot SHOT-NOISE collective detection (~M/2
photons) is no better (peak AUC 0.55-0.72, M* none).

The chiral COLLECTIVE coupling does NOT beat -- it is WORSE than -- the independent single-copy
read. The off-diagonal chiral mixing scrambles the clean per-copy quadrature instead of
concentrating it. There is no superradiant speedup of the orientation.

### 3b. Structured frequency sets (dyadic ladder, contiguous matched) + the resource caveat

| n | dyadic chiral | dyadic indep | matched chiral | matched indep |
|---|---|---|---|---|
| 4 | 0.686 | 0.979 | 0.652 | 1.000 |
| 6 | 0.623 | 0.944 | 0.864 | 1.000 |
| 8 | 0.598 | 0.941 | 0.942 | 1.000 |
| 10 | 0.578 | 0.930 | 0.942 | 1.000 |

Structured sets read the orientation easily -- but ONLY because they contain SMALL labels (k = 1,
2, ...: the conjugate quadrature Im(p_1) = -sin(2 pi d/N) IS the orientation, the B0 corrective).
Under the standard random-label oracle, synthesizing chosen small labels costs the Kuperberg sieve
2^{O(sqrt n)} (the b2b collimation). So the structured read re-prices to subexp, not poly, and again
the independent read dominates the collective one.

## 4. No-smuggle verdict

| control | n=6 | n=8 | n=10 | verdict |
|---|---|---|---|---|
| achiral (mirror-symmetric) | 0.500 | 0.500 | 0.500 | BLIND (structural) |
| d-locked homodyne (uses d) | 1.000 | 1.000 | 1.000 | FAIL_SMUGGLE |
| useless-even (reads abs(p)^2) | 0.500 | 0.500 | 0.500 | FAIL_CHANCE |

The array geometry (k0, channel, label set) is FIXED and d-INDEPENDENT; orientation labels enter
ONLY the supervised AUC scoring, never the operator. The lab hardened_gate (reused) catches reads_d
and reads_sin (FAIL_SMUGGLE) and passes useless_even (FAIL_CHANCE) -- the gate machinery is alive. A
resonant k0 = 2 pi d/N is sign-definite (the resonance sin^2 is EVEN in d) so it cannot even read
the orientation: tuning the geometry to d does not help. No-smuggle: CLEAN.

## 5. The honest outcome class: SUBEXP / KUPERBERG (no poly crossing)

| object | operator | cost in n | rings? |
|---|---|---|---|
| collective decay rate / bright-mode eigenvalue | chiral array spectrum | n/a | NO - d-invariant (similarity) |
| orientation | achiral (mirror-symmetric) array | n/a | NO - Im(Gamma)=0, AUC 0.5 exact |
| orientation | chiral COLLECTIVE emission asymmetry | EXP / unresolved at poly M | weakly, decaying |
| orientation | independent single-copy conjugate (B1) | EXP (M* = Theta(2^n)) | yes, at exp sample cost |
| orientation | structured (dyadic/matched) chiral read | SUBEXP (labels cost the sieve) | yes, 2^{O(sqrt n)} |

Does the chiral superradiant combination ring the orientation? YES, it can -- the chirality is the
correct key for the mirror wall (the achiral control is blind to machine precision; the chiral
statistic is genuinely orientation-odd). Does it beat the sieve? NO. The collective superradiant
combination provides NO speedup over the single-copy conjugate read, and NO poly crossing. The price
stays B1-exponential for random labels and Kuperberg-subexp once the informative (small) labels are
paid for. Honest outcome class: SUBEXP / KUPERBERG -- the expected, theorem-consistent result (a
poly orientation focus would break lattice via Regev). Nothing rang poly; no extraordinary claim.

WHY superradiance does not beat the sieve. The collective enhancement is a sum-rule-conserving,
POLYNOMIAL amplitude gain (sum_j Gamma_j = M gamma; the chiral channel concentrates it into one
bright mode of rate M). But reading the orientation needs the RESONANT matched filter k0 = 2 pi d/N,
which a FIXED d-independent geometry cannot supply for unknown d -- and a geometry tuned to d becomes
sign-definite (even in d), so it reads nothing. The chiral channel supplies the missing HANDEDNESS
(it breaks the mirror, which is necessary), but not the missing RESONANCE (which is what costs the
sieve). Off-resonance, the collective pair-sum fluctuates with quasi-random signs, so the collective
read is no better than -- in fact worse than -- summing the independent per-copy quadratures, which
is itself the B1 Theta(N) operator. The handedness reference removes the representation-theory
blindness of B3 but not the sub-exponential price of focusing a non-character.

## 6. Artifacts

- chiral_engine.py - the validated non-Hermitian collective engine (Dicke, sum rule, chiral vs
  achiral traveling/standing-wave decay matrices, bright/dark modes).
- coset_array.py - coset-state encoding; collective rate R, chiral handedness statistic T,
  asymmetry a; O(M) closed forms; eigenvalue-blindness (similarity); independent B1 control.
- scaling.py - random-label M(n) scaling, one-shot shot-noise model, structured sets, controls, fits.
- run_all.py - driver; superradiant_sieve_result.json - full numbers and seeds (21.6s).

Built by the superradiant-sieve agent (Fable). Engine validated against the lab's superradiance
result; adversarially smuggle-audited (achiral structural control + d-locked cheat + lab
hardened_gate); stayed in the coherent collective order, translated out only at the final chiral
emission read.
