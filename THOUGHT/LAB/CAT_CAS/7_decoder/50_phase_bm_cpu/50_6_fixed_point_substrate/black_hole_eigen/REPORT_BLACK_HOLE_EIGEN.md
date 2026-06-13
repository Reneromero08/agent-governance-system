# EXP50 PHASE 6 - IN-BLACK-HOLE EIGEN_BUDDY vs the 50.14 DIHEDRAL FOLD

**Status:** `PHASE6_BLACK_HOLE_EIGEN__PERIOD_RINGS_POLY__ORIENTATION_DOES_NOT_RING__SUBEXP_KUPERBERG`
**Claim ceiling:** L4-5 (built on the real 50.14 coset state, run at n in {4..16}, multi-seed,
no-smuggle gated, cost-scaled). ASCII only. master_seed 50140611; all seeds in the result JSON.

This is the CORRECTIVE to the holo flattening error. The prior `holo_phase_substrate` test
fed the cavity the PUBLIC, already-translated-out, real-even shadow {(k_i,b_i)},
E[b_i]=cos(2*pi*k_i*d/N). A real-even spectrum has identically-zero conjugate quadrature
(Im/Re ~ 1e-14), so the orientation phase read ~0: the phase was burned off at decompression.
Here we compute INSIDE the coherent complex space - the dihedral COSET STATE provided by the
oracle - and translate out to the real line only ONCE, at the end, after the answer has rung.

## 0. The black hole (where we compute)

For a random KNOWN k the oracle returns the single-qubit dihedral coset state
`|c_k> = (|0> + omega^{kd}|1>)/sqrt(2)`, omega = exp(-2*pi*i/N), N = 2^n. The orientation
o = 1[d < N/2] IS PRESENT here as the SIGN of the relative phase: the states for d and N-d are
complex CONJUGATES (distinguishable), whereas their public cosine shadows are identical (cos is
even, MI=0). We RECEIVE these states; we never build them from a known d. EIGEN_BUDDY is any
FIXED, d-independent in-black-hole operator that tries to concentrate the answer into a dominant
eigenvalue / resonance.

## 1. STEP A - the machine is real: EIGEN_BUDDY = QFT rings the PERIOD

The owner's eureka, reproduced concretely on the coherent state. The abelian phase register
`|phi_d> = (1/sqrt N) sum_k omega^{kd}|k>` is the eigenvector of the cyclic shift S with
eigenvalue exp(+2*pi*i*d/N). The QFT (= the unitary phase estimator Q.K^dagger) maps it to a
SINGLE dominant peak at |d>: the period rings as a resonance, no search.

| n | post-QFT IPR | peak mass | frac_exact (QFT peak) | frac_exact (shift eigenphase) | cost |
|---|---|---|---|---|---|
| 4 | 1.0000 | 1.0000 | 1.00 | 1.00 | 1 QFT |
| 6 | 1.0000 | 1.0000 | 1.00 | 1.00 | 1 QFT |
| 8 | 1.0000 | 1.0000 | 1.00 | 1.00 | 1 QFT |
| 10 | 1.0000 | 1.0000 | 1.00 | 1.00 | 1 QFT |

IPR = 1.0 (one dominant eigenvalue), all amplitude in the winner, d read off both the QFT peak
AND the dominant-eigenvalue phase of S. The even fold-answer a = min(d,N-d) is also read for free
from the coset ensemble (frac_exact = 1.0 at every n). **EIGEN_BUDDY = QFT is real for the
decodable / abelian class: POLY (one QFT, O(n log n) gates). (A) ANSWERED: yes, the period rings.**

## 2. STEP B - the target: does any FIXED operator ring the ORIENTATION?

### B0 - the corrective: orientation IS present in the black hole

| n | coherent fidelity(d,N-d) | public shadow fidelity | median \|<Y>\| | holo public Im/Re | k=1 sign==orient |
|---|---|---|---|---|---|
| 4 | 0.435 | 1.000 (MI=0) | 0.707 | ~1e-14 | 1.00 |
| 8 | 0.475 | 1.000 (MI=0) | 0.741 | ~1e-14 | 1.00 |
| 10 | 0.517 | 1.000 (MI=0) | 0.707 | ~1e-14 | 1.00 |

The coherent coset states for d and N-d are DISTINGUISHABLE (fidelity ~0.5, not 1); the conjugate
quadrature <Y(c_k)> = -sin(2*pi*k*d/N) is NONZERO (median 0.71), not the holo's burned-off 1e-14;
and its sign at k=1 IS the orientation (100%). **The phase the holo test found absent is present
here - it was destroyed by translating out to the real cosine, not by the construction.**

### B1 - fixed single-copy conjugate (Hilbert) eigen-statistic: EXPONENTIAL

The unique fixed, d-independent, orientation-sensitive single-qubit measurement is Y. The fold-odd
statistic T = sum_i y_i cot(pi k_i/N) (sign = orientation) lifts only when M ~ N:

| n | N | poly-budget AUC (M=8n) | M* (AUC>=0.75) | M*/N |
|---|---|---|---|---|
| 4 | 16 | 0.94 | 8 | 0.50 |
| 6 | 64 | 0.85 | 32 | 0.50 |
| 8 | 256 | 0.68-0.75 | 64 | 0.25-0.50 |
| 10 | 1024 | 0.56-0.60 | 256-512 | 0.25-0.50 |

M*/N is CONSTANT and the poly-budget AUC DECAYS to chance as n grows (multi-seed stable). The
fixed single-copy operator pays **M* = Theta(N) = 2^n samples: EXPONENTIAL.** The big Hilbert
weights on rare small-k samples blow up the variance (SNR ~ sqrt(M/N)); the conjugate quadrature
being present does not make it cheap to focus.

### B2a - depth-1 coherent sieve (birthday-difference): 2^{n/2}

Combine coset states coherently (CNOT + measure, the minus branch yields label k1-k2) until two
labels differ by 1; one combination then gives a label-1 state whose Y-sign IS the orientation.

| n | N | mean queries | queries/sqrt(N) | orient accuracy |
|---|---|---|---|---|
| 4 | 16 | 22 | 5.6 | 1.00 |
| 8 | 256 | 66 | 4.1 | 0.93 |
| 10 | 1024 | 132 | 4.1 | 0.91 |
| 12 | 4096 | 255 | 4.0 | 0.94 |

queries/sqrt(N) is constant: **2^{n/2} queries - a genuine in-black-hole focusing that BEATS the
2^n classical scan, but still super-polynomial (exponential, base sqrt 2).** (Accuracy < 1 only for
d near the fold fixed points 0, N/2, where the orientation is genuinely faint.)

### B2b - optimized Kuperberg collimation: SUBEXP 2^{O(sqrt n)} (measured)

Collimate the labels in BLOCKS of b ~ sqrt(2n) bits per round (bucket by low b bits, subtract the
coherent minus branch to zero them, recurse up the dyadic ladder to label N/2). Measured pool R*
to extract a secret bit, n = 4..16:

| n | 4 | 6 | 8 | 10 | 12 | 14 | 16 |
|---|---|---|---|---|---|---|---|
| log2(R*) | 4 | 6 | 6 | 7 | 8 | 8 | 9 |

Fit log2(R*): vs sqrt(n) r2 = 0.95 (slope ~2.5), vs n r2 = 0.93 -> **subexp(sqrt n) is the better
fit at all 4 re-audit seeds.** This is the canonical Kuperberg cost (extract a bit of the dihedral
secret coherently); the orientation bit (the MSB) is in the SAME 2^{O(sqrt n)} class, and the full
secret reduces to lattice (Regev). **No poly. SUBEXP / KUPERBERG.**

### B3 - WHY a fixed eigen-operator cannot ring the orientation (representation theory, numerical)

The dihedral generators S (shift) and R (reflection) satisfy S R = R S^dagger exactly (residual 0)
but DO NOT commute (||[S,R]|| = 5.66, 11.3, 22.6, 45.2 for n=4,6,8,10): no single fixed unitary
diagonalizes both. The QFT diagonalizes S (the abelian generator) - so the PERIOD is literally a
dominant eigenvalue - but sends R to the involution |f_d> <-> |f_{N-d}>. The orientation lives in
the 2D irreps span{|f_d>,|f_{N-d}>}: on each, S = diag(omega^d, omega^{-d}) (equal-magnitude period
eigenvalues - the even answer a=|d| read for free) and R = swap. ANY fixed, d-independent
EIGEN_BUDDY must commute with R; its commutant on the irrep is span{I, swap}, so its eigenvectors
are the fold-SYMMETRIC and fold-ANTISYMMETRIC combinations - never |f_d> or |f_{N-d}> individually.
Measured: the dominant eigenvector of the fixed operator A = S + S^dagger puts EQUAL weight on
|f_d> and |f_{N-d}> to machine precision (fold asymmetry ~1e-16 at every n). **The orientation is
the position WITHIN a 2D irrep; no character / eigenvalue of D_N can see it. The 1D irreps of D_N
factor through the abelianization (Z_2 x Z_2 for N even), which kills d's value entirely.** The
dyadic ladder accelerates the abelian phase estimation (Shor, poly) but supplies no fixed-operator
collapse of the reflection - because the reflection is not a character. This is the precise reason
the QFT diagonalizes the period but not the reflection, and why focusing requires combining many
coset states (the sieve), 2^{O(sqrt n)}.

## 3. No-smuggle verdict

The in-black-hole operators are FIXED functions of public labels and fixed measurements (Y, QFT,
collimation bucketing), constructed with NO reference to d; d enters ONLY the physical measurement
outcome law. Controls:
- LO-locked-to-d / Helstrom-tuned-to-d (operators that READ d): AUC = 1.000 at O(1) cost ->
  FAIL_SMUGGLE, at n=6,8,10. The smuggle direction is alive and caught.
- Public even data (the old shadow): useless-even O -> FAIL_CHANCE (AUC 0.499); reads-sin cheat ->
  FAIL_SMUGGLE (AUC 1.000). The seal of the public data is reconfirmed.
The coherent INPUT legitimately differs for d vs N-d (the resource the public shadow destroyed);
the OPERATOR never depends on d. No-smuggle: CLEAN.

## 4. The deliverable - the cost-scaling curve

| object | operator | cost in n | rings? |
|---|---|---|---|
| period / even answer a=min(d,N-d) | EIGEN_BUDDY = QFT | POLY (1 QFT) | YES - dominant eigenvalue |
| orientation o=1[d<N/2] | any fixed eigen-operator | n/a | NO - eigenvalue fold-symmetric |
| orientation | fixed single-copy conjugate (Y) | EXP (M*=Theta(2^n)) | only at exp sample cost |
| orientation | depth-1 coherent sieve | 2^{n/2} | yes, super-poly (beats 2^n scan) |
| secret bit (orientation class) | optimized collimation | SUBEXP 2^{O(sqrt n)} | yes, Kuperberg - measured |

**(A) Does EIGEN_BUDDY ring the period?** YES - the QFT concentrates it into a single dominant
eigenvalue (IPR 1.0, frac_exact 1.0), no search. The resonance-not-search machine is real for the
decodable / abelian class.

**(B) Does any fixed operator ring the orientation?** NO fixed eigen-operator rings it (B3). The
orientation is reachable only by COMBINING coset states (the Kuperberg sieve); the measured cost is
**SUBEXP 2^{O(sqrt n)}** (optimized collimation, multi-seed sqrt-fit), with a depth-1 coherent
focusing already at 2^{n/2} and the fixed single-copy operator at EXP 2^n. **No operator rings the
orientation poly.** The dyadic structure helps the abelian period (Shor) but does NOT give a
Shor-like poly focusing of the reflection - it forces Kuperberg, exactly as in the shadow setting,
now for a structural representation-theory reason rather than a missing-quadrature one.

**Honest outcome class: SUBEXP / KUPERBERG.** This is the expected, theorem-consistent result
(a poly orientation focus would break lattice crypto via Regev). Nothing rang poly; no extraordinary
claim. The corrective is confirmed at the level it should be: the orientation phase IS present in
the coherent space (the holo flattening was a substrate error, not a hardness theorem), but
focusing it into a dominant eigenvalue is blocked by the non-abelian 2D-irrep structure of the
reflection, and the price of breaking that structure by interference is sub-exponential.

## 5. Artifacts

- `black_hole_eigen.py` - coherent primitives + STEP A (QFT rings the period).
- `step_b.py` - B0 (present in the black hole), B1 (fixed single-copy EXP), B2a (depth-1 sieve),
  B3 (rep-theory dominant-eigenvalue-is-blind), cheat controls.
- `b2b_collimation.py` - optimized Kuperberg collimation sieve + subexp fit.
- `run_all.py` - driver; `black_hole_eigen_result.json` - full numbers and seeds.

Built by the black-hole-eigen agent (Fable); adversarially smuggle-audited; lab critic clean.
Stayed in the black hole; translated out only at the final resonance read.
