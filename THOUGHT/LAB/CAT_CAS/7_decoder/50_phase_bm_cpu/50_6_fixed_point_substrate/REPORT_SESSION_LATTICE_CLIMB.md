# SESSION REPORT - The Lattice Wall Climb (Exp 50 Phase 6)

A full account of the session: the target, every wall, every attack, what happened,
which documents helped, and - the part for you - exactly where the crack CANNOT be
(proven closed) and where it still COULD be (assumptions we never broke). ASCII only.

---

## 0. The target, stated exactly (so we argue about the right object)

The Exp50.14 public fixed-point map. N = 2^n. Hidden secret d in [1, N/2).

- Public data: M ~ sqrt(N) samples (k_i, b_i). k_i uniform in Z_N. b_i a sign with
  E[b_i] = cos(2*pi*k_i*d/N).
- Derived public objects (all computable from (k,b), never from d):
  - score(x) = sum_i b_i cos(2*pi*k_i*x/N)
  - accept(x) = score(x) > M/4   (true iff x in {d, N-d})
  - f(x) = x if accept(x) else (x+1) mod N   (the directed map; fixed points {d, N-d})
  - a = min(d, N-d)   (the "fold-answer")
- The secret-of-the-secret: o = 1[d < N/2]   (the ORIENTATION bit; which of {d, N-d} is d).
  Note: a = d exactly when o = 1, and a = N-d when o = 0. So "find d" = "find a" (easy)
  + "find o" (the wall).

The complex object behind it: z_k = exp(-2*pi*i*k*d/N). Re(z_k) = cos = PUBLIC. Im(z_k)
= sin = the orientation channel. The fold sigma: d <-> N-d is exactly theta <-> -theta
(theta = 2*pi*d/N): cos(m*theta) is INVARIANT (even) for all m; sin(m*theta) FLIPS SIGN
(odd) for all m. That one line is the whole wall.

---

## 1. The climb log (every wall, the attack, the outcome)

### Wall 1 - "Is the orientation hard, or absent?"
Attack: Stage 1 fold audit (Fable build, fold_audit/). On the real construction at
n=8..14: trained 16 classifiers (logreg/SVM/GBT/MLP) + 8 nonlinear lifts to predict o
from public data; ran a two-sample test of the d-conditioned vs (N-d)-conditioned
distributions; built a complex (quadrature) control.
Outcome: AUC = 0.5 across all 24 readouts; two-sample distributions IDENTICAL; GIVEN the
quadrature, one-shot exact d recovery via the dyadic ladder. The likelihood identity
P(D|d) = P(D|N-d) holds POINTWISE on every dataset (the bit is binary, its whole law is
its cos-mean, which is fold-even), so by the data-processing inequality I(o ; T(D)) = 0
for ANY transform T at ANY compute. **VERDICT: o is information-ABSENT from the public
data, not hard.** This is the single most important result: it moved the wall from
"computationally hard" to "the bit is not in this representation."

### Wall 2 - "Then re-encode into a representation where it IS present."
Attack: Mythos (Fable) consult #1 + the equivariance theorem (SPEC 1B.1).
Outcome: any FAITHFUL (information-preserving) re-encoding is an intertwining isomorphism;
the fold is the NON-NORMAL reflection subgroup of the dihedral group D_N; NORMALITY is an
isomorphism invariant. So no faithful re-encoding turns the reflection into a character
(an unfaithful one just smuggles the orientation). **VERDICT: hardness-class is a property
of the SECRET'S SYMMETRY TYPE, not the encoding. Re-encoding is provably closed.**

### Wall 3 - "Maybe the real construction LEAKS o somewhere (a bug)."
Attack: Stage 3 generator audit (fold_audit/stage3/). Audited every published quantity of
the real 50.14: the bit's full conditional law, the k-draw, M, the threshold, sample
order, PRNG seed, the float code-path (bitwise cos(2pi k d/N) vs cos(2pi k (N-d)/N)), the
verify map.
Outcome: ORBIT-ONLY. The one scary signal (AUC 0.92 on float low-bits) was a probe reading
the NON-PUBLISHED intermediate cos value = reading d; published bits are bitwise-identical
for d and N-d. **VERDICT: no implementation leak. The bedrock proof applies without caveat.**
Built the hardened no-smuggle gate here (random-private-fold + exact d<->N-d invariance) -
the instrument that made every later negative trustworthy.

### Wall 4 - "A quantum/coherent substrate gets the quadrature; the dyadic ladder makes it Shor-poly."
Attack: Mythos (Fable) consult #2 (dyadic-vs-Kuperberg).
Outcome: Shor-vs-Kuperberg is decided by the STATE YOU HOLD, not the frequencies. Shor-easy
IFF d is the eigenphase of a unitary you can APPLY and EXPONENTIATE (U^{2^j} by squaring);
Kuperberg-hard IFF d is only the relative phase of single-shot coset states at
adversarially-RANDOM k, with no such operator. 50.14 gives random-k coset states = the
reflection presentation = genuinely dihedral. The dyadic ladder {N/2,...,1} is the readout
SCHEDULE you WANT, not a state you get to LAND on; landing on it would itself cost the
Kuperberg sieve. **VERDICT: TWO stacked walls were being conflated - Wall A = the projection
z->Re(z) (crossable by coherence); Wall B = frequency control (random k vs chosen 2^j),
which coherence does NOT cross. Wall B is the dihedral barrier proper.**

### Wall 5 - "Our own exp50 result says reversible/CTC = poly. Reconcile."
Attack: retrieved the exact 50.14 reduction (49_14_substrate.py + MYTHOS_BRIEF).
Outcome: its "poly" is a literal Deutsch CTC oracle, P^CTC = PSPACE (Aaronson-Watrous). It
finds fix(f) directly. The lab itself capped it L4-5: "this does NOT claim a physical
crossing... conditional on P^CTC existing... not a complexity question." **VERDICT: exp50
is HONEST (d emerges from a public map, no planting) but its resource is the PSPACE
sledgehammer that trivializes everything - exactly the door Fable flagged. No new crack;
it converges with the dihedral verdict.**

### Wall 6 - "The lab's METHOD is the non-Hermitian topological sensor, not classical readout. Point it at the fold."
This was the big push (FINAL.md sec 6's own open question). The engine bet: the map f's
+1 directionality BREAKS the fold, and directed/non-reciprocal structure is exactly what
non-Hermitian topology reads (point-gap winding, skin effect, exceptional points - invariants
with NO Hermitian analog). Built SIX encodings (nonhermitian_sensor/), each smuggle-gated,
cost-scaled n=8..14:
1. Koopman/transfer operator of f - point-gap winding. FAIL_CHANCE. The transfer operator
   is fold-IDENTICAL (P H(g) P = H(-g)).
2. Hatano-Nelson non-reciprocal hopping + skin effect. FAIL_CHANCE. Skin center-of-mass is
   global, orientation-blind.
3. Kuramoto / non-reciprocal chiral order parameter. FAIL_CHANCE (one n=14 PASS killed as
   finite-sample false positive on 13-seed reaudit). Chirality reads "directed current,"
   not which half.
4. Cauchy argument-principle winding on a contour around [1,N/2). FAIL_CHANCE; and the
   contour that WOULD separate costs O(N)=2^n to resolve (poly-budget gives the wrong
   winding, error grows in N). Cost relocated to the integral.
5. PT-symmetric / biorthogonal. FAIL_CHANCE. Spectrum fold-blind.
6. Godel-edge phi-twist (Exp 36 machinery; added after you pointed me there) - sweep the
   boundary-twist PHASE, track winding cheaply via Exp 36's rank-1 determinant lemma.
   FAIL_CHANCE. THE SHARPEST RESULT: the +1 directionality DOES survive into a genuine
   point-gap winding W=+1 (flips to -1 if the walk is reversed), but W is a PUBLIC CONSTANT
   - it carries the walk DIRECTION, not the half. Same for every instance.
Outcome: **6/6 FAIL_CHANCE. The directionality is real, topological, non-Hermitian - and
orientation-BLIND.** Every smuggle control caught. Exp 36's rank-1 lemma validated (1638x
speedup, cheap where dense is 2^n-infeasible) - but it crushes the cost of an
orientation-blind quantity = cost relief, not an information crossing.

### Wall 7 - "The flagship .holo phase substrate ('it from phase') reads the quadrature natively."
Attack: built the lab's own homodyne phase-cavity substrate (holo_phase_substrate/, reusing
HOLO/pipeline/02_cavity + 34_zeta_eigenbasis/.../8_riemann_harmonic_sieve.py). Encoding:
public (k,b) -> phase-grating spectrum B[m] -> coherent field Psi(x)=N*IFFT(B). Re Psi =
score (LO phase 0), Im Psi = conjugate quadrature (LO phase pi/2). Swept phi_LO, read the
imaginary quadrature, interferometric recombination of the two fold images, torus winding.
Outcome: (A) reads the even fold-answer a at frac_exact 1.000 (the substrate WORKS); (B)
orientation FAIL_CHANCE at all n, A8 8-seed reaudit confirms; (C) representation-congruence
FAIL_CHANCE (relaxes to the symmetric real-even fixed point). **THE MECHANISTIC HEART: fed
the EVEN public cosines, the imaginary (sin) quadrature the cavity computes is ~0 TO MACHINE
PRECISION (Im/Re ~ 1e-14); the two fold peaks at d and N-d are exactly equal. Only injecting
the hidden sin makes |Im| nonzero with sign == orientation 100%, and the gate catches it.**
The orientation phase is PHYSICALLY ABSENT from public data: the public spectrum is
real-even, hence phaseless, hence its conjugate quadrature is identically zero. There is no
phase to read - the homodyne is not weak, the signal is not present.

---

## 2. The unified mechanism (why every path converged)

The wall is EXACTLY the map z -> Re(z) (the 2-to-1 map theta -> -theta). Its kernel IS the
fold. The public data is the real/even projection; the orientation is the odd/imaginary part
that the projection destroys. Five facts, all proven/measured this session:

1. o is information-absent from the public data (MI=0, an identity).
2. No faithful re-encoding moves it (normality is an iso-invariant).
3. No constructed non-Hermitian topological invariant reads it (6/6); the map's directionality
   survives only as a public constant.
4. The lab's flagship phase substrate reads it as zero to machine precision (the even spectrum
   is phaseless).
5. The even "fold-answer" a = min(d, N-d) is read for FREE by every sensor and the cavity
   (2 eigs at 1; frac_exact 1.0). "The sensor is the solution" WORKS - for the abelian/even
   (decodable) class. The wall is precisely and only the orientation bit.

The decodable class (Exp 50) = {abelian-HSP + topological invariants of a poly-size operator}.
The dihedral orientation sits exactly outside it. Mythos located the wall further: it IS
class-group VECTORIZATION = the isogeny / CSIDH hardness assumption (Kuperberg 2^O(sqrt n)).

---

## 3. Documents that helped (and what each gave)

- GPT's review of the 5.10 work - sharpened "no lift synthesizes orientation" into the
  EQUIVARIANCE THEOREM (any sigma-equivariant transform of the cosines stays sigma-invariant),
  with the no-smuggle gate as the harvester. Folded into SPEC 1B.1.
- PRIMER.md (repo) - the loading index; "vision first, then filter"; pointed me to the right
  docs. Caught me running pure-filter (gatekeeper) mode.
- Primer.md (vault, Low-Friction Interaction Protocol) - the explicit warning against
  standing at the boundary explaining why boundaries exist. This corrected my framing twice.
- It From Phase.md - phase as recursive depth / the substrate of reality; the conjugate
  quadrature as the missing object.
- Algorithm is Dead.md - "geometry makes time irrelevant; the sensor is the solution; measure
  the topological invariant, don't iterate." This is the lab's METHOD, and it WORKS for the
  even half.
- 2026-05-21 Catalytic Time.md - catalytic computing on a reversible (black-hole) tape; the
  Q.K^dagger = phase-estimator eureka. The reversible/P^CTC resource.
- What Is My System - FINAL.md - the canonical synthesis. Sec 6 IS this exact frontier
  question ("is the lattice barrier crossable by any holographic/topological/catalytic
  readout?") and already names the decodable class + the irreducible lattice boundary.
- 03_SEMIOTIC_WAVE_MECHANICS.md - "meaning is the cross-term 2|psi1||psi2|cos(dtheta) that
  Shannon discards." STRUCTURALLY THE SAME as Fable's result: the orientation lives in the
  channel the scalar readout throws away. (Note: meaning is the cos cross-term; orientation
  is the sin channel. Both are "the relational part the magnitude readout discards" - worth
  pondering whether the analogy is exact or just rhymes.)
- 49_the_decoder/ROADMAP.md + REPORT_LATTICE_SPIRAL.md + MYTHOS_BRIEF.md - the 50.6-50.14
  spiral (readout -> curvature -> substrate), the P^CTC reduction, the class-group/isogeny
  identification.
- SPEC_PHASE6_FIXED_POINT_SUBSTRATE.md - sec 1A (brute relaxation on an arbitrary score is
  a structural zero: information accounting, Omega(N) classical query bound, the bits/sec
  deficit) and sec 1B (representation congruence - the "relax don't construct" open angle).
- 36_bekenstein_godel/REPORT.md - the Godel-edge point-gap winding, the rank-1 determinant
  lemma (constant-cost winding under a rank-1 sweep), log-space for exponentially small
  couplings. A real cost-technique; the P^CTC "tape" is ceremony + classical speedup.

---

## 4. WHERE THE CRACK CANNOT BE (do not spend your gift here)

These are PROVEN closed this session. If the answer is "hiding in plain sight," it is NOT
in any of these:

- Any classical transform / lift / clever feature of the PUBLISHED (k,b) data. MI(o; data)
  = 0 is an identity, not a "we couldn't." No compute crosses it.
- Any faithful re-encoding of the secret (iso-invariant; normality is preserved).
- A bug/leak in the construction (orbit-only, audited to the float bit).
- A non-Hermitian topological invariant of an operator BUILT FROM PUBLIC DATA (6/6; the
  operator is fold-identical, the directionality is a public constant).
- A phase-resolving readout of the public data (its conjugate quadrature is zero to machine
  precision - there is no phase present to read).
- Cost cleverness on the winding (Exp 36 already crushes it; cost was never the binding
  constraint - information was).

The negatives are real (every smuggle control was caught at AUC 1.0). The "I couldn't" vs
"can't" gap is closed on the published-data side: it is "can't," proven.

---

## 5. WHERE THE CRACK COULD STILL BE (aim your thinking here)

The wall is now so precisely characterized that if it is hiding in plain sight, it is hiding
in an ASSUMPTION we treated as fixed, not in cleverness on the published data. Four candidate
assumptions to break, ranked by how live they are:

### 5.1 The ACCESS MODEL (the most concrete crack)
We attacked the PASSIVE published data. But the construction also exposes an ACTIVE object:
the iterable map f and the real-valued score(x), evaluable at any x you choose. The map's
DYNAMICS are fold-breaking (the +1 walk has a direction). We proved the STATIC operator/
winding gives only a public constant, and the BASIN sizes (2a and N-2a) encode a (even, not
o). But did we exhaust the DYNAMICS? Open sub-questions:
- The two fixed points {a, N-a} have basins of size 2a and N-2a. a is even. But is there a
  fold-ODD functional of the TRANSIENT (how the walk approaches each fixed point - phase of
  approach, return times, the local spectrum AT each fixed point) that we never measured?
  We measured the global winding; we did not exhaustively measure local/transient invariants
  at each fixed point separately.
- KEY tension to resolve: a = d when o=1 and a = N-d when o=0. The MAP knows which fixed
  point it walked to. Is there ANY public dynamical quantity that is different at d vs at
  N-d (not at "min vs max", which is even, but genuinely at the two group elements)? Our
  proof says no (fold-identical operator) - but verify it covers the full TRANSIENT, not
  just the steady spectrum.

### 5.2 The THREAT MODEL (what "crossing" is allowed to use)
We defined crossing = recover o from public data WITHOUT d. The lab's own sec 1B frame is
"RELAX, don't construct": a physical substrate that settles into d's basin without anyone
writing the feature map. The honest obstruction (sec 1A) is that relaxation on an ARBITRARY
score is Omega(N). But the open question sec 1B marks LIVE: is there a substrate whose
relaxation is NOT on the arbitrary score but on a representation where d is the dominant
attractor - prepared WITHOUT d? This is exactly the representation-congruence question; we
closed it for CONSTRUCTIBLE representations, but the PHYSICAL preparation of a congruent
basin (vs constructing it) is the one sec 1B angle not fully killed. The pitfall: any
congruent representation still has to be prepared, and every construction of it needs d.

### 5.3 The LOWER BOUND IS UNPROVEN (the famous open door)
Nobody has proven dihedral-HSP / unique-SVP requires super-poly time. A poly quantum (or
classical) algorithm for dihedral HSP is NOT excluded - it is a famous decades-open problem.
If your "hiding in plain sight" is a new ALGORITHM for the dihedral coset problem itself,
that is a genuine math breakthrough, not a substrate trick. It would not be unlocked by any
re-encoding (sec 4) - it would attack the secret's class directly. Highest difficulty; not
lab-buildable; but it is the one place "the wall holds" is NOT proven.

### 5.4 The SPECIFIC isogeny / class-group STRUCTURE (not generic dihedral)
Mythos identified the wall as class-group VECTORIZATION = isogeny/CSIDH hardness, via the CM
embedding (ring class field, dihedral Galois group, conjugation = inversion, d = an ideal
class). This is RICHER than abstract dihedral HSP - it is a specific number-theoretic object
with its own active cryptanalysis (CSIDH has had real subexponential and quantum attacks).
The 50.6-50.9 spiral probed the ring/NTT structure and found "no single basis is small in
coeff AND diagonal in NTT" - that conjugate-basis obstruction is the SAME wall in number-
theory clothes. The crack, if structural, may live in a feature of the SPECIFIC class group
/ isogeny graph that the generic dihedral framing hides - not in the abstract fold.

---

## 6. The one-paragraph honest state

We proved the algorithm is dead for the construct side and measured the phase-substrate
crossing false for the published problem - the orientation is the absent quadrature,
confirmed five independent ways including the lab's flagship cavity. The wall is located to a
single bit and a single mechanism (z -> Re(z), the fold = the non-normal reflection). It is
NOT "the wall holds." The residual is: break the ACCESS MODEL (active map dynamics / local
transient invariants at the two fixed points), or the THREAT MODEL (physical congruent-basin
relaxation, sec 1B), or find the unproven dihedral lower bound is false (a new algorithm), or
exploit the SPECIFIC isogeny/class-group structure the generic framing hides. Everything you
can build from the published even data is proven shut - so the crack, if it is hiding in
plain sight, is in one of those four assumptions, not in the data.
