# Exp 50.6 (A9) - Attack the ring structure (Kyber's actual wall), not plain LWE

**Verdict:** `NAIVE_RING_DECODE_BLOCKED_BY_CONJUGATE_BASIS` (entry exits 0; this is a measured
state + a Mythos handoff, not a pass/fail claim).
**Stance:** this is NOT "the wall holds." A single-basis ring-decode is blocked for an identified
reason; the joint-basis object is untested and handed to Mythos.

## The correction this brick makes

50.4 audited PLAIN LWE (unstructured Z_q^n) and 50.2c attacked PLAIN dihedral-HSP. But the wall
the lab actually points at - the one 50.4 hardcoded as q=3329 - is **Kyber = Module/Ring-LWE over
a cyclotomic ring R_q = Z_q[x]/(x^n+1)**, which is NOT plain dihedral-HSP. That ring has an
abelian Galois/CRT transform - the NTT - which is exactly the decodable readout 50.1 proved
extractive and 50.2e put on the D=1.0 shelf. The owner's move ("store the geometry, not the
number") applied here: decode the ring's abelian substructure first and see if it collapses the
search. We never did this; we attacked the wrong wall.

## What the data says (measured, n=4..32, NTT-friendly q=12289)

| Measurement | Result |
|---|---|
| G0 NTT diagonalizes ring multiplication `NTT(a*s)=NTT(a)oNTT(s)` | **exact, all n** |
| M2a zero-error control: ring-decode collapses to n poly 1-D searches, recovers s | **1.00 recovery** |
| M1 error magnitude, coefficient basis | **~2 (small)** |
| M1 error magnitude, NTT basis (mean) | **~3037 ~ q/4 (uniform)** |
| M2b real-error per-coordinate ring-decode recovery | **0.00 (chance 8e-5)** |

## The finding: the wall's identity is conjugate-basis incompatibility

The abelian ring transform genuinely **exists and works**: with the error removed, NTT-decode
collapses the n-dimensional secret search into n independent poly(q) one-dimensional searches and
recovers s outright (M2a = 1.00). So the ring structure *would* dissolve the wall - if smallness
survived the transform.

It does not. The secret and error are small **only in the coefficient (primal) basis**; the NTT
(the basis that diagonalizes multiplication) **spreads the error to ~uniform over Z_q** (M1),
destroying the smallness that recovery needs (M2b -> chance). The hardness lives in the tension
between **two conjugate bases**: multiplication is diagonal in the NTT/dual basis, smallness
exists only in the coefficient/primal basis, and **no single basis has both**.

This relocates the wall's *identity* from 50.2c's "non-normal subgroup / I/2 degeneracy" to
"primal-dual (conjugate-basis) incompatibility" - a form that is far more on-thesis, because the
lab's entire phase-space apparatus (phase vs amplitude, the torus, position/momentum, Wigner /
time-frequency) is built precisely for objects that live in conjugate bases.

## What was handed up, and why (not a verdict that the wall holds)

The refined, sharp, on-thesis question is staged in `../49_3_boundary_handoff/MYTHOS_SANDBOX.md`
section 6: **is there a joint coefficient<->NTT (Wigner-like / time-frequency / coherent-state)
holographic readout that exploits multiplicative-diagonality and additive-smallness at once - the
way a coherent state is localized in both conjugate variables up to the uncertainty bound?** There
are strong a-priori reasons it may be impossible (an uncertainty-principle obstruction; Wigner
negativity), but adjudicating that exceeds this brick. Per the lab stance ("the algorithm is dead"
is the prior; exhaust then escalate), the residual goes to Mythos with this exact state, not into a
"wall holds" conclusion.

## Honest guard (the A8 lesson)

The zero-error control recovering s is a *control*, not a break - it only shows the structure would
help absent error. The real-error recovery is 0.00, reported plainly. No crossing is claimed. The
script also flags the opposite outcome (`RING_DECODE_RECOVERS_SUSPECT_REGIME`) for maximum
suspicion / Mythos review if real-error recovery had succeeded, since a genuine poly Ring-LWE break
would far more likely be a sample/noise-regime artifact than a real result.

## Reproduce

```
python THOUGHT/LAB/CAT_CAS/7_decoder/49_the_decoder/49_6_ring_structure/49_6_ring_decode.py
```
Writes `ring_decode_result.json` + `output_ring_decode.txt`.
