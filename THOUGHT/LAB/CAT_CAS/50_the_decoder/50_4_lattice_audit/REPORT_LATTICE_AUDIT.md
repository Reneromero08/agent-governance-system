# Exp 50.4 - Lattice Audit: does Exp 25's holographic LWE attack cross the located barrier?

**Verdict:** `EXP25_TOY_SCALE_ONLY` (all 4 gates pass, entry exits 0).
**Claim level:** 4-5 (a bounded audit of one solver under matched nulls; not a proof about all lattice attacks).

## The question

In 50.2c we located the irreducible decodability bedrock: the dihedral-HSP slope is
info-cheap but compute-hard = 1-bit-LWE / unique-SVP (lattice) hardness (Regev). Exp 25
(`25_lattice_holography/2_holographic_svp.py`) claims to **break** LWE with a holographic
phase-resonance readout and prints `LATTICE BROKEN!`. Those two facts collide: either
Exp 25 crosses the bedrock we located (extraordinary), or its "break" is a toy-scale
artifact and the barrier holds. This brick adjudicates by running Exp 25's **own** attack
under the Exp-50 null discipline.

## Mechanism (what the audit runs)

Faithful reproduction of `HolographicLatticeSolver`: map `A`, `B` to the torus
(`*2pi/q`), optimise a continuous secret `S_phase` with Adam (lr=0.1) to maximise phase
resonance `mean Re(Z_pred_sieved * conj(Z_B_sieved))`, with the Exp 25 FFT low-pass
(`cutoff = 0.15*m`) as the "phase cavity." Two readouts:

- **faithful** - exactly Exp 25: `S_pred = round((S_phase % 2pi)/(2pi) * q)`. The forward
  model wants `S_phase` = the integer secret, but this readout treats it as an angle in
  `[0,2pi)` - inconsistent unless secret entries are `< ~6`.
- **charitable** - the obvious bug fixed: `S_pred = round(S_phase) % q`, the integer the
  forward model actually optimises. Gives the attack its best possible shot, so the
  verdict cannot be dismissed as killing a typo.

No catalytic tape: this brick audits an existing solver's scaling; it borrows no
substrate, so a tape would be ceremonial (CAT_CAS_OS s.7). The mechanism is the Exp 25
attack itself, measured against nulls.

## Results (q = 3329 Kyber prime; chance per-coordinate accuracy = 1/q = 3.0e-4)

**Sweep A - recovery vs lattice dimension n (sigma=0, the attack's best case):**
exact-secret recovery is **0.00 at every n from 2 to 128**, for both readouts; coordinate
accuracy CI brackets chance throughout. Resonance peaks ~0.22 around n=16-32 and decays to
~0 by n=128 - i.e. the objective the attack maximises does not track recovery.

**Sweep B - recovery vs noise sigma (charitable, n=8):** exact recovery **0.00 at every
sigma in {0,1,2,4,8}**. No recovery survives even sigma=1, let alone the Kyber regime.

**Sweep C - tiny modulus q (n in {2,4}, sigma=0, 800 epochs, the absolute best case):**
the attack **does** work here, which is why G1 passes and we are auditing a real attack:
- charitable, q=5, n=2: exact_rate **0.25** (recovers the whole secret 25% of trials)
- charitable, q=11, n=2: exact_rate **0.12**
- partial coordinate recovery (CI above 0) at q in {5,11,17} for n=2.
Recovery vanishes by q=37 and at n=4 for the larger tiny moduli.

**Null block - planted vs no-secret (B uniform random), n=8:** planted resonance
0.215+/-0.037 vs null resonance 0.210+/-0.019; **Cohen d = 0.16**. The attack reaches the
same "success" with no secret present - its objective is decoupled from secret recovery.

## Gates

| Gate | Result | Detail |
|---|---|---|
| G1 attack works at toy scale (real attack, not broken) | PASS | best tiny-q exact_rate = 0.25 |
| G2 recovery collapses as n -> Kyber-256 | PASS | n=128 exact_rate 0, acc CI at chance |
| G3 recovery collapses under realistic noise | PASS | sigma>=2 exact_rate 0 |
| G4 no-secret null indistinct in the attack's objective | PASS | resonance Cohen d = 0.16 (< 0.8) |

## Interpretation

Exp 25's holographic LWE attack is genuine but **toy-scale-only**. It recovers small
secrets at a tiny modulus (q=5, n=2) and then collapses to the chance baseline as the
lattice dimension grows toward Kyber-256, as the noise reaches the real LWE regime, and -
decisively - **at its own shipped default of n=128, q=3329, where it recovers nothing.**
Its resonance objective cannot distinguish a planted instance from a no-secret null. This
is exactly what the lattice barrier located in 50.2c predicts: the holographic phase
readout does not cross unique-SVP hardness. The `LATTICE BROKEN!` print is an error-free /
tiny-scale artifact, not a crossing of the bedrock.

This **relocates Exp 25's "breaks post-quantum crypto" claim onto the barrier we located**
and shows it does not survive it - which is where the lab's boldest crypto claim needed to
be scrutinised hardest. The frontier question ("is the unique-SVP barrier itself crossable
by any holographic/topological/catalytic readout?") is unchanged and remains the Mythos
question; this audit only shows that *this* attack is not such a crossing.

## Honest caveats

- We did **not** prove lattice hardness; we located the barrier there (50.2c) consistent
  with the standard Regev result. This audit shows one attack fails to cross it.
- "Recovery" is Exp 25's own success criterion (exact secret, error norm 0) plus a
  coordinate-accuracy-vs-chance sensitivity check; both are 0 above toy scale.
- Sweeps are bounded (n<=128, trials 5-8, 250-800 epochs) for wall-clock; the trend
  (recovery -> chance with n, sigma, q) plus the no-secret-null decoupling carry the
  verdict, not exhaustive scale.

## Reproduce

```
python THOUGHT/LAB/CAT_CAS/50_the_decoder/50_4_lattice_audit/50_4_lwe_audit.py
```
Writes `lattice_audit_result.json` + `output_lattice_audit.txt`. Exits 0 iff all gates pass.
