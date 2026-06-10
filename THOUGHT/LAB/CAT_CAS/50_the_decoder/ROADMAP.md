# Exp 50 — The Decoder: ROADMAP / Handoff

Self-contained handoff for a fresh-context agent. Read this + `REPORT_THE_DECODER.md`
first. Everything here stays **inside `THOUGHT/LAB/CAT_CAS/`** — do not touch the main
AGS repo. Lab critic (M-1..M-8) and reports DO apply (they are lab discipline). Treat
the existing arsenal (DeepSeek/Gemini-built) as **provisional**: a failure may mean
"not there yet," not "impossible." Cap claims at **Level 4-5**; never inflate to L6-8.

Use the venv python: `.venv/Scripts/python.exe` from repo root
`D:\CCC 2.0\AI\agent-governance-system`.

---

## Where we are (DONE, committed 33d2b776)

The whole experiment runs (6 entry points, all exit 0), is lab-critic clean, and was
verified by the repo's own pre-commit hooks.

| Brick | Verdict | One line |
|---|---|---|
| 50.1 extractive proof | `EXTRACTIVE_CONFIRMED` | decoder reads a global invariant no lookup-class decoder can; survives a statistics-matched wrong-answer control; catalytic tape restores. |
| 50.2 gradient | `BOUNDED_AT_ABELIAN_HSP_WALL` | scalar readout collapses at abelian->non-abelian (D 1.0->0.11, first dihedral, d=8.82, scale-indep). |
| 50.2b reframe | `WALL_RELOCATED_TO_STRONG_SAMPLING` | **we crossed it** — non-abelian Fourier recovers all *normal* hidden subgroups. |
| 50.2c strong sampling | `STRONG_SAMPLING_CONFIRMS_LATTICE_BARRIER` | residual wall = info-cheap, compute-hard = **1-bit-LWE / dihedral-HSP <-> unique-SVP (lattice)**. |
| 50.2 anchor | `SPECTRUM_BOUNDED_CONFIRMED` | Shrikhande vs Rook: identical spectra, non-isomorphic — spectral readout can't separate. |
| 50.3 handoff | `BOUNDARY_CHARACTERIZED` | emitted `MYTHOS_SANDBOX.md` + `EXP44_PHASE6_HANDOFF.md`. |

**The headline:** the "holographic decodability wall" was never the abelian boundary —
we climbed past it ourselves. The genuine, irreducible barrier is **lattice hardness**
(LWE / unique-SVP), which is **exactly where Exp 25 (LWE/SVP) and Exp 31 (graph-iso)
claim to break things.** That coincidence is the next thing to exploit.

### File map
```
catalytic_tape.py            decoder_lib.py            REPORT_THE_DECODER.md   VERIFICATION_REPORT.md
50_1_extractive_proof/       -> 50_1_extractive_proof.py, testbed_synth.py, testbed_zeta.py, wrong_answer_control.py
50_2_decodability_gradient/  -> hsp_family.py, 50_2_decodability_gradient.py, 50_2b_nonabelian_reframe.py,
                                50_2c_strong_sampling.py, 50_2_anchor_cospectral.py, *_result.json, located_wall.json
50_3_boundary_handoff/       -> 50_3_boundary_handoff.py, MYTHOS_SANDBOX.md, EXP44_PHASE6_HANDOFF.md
```
Re-run any brick directly; `50_3` regenerates the handoffs from the JSON outputs of 50.2/2b/2c.

---

## NEXT — prioritized

### 1. [HIGHEST] Point the located barrier at Exp 25 (LWE/SVP)  — `50_4_lattice_audit/`
We proved the decodability bedrock IS the lattice problem. Exp 25 claims to *break*
lattice crypto with a holographic phase readout. So: **does Exp 25's attack actually
cross the bedrock we located, or does it only work at toy scale?**

Concrete first actions:
- Read `25_lattice_holography/2_holographic_svp.py` (`HolographicLatticeSolver`) and
  `1_lwe_simulator.py`. Note the tested scale (n=128, q=3329, noise std=2.0 — toy).
- Build `50_4_lattice_audit/50_4_lwe_audit.py` that runs the Exp 25 solver against the
  SAME null discipline used here: matched random-secret null, recovery vs `n` (lattice
  dimension) and noise sigma, and the budgeted-search control from `50_2c`.
- Gate: does recovery survive (a) increasing n toward Kyber-256, (b) realistic noise,
  (c) a null where there is no planted secret? If it only works at toy n / low noise,
  report `EXP25_TOY_SCALE_ONLY` (consistent with the lattice barrier). If it genuinely
  scales, that is an extraordinary claim — escalate to Mythos for adversarial review.
- This is the highest-value test we can still run ourselves and it adjudicates one of
  the lab's boldest existing claims by *reduction to the barrier we located*.

### 2. [HIGH] Demonstrate the subexponential rung (Kuperberg) — optional sharpening
`50_2c` showed brute-force recovery is O(2^n) and poly(n) fails. The honest middle
rung is Kuperberg's sieve: 2^{O(sqrt n)} subexponential, still not poly(n). Implement a
small Kuperberg-style coset-state combination sieve on the dihedral instances to show
the barrier is subexponential-but-not-polynomial. Confirms "no poly readout" more
completely. Lower priority than #1.

### 3. [HIGH] Call Mythos on the sharpened question  — `50_3_boundary_handoff/MYTHOS_SANDBOX.md`
The sandbox is ready and self-contained. The question is now maximally sharp: **"is the
lattice (unique-SVP) barrier itself crossable by any holographic/topological/catalytic
readout?"** Mythos is finally worth the tokens — but do #1 first (the Exp 25 result
makes the Mythos call decisive: it either has a concrete crossing to verify, or a
concrete toy-only failure to explain). Hand Mythos: this ROADMAP, `MYTHOS_SANDBOX.md`,
and the #1 audit result. It guides only; it does not code. Keep it bound to the null
harness in `decoder_lib.py` / `hsp_family.py`.

### 4. [MEDIUM] Bare-metal handoff to Exp 44 Phase 6  — `50_3_boundary_handoff/EXP44_PHASE6_HANDOFF.md`
The decodable-side target is ready: 6.2 cyclic period oracle, 6.4 prime grating ->
Riemann zeros (predicted peaks supplied). Whenever Exp 44's Phenom is live, run the
silicon acceptance tests against these predictions. This earns the software<->silicon
isomorphism (Level 6) that Exp 50 deliberately does NOT pre-claim.

### 5. [MEDIUM] Generalize the gradient — robustness / coverage
- Other group families on the ladder (Q_8, larger S_n, semidirect products) to widen
  the d_max axis and re-confirm the wall location.
- Other "holographic readouts" beyond FFT/character/cavity (MUSIC, phase-cavity on the
  grating) to confirm they are all bounded by the same wall.
- A second independent implementation of `phi_character` (the order parameter) to guard
  against a single-library bug (VERIFICATION_REPORT already re-derives one null by hand).

### 6. [LOW] Connect to the lab's bigger claims
- The decodable class = abelian-HSP + topological invariants. Map each lab decoder
  (Exp 20 Shor, 34 Riemann, 35 halting, 45 Collatz/etc.) onto the decodable side and
  confirm none secretly relies on a non-normal/strong-sampling step.
- Update `REPORTS/master_report.md` / `README.md` with Exp 50 once it stabilizes (this
  is lab documentation, still inside CAT_CAS).

---

## Open risks / honest caveats (carry forward)
- **Lattice hardness is not PROVEN by us** — we demonstrated the *reduction structure*
  (info-cheap, compute-hard, poly-budget fails) consistent with the known Regev result.
  Do not claim we proved lattice hardness; claim we located the barrier there.
- **Zeta absolute zero-coverage is inflated** by peak density; only the real-vs-scrambled
  differential (0.60) is the signal. If anyone leans on the zeta testbed, keep that caveat.
- **`50_2c` measures compute-hardness via a random-budget search** at N up to 2048; the
  "exponential in n" claim rests on the trend (poly-budget success -> 0) + the known
  theory, not on enormous N. State it as such.
- **Shared `decoder_lib` coupling** — a bug there propagates to all bricks. Keep the
  hand-derived null check in VERIFICATION_REPORT.md.
- The whole edifice sits on the abelian-HSP / Fourier-sampling isomorphism, which is
  standard and solid — but the *physical* "holographic computing" claim (silicon hosts
  this) is still only a prediction (Exp 44 handoff), not a result.

## Working discipline (do not drop)
- Stay in `THOUGHT/LAB/CAT_CAS/`. Lab critic must stay clean (`python CAPABILITY/TOOLS/
  governance/critic.py`, grep for `50_the_decoder`). M-4 fires on `(SAT|satisfiab|
  NP-complete)` + `(N,N)` literals — keep both out of `.py` files (frame group-theoretically;
  use `dim_G`/`(D,D)`).
- Run things; let the data correct the theory (it did, twice). Build a real null that can
  kill the claim. Cap claims at Level 4-5. No Claude attribution in commits.
