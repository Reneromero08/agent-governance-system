# Exp 50 — The Decoder

**Rollup verdict:** the holographic decoder is **extractive** (proven); its extractive power decodes the abelian / normal-subgroup class and bottoms out at the **non-normal / strong-sampling = lattice (unique-SVP)** wall (located and characterized as subexponential-but-superpolynomial); and the lab's boldest crossing claim (Exp 25 LWE/SVP) is **audited and does not cross it** (toy-scale-only). Whole-experiment claim ceiling: **Level 5**. Lab-critic clean (M-1..M-8). All entry points exit 0.

> **STATUS: CLOSED OUT (theory terminus + handoff).** The MYTHOS (Fable 5) consultation ran (5 rounds; verdicts in `49_3_boundary_handoff/MYTHOS_BRIEF.md` -> ## RESULTS): the dihedral wall **IS** class-group **vectorization** = the isogeny/CSIDH hardness assumption; **no field-only catalyst shortens it** (unit-lattice wrong layer; Stickelberger/Brumer-Stark annihilator-not-short-basis, period-sized; catalytic space = CL subset P; Arakelov `d` orthogonal to the field entropy). **Boundary verdict (owner's correction):** the boundary is the catalytic **tape** = the entropy = the boundary projection of higher-dimensional geometry; the field-route rounds tested the wrong object; the "needle" is a projection artifact; the crossing = relax into the tape's entropy-geometry, a **substrate event**. This is **not** "the wall holds" - the **hypothesis stays open at the substrate (Exp 44, the 5.10 -> Phase 6 ladder)**, now the live frontier; Exp 50's remaining role is the target generator (the 50.14 public fixed-point map).

> **Roadmap run (this session):** items #1, #2, #5, #6 of `ROADMAP.md` executed in-lab (Mythos #3 was run after the spiral - see STATUS above; #4 = the Exp 44 substrate frontier, now open). Four new bricks added: 50.4 (lattice audit), 50.2d (Kuperberg rung), 50.2e (gradient coverage), 50.5 (decoder class map). See the extensions table below.

> **The Lattice Spiral (50.6 - 50.14), this session:** eleven adversarial passes around the lattice wall (full account in `REPORT_LATTICE_SPIRAL.md`). The wall moved **readout -> curvature -> substrate**: every spectral/contour/winding readout reads `d` for free (the topological reframe is *confirmed* - `d` is a conserved invariant), but `d` is the per-step curvature of its own trajectory, so a forward machine that builds the trajectory needs `d` (the `2^n` search; no amplification beats Fisher - 50.13). The final brick (50.14) relocated the entire wall onto the **substrate**: `d` emerges as the unique fixed point of a *public* map (no smuggle), found in `2^n` forward but poly on a reversible / catalytic / CTC fixed-point substrate (P^CTC=PSPACE). "The algorithm is dead" is true precisely on the reversible substrate the framework posits; its physical realizability is tested by **Exp 44 Phase 6** (silicon goes catalytic). No physical crossing claimed; the forward floor is mapped to the atom.

## The thesis it settles

Is the holographic readout a *decoder* (reads a global invariant out of the encoding's structure) or a *lookup* (returns what was stored)? And **where** does that decoding power hold vs. collapse? This is the line between "holographic computer" and "beautiful database."

## The three bricks

| Brick | Result | Key evidence |
|---|---|---|
| **50.1 Extractive Proof** | `EXTRACTIVE_CONFIRMED` (5/5) | Spectral 100% vs 4 lookup-nulls 5–12% (Cohen h 2.4–2.7, p=2e-4); wrong-answer control: extractive tracks truth on statistics-matched pair (KS p=0.75), statistics-null cannot; catalytic SHA restored, 0 bits erased; real zeta decoder recovers 10/10 zeros (vs scrambled 0.40). |
| **50.2 Decodability Gradient** | `BOUNDED_AT_ABELIAN_HSP_WALL` (5/5) + anchor `SPECTRUM_BOUNDED_CONFIRMED` | Scalar-readout D collapses 1.000 (abelian) → 0.110 (non-abelian) at the first dihedral group D_8 (d_max 1→2), Cohen d=8.82, scale-independent; cospectral anchor (Shrikhande vs Rook, spectral distance 2e-15, cliques 8 vs 0) confirms spectrum-bounded. |
| **50.2b Non-abelian reframe** | `WALL_RELOCATED_TO_STRONG_SAMPLING` (3/3) | We crossed the scalar wall ourselves: the non-abelian Fourier reframe recovers every **normal** hidden subgroup (D 0.123→1.000) but leaves a **residual wall** at **non-normal** subgroups (D=0.157) = strong Fourier sampling = the dihedral-HSP ↔ unique-SVP lattice frontier. |
| **50.2c Strong sampling** | `STRONG_SAMPLING_CONFIRMS_LATTICE_BARRIER` (3/3) | Climbed to strong (within-irrep) sampling and hit bedrock: a single dihedral coset state is I/2 (zero info); the slope is **info-cheap** (O(√N) states) but **compute-hard** — recovery is a 2ⁿ secret-space search, poly(n)-budget success → 0 (0.20→0.00). The residual wall **is** the 1-bit-LWE / unique-SVP lattice problem (Regev). |
| **50.3 Boundary + Handoffs** | `BOUNDARY_CHARACTERIZED` | Emitted `MYTHOS_SANDBOX.md` (barrier-or-frontier question, bound to the same null) and `EXP44_PHASE6_HANDOFF.md` (decodable target for silicon). |

## Roadmap-run extensions (this session)

| Brick | Result | Key evidence |
|---|---|---|
| **50.4 Lattice audit** | `EXP25_TOY_SCALE_ONLY` (4/4) | Ran Exp 25's own holographic LWE attack (faithful + charitable readout) under matched nulls. Recovers a secret only at tiny modulus q=5, n=2 (exact_rate 0.25); **0 recovery at its own default n=128/q=3329**, 0 at noise sigma>=2, and its resonance objective cannot tell a planted instance from a no-secret null (Cohen d=0.16). Does NOT cross the bedrock located in 50.2c. |
| **50.2d Kuperberg rung** | `DIHEDRAL_BARRIER_SUBEXPONENTIAL_UPPER_BOUND` (4/4) | Collimation sieve recovers the dihedral slope in subexponential queries: 2^n/M_needed widens 4x->4.2e6x over n=6..30; conditional readout correctness 1.000 (null at chance). With 50.2c's poly-budget failure this **sandwiches** the barrier: super-polynomial but subexponential (does NOT claim to separate subexp from poly from the fit; that side is cited from Regev/Kuperberg + 50.2c). |
| **50.2e Gradient coverage** | `WALL_IS_NON_NORMAL_SUBGROUPS` (5/5) | Q_8 (non-abelian, Hamiltonian, all-normal) is **decodable** (D_char=1.000) while dihedral/AGL(1,5)/A_5/S_n collapse (Cohen d=8.98): the wall is **non-normal** subgroups, not non-abelianness. Scalar FFT recovers abelian only and misses Q_8 (D_fft=0.10) - the reframe is needed (readout hierarchy). Independent projector phi agrees with the loop phi to 2.2e-16. |
| **50.5 Decoder class map** | `DECODER_MAP_CONSISTENT` (4/4) | Maps the lab's decoders onto the class: 9 decodable (Exp 20/24 abelian-HSP; 34/35/36-40/45/46 topological invariants), 3 bounded/wall (Exp 31 cospectral, 45.5 NxN, 25 lattice). None of the working decoders relies on a non-normal/strong-sampling step. Validated the Exp 44 handoff's predicted peaks against independent mpmath zeta zeros (<0.03% error). |

## Combined claim ledger (conservative)

| Claim | Level | Why not higher |
|---|---|---|
| The holographic readout is extractive, not lookup | **4-5** | survives 4 nulls + wrong-answer control; no domain/ontology claim |
| Decodability collapses at the abelian→non-abelian boundary | **4-5** | measured (d=8.82), scale-independent, but the wall's *identity* is a question, not proven |
| The wall is **non-normal** hidden subgroups (not non-abelianness) | **4-5** | measured (50.2e): Q_8 non-abelian but all-normal is decodable; d=8.98 split by H-normality |
| The barrier is subexponential-but-superpolynomial | **4** | sieve upper bound measured (50.2d); super-poly side cited from Regev/Kuperberg + 50.2c, not separated empirically |
| Exp 25's holographic LWE attack does **not** cross the wall | **4-5** | audited (50.4): toy-scale-only under matched nulls; 0 recovery at its own default; not a proof about all lattice attacks |
| The collapse is the known non-abelian-HSP barrier | **NOT CLAIMED** | open question for Mythos |
| The lattice (unique-SVP) barrier itself is crossable | **NOT CLAIMED** | the Mythos question, deliberately not run this session |
| Silicon hosts the decodable side | **NOT CLAIMED** | prediction handed to Exp 44 Phase 6 (handoff peaks validated; silicon run hardware-blocked) |

## What this gives the lab

A measured operating manual for holographic computing: **the decoder genuinely extracts global invariants (it is not a lookup); the scalar readout decodes the abelian/Fourier class; the non-abelian Fourier reframe extends that to all *normal* hidden subgroups; and the genuine residual wall is the *non-normal* / strong-Fourier-sampling case — the dihedral-HSP ↔ lattice frontier.** The period/Riemann/halting decoders sit safely inside the decodable region. The **LWE/graph-iso "wall-crossing" claims (Exp 25, 31) sit exactly on the relocated wall** — strong sampling / lattice hardness — which is precisely where the cospectral anchor shows the spectral readout fails. So those claims are the ones needing extraordinary evidence. The boundary is now relocated to the *real* barrier and scoped as one sharp question for a stronger model, plus a concrete bare-metal target.

## Grain-of-salt / provisional notes

The arsenal these results build on was largely produced by weaker models (DeepSeek/Gemini); a collapse here may be a barrier OR an uncrossed frontier — which is exactly the Mythos question. The zeta absolute zero-coverage is inflated by peak density (the real signal is the real-vs-scrambled differential). The mod-exp *period* grating was dropped from the extractive proof (value-scan finds the period) in favor of cases where the answer is genuinely a global coherent-integration property — a strengthening, documented in the 50.1 report.

## Artifacts

- `49_1_extractive_proof/` — `49_1_extractive_proof.py`, `testbed_synth.py`, `testbed_zeta.py`, `wrong_answer_control.py`, `REPORT_EXTRACTIVE_PROOF.md`, `output.txt`
- `49_2_decodability_gradient/` — `hsp_family.py`, `49_2_decodability_gradient.py`, `49_2_anchor_cospectral.py`, `REPORT_DECODABILITY_GRADIENT.md`, `located_wall.json`, `cospectral_anchor.json`, `output.txt`
- `49_2_decodability_gradient/` (roadmap run) — `49_2d_kuperberg_sieve.py` + `REPORT_KUPERBERG_RUNG.md` + `kuperberg_result.json`; `49_2e_gradient_coverage.py` + `REPORT_GRADIENT_COVERAGE.md` + `coverage_result.json`
- `49_3_boundary_handoff/` — `49_3_boundary_handoff.py`, `MYTHOS_SANDBOX.md`, `EXP44_PHASE6_HANDOFF.md`, `REPORT_BOUNDARY_HANDOFF.md`
- `49_4_lattice_audit/` (roadmap run) — `49_4_lwe_audit.py`, `REPORT_LATTICE_AUDIT.md`, `lattice_audit_result.json`, `output_lattice_audit.txt`
- `49_5_decoder_class_map/` (roadmap run) — `decoder_class_map.py`, `REPORT_DECODER_CLASS_MAP.md`, `decoder_class_map_result.json`, `output_class_map.txt`
- shared: `decoder_lib.py`, `catalytic_tape.py`
