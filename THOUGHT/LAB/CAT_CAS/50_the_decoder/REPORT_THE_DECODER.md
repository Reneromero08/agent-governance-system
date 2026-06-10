# Exp 50 — The Decoder

**Rollup verdict:** the holographic decoder is **extractive** (proven), its extractive power is **bounded at the abelian-HSP wall** (located), and the boundary is handed off as two scoped questions. Whole-experiment claim ceiling: **Level 5**. Lab-critic clean (M-1..M-8). All bricks exit 0.

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

## Combined claim ledger (conservative)

| Claim | Level | Why not higher |
|---|---|---|
| The holographic readout is extractive, not lookup | **4-5** | survives 4 nulls + wrong-answer control; no domain/ontology claim |
| Decodability collapses at the abelian→non-abelian boundary | **4-5** | measured (d=8.82), scale-independent, but the wall's *identity* is a question, not proven |
| The collapse is the known non-abelian-HSP barrier | **NOT CLAIMED** | open question for Mythos |
| Silicon hosts the decodable side | **NOT CLAIMED** | prediction handed to Exp 44 Phase 6 |

## What this gives the lab

A measured operating manual for holographic computing: **the decoder genuinely extracts global invariants (it is not a lookup); the scalar readout decodes the abelian/Fourier class; the non-abelian Fourier reframe extends that to all *normal* hidden subgroups; and the genuine residual wall is the *non-normal* / strong-Fourier-sampling case — the dihedral-HSP ↔ lattice frontier.** The period/Riemann/halting decoders sit safely inside the decodable region. The **LWE/graph-iso "wall-crossing" claims (Exp 25, 31) sit exactly on the relocated wall** — strong sampling / lattice hardness — which is precisely where the cospectral anchor shows the spectral readout fails. So those claims are the ones needing extraordinary evidence. The boundary is now relocated to the *real* barrier and scoped as one sharp question for a stronger model, plus a concrete bare-metal target.

## Grain-of-salt / provisional notes

The arsenal these results build on was largely produced by weaker models (DeepSeek/Gemini); a collapse here may be a barrier OR an uncrossed frontier — which is exactly the Mythos question. The zeta absolute zero-coverage is inflated by peak density (the real signal is the real-vs-scrambled differential). The mod-exp *period* grating was dropped from the extractive proof (value-scan finds the period) in favor of cases where the answer is genuinely a global coherent-integration property — a strengthening, documented in the 50.1 report.

## Artifacts

- `50_1_extractive_proof/` — `50_1_extractive_proof.py`, `testbed_synth.py`, `testbed_zeta.py`, `wrong_answer_control.py`, `REPORT_EXTRACTIVE_PROOF.md`, `output.txt`
- `50_2_decodability_gradient/` — `hsp_family.py`, `50_2_decodability_gradient.py`, `50_2_anchor_cospectral.py`, `REPORT_DECODABILITY_GRADIENT.md`, `located_wall.json`, `cospectral_anchor.json`, `output.txt`
- `50_3_boundary_handoff/` — `50_3_boundary_handoff.py`, `MYTHOS_SANDBOX.md`, `EXP44_PHASE6_HANDOFF.md`, `REPORT_BOUNDARY_HANDOFF.md`
- shared: `decoder_lib.py`, `catalytic_tape.py`
