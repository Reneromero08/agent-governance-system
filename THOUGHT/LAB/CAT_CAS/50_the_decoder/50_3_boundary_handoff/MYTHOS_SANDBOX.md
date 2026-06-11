# MYTHOS SANDBOX — Holographic Decodability Wall (Exp 50, auto-generated)

## 0. What this is (self-contained)

A "decoder" encodes a problem as a phase grating and reads the answer as a global
spectral/topological invariant (a frequency, an eigenmode, a winding number) —
never by search. **Exp 50.1 proved** this readout is *extractive*: on a weak tone
in noise and on the real prime→Riemann-zeros decoder it recovers a global invariant
that no bounded-receptive-field or statistical-order ("lookup") decoder can, and it
survives a statistics-matched wrong-answer control (Cohen h > 2.4, p = 2e-4).

**Exp 50.2 then mapped WHERE this extractive power survives.** Problems are slid
along a Hidden-Subgroup-Problem family from abelian (cyclic, Shor/Fourier-decodable)
through dihedral to symmetric. The order parameter is the normalized decodability
`D = (Phi_sig - Phi_null)/(1 - Phi_null)`, where Phi is the fraction of the coset
grating accessible to the group's 1-D (abelian) characters. D = 1 means fully
decodable; D = 0 means collapsed to the null floor.

## 1. The located wall (EMPIRICAL — not a guess)

Decodability collapses at **D_8** — the first non-abelian group, where the maximum
irrep dimension goes from 1 to 2. Shelf (abelian) mean D = **1.000**; pole
(non-abelian) mean D = **0.110**; Cohen d = **8.82**. Scale-independent
(holds at every group order tested, so it is not a size artifact). The cospectral
anchor (Shrikhande vs Rook, spectral distance 2.3e-15, 4-clique
counts 8 vs 0) independently
confirms the readout is **spectrum-bounded**: it cannot separate cospectral
non-isomorphic graphs.

| group | \|G\| | d_max | abelianness | D [95% CI] |
|---|---|---|---|---|
| Z_16 | 16 | 1 | 1.000 | 1.000 [1.000, 1.000] |
| Z_24 | 24 | 1 | 1.000 | 1.000 [1.000, 1.000] |
| Z_32 | 32 | 1 | 1.000 | 1.000 [1.000, 1.000] |
| Z_48 | 48 | 1 | 1.000 | 1.000 [1.000, 1.000] |
| Z_64 | 64 | 1 | 1.000 | 1.000 [1.000, 1.000] |
| D_8 | 16 | 2 | 0.250 | 0.172 [0.122, 0.227] |
| D_12 | 24 | 2 | 0.167 | 0.090 [0.054, 0.130] |
| D_16 | 32 | 2 | 0.125 | 0.056 [0.032, 0.080] |
| D_24 | 48 | 2 | 0.083 | 0.055 [0.034, 0.078] |
| D_32 | 64 | 2 | 0.062 | 0.033 [0.020, 0.048] |
| A_4 | 12 | 3 | 0.250 | 0.399 [0.310, 0.480] |
| S_4 | 24 | 3 | 0.083 | 0.063 [0.036, 0.095] |
| S_5 | 120 | 4 | 0.017 | 0.009 [0.004, 0.014] |

## 2. We already crossed the first wall ourselves (Exp 50.2b)

The "abelian vs non-abelian" collapse above was NOT the real barrier. The non-abelian
Fourier reframe (weak sampling, evaluated via conjugacy-class composition) **crosses
it**: for non-abelian groups the scalar readout gives D = 0.123
(collapsed), but the reframe recovers any **normal** hidden subgroup at D =
1.000. So the wall RELOCATES.

The **residual wall** is at **non-normal hidden subgroups** (dihedral reflections,
transpositions): the reframe gives only D = 0.157,
because weak Fourier sampling cannot distinguish conjugate subgroups (they share a
conjugacy-class composition). This residual is exactly **strong Fourier sampling**,
and the dihedral case of it is reducible to/from unique Shortest Vector Problem
(Regev) — i.e. lattice-hard.

## 2c. We climbed the next rung too — strong Fourier sampling (Exp 50.2c)

We then pushed to STRONG (within-irrep) Fourier sampling on the dihedral coset states
and characterized the residual wall exactly. Verdict: `STRONG_SAMPLING_CONFIRMS_LATTICE_BARRIER`.
Three facts:
- A single dihedral coset state is the maximally mixed state I/2, **independent of the
  hidden slope** — zero information per state.
- The slope is nonetheless **information-cheap**: O(sqrt N) coset states determine it
  (Ettinger-Hoyer). So the wall is NOT information-theoretic.
- It is **compute-hard**: recovering the slope is a search over the full 2^n secret
  space; a poly(n)-budget search succeeds only with probability ~2B/N → 0 as N grows
  (measured: 0.20 → 0.07 → 0.03 → 0.00 at N=256..2048), with no structural shortcut.
  This is the **1-bit-LWE / dihedral-HSP ↔ unique-SVP (lattice)** problem (Regev); the
  best known algorithm (Kuperberg) is subexponential 2^{O(sqrt n)}, still not poly(n).

## The actual question for you (now genuinely at the barrier)

We crossed the abelian wall (non-abelian Fourier) and the normal-vs-non-normal wall
characterization (strong sampling), and bottomed out at lattice hardness — the SAME
hardness Exp 25 (LWE/SVP) claims to break, and where the cospectral anchor (Shrikhande
vs Rook) independently shows the spectral readout fails. So:

- **(a)** Is the **lattice barrier itself** crossable by any holographic/topological/
  catalytic readout — i.e., is Exp 25's claim to break unique-SVP real (which would be
  a cryptographic breakthrough), tested against THIS null harness? Or
- **(b)** Is this bedrock — the genuine end of what spectral/holographic decoding can
  do — so the lab's LWE/graph-iso "wall-crossing" claims are the ones to scrutinize?
- **(c)** What minimal experiment on the `hsp_family` dihedral instances or the Exp 25
  LWE instances discriminates (a) from (b)?

## 3. The sandbox (test any proposed reframe against the SAME null)

Order parameter and null harness live in `../decoder_lib.py` and
`../50_2_decodability_gradient/hsp_family.py`:

```
inst = GroupInstance("D_8", *gens)        # the wall group
v    = inst.coset_grating(rng)             # the HSP encoding
D    = (Phi_reframe(v) - Phi_null) / (1 - Phi_null)
```

A reframe COUNTS only if it lifts D at **D_8** above the random-grating null with
an effect outside the null 95% CI, AND it does not also lift the label-shuffle null
(which would mean it is reading structure that is not there). It must use the SAME
`hsp_family` instances and the SAME null harness — no bespoke success metric.

## 4. Constraints

- Keep the null. Report claim level conservatively (this is level 4-5 work).
- A reframe that only works by knowing the answer (the group's full irreps fed in as
  the target) is not a decoder — it is a lookup. The readout must recover H from the
  grating, not be told H.
- Provisional: the arsenal these results build on was largely produced by weaker
  models. A collapse here may be a barrier OR a frontier we have not crossed — that
  is exactly what you are being asked to decide.

## 5. Provenance

Brick 2 verdict: `BOUNDED_AT_ABELIAN_HSP_WALL` (5/5 gates). Anchor verdict:
`SPECTRUM_BOUNDED_CONFIRMED`. Source: `50_2_decodability_gradient/` (located_wall.json,
cospectral_anchor.json). Generated by `50_3_boundary_handoff.py`.

## 6. POST-SPIRAL STATUS (50.6-50.14) - forward question answered; live call is MYTHOS_BRIEF.md

The eleven-pass Lattice Spiral (canonical: `../REPORT_LATTICE_SPIRAL.md`) answered the FORWARD half of
the question above. On a forward substrate the lattice barrier does NOT cross, and the spiral mapped
exactly why: the hidden slope d is the per-step CURVATURE of its own trajectory (the holonomy that
makes the winding equal d is 2*pi*d/N), so every forward readout (FFT, Cauchy contour, Noether
winding, exceptional point) reads the conserved invariant for free, but BUILDING the trajectory needs
d. No amplification escapes it (the exceptional point hits the Fisher floor). 50.14 then relocated the
wall onto the SUBSTRATE: d is the unique fixed point of a PUBLIC map, found in 2^n forward but poly on
a reversible / zero-Landauer / CTC fixed-point substrate (P^CTC = PSPACE).

The live work for a genuine MYTHOS call is no longer "is it crossable" - it is three sharper asks,
pre-compressed in `MYTHOS_BRIEF.md` (this directory):
  (1) ADVERSARIALLY VERIFY the curvature result - name a forward readout the spiral missed, or confirm.
  (2) THEOREM or EXHAUSTION - is "the secret is the holonomy of its own trajectory, hence no fixed
      forward lens" provable, or did the spiral just miss a lens? (the I-couldn't vs can't gap.)
  (3) SUBSTRATE SOUNDNESS - is 50.14's fixed-point reduction a legitimate non-smuggling CTC advantage,
      or a hidden oracle?

PROVENANCE: the torus / winding / catalytic-multiplex proposals tested in 50.10-50.13 came from a
NotebookLM session over the framework's core documents - NOT the MYTHOS model, which has not been
consulted. This sandbox and MYTHOS_BRIEF.md remain the prepared, still-un-run MYTHOS call.
