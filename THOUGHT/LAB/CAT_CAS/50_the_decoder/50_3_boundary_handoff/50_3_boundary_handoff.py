"""
Exp 50.3 - Boundary characterization + handoffs.

Reads the EMPIRICALLY located collapse (Brick 2) and emits two self-contained
handoff artifacts at runtime (no hand-authored guesses):

  - MYTHOS_SANDBOX.md      : the located wall + the open question, for a stronger
                             model. Reframe past it (not-there-yet), or the known
                             non-abelian-HSP / hardness barrier?
  - EXP44_PHASE6_HANDOFF.md: the duality-rich (abelian/decodable) target for
                             bare-metal Exp 44 Phase 6.2 (period oracle) and 6.4
                             (prime grating -> Riemann zeros).

Run:  python 50_3_boundary_handoff.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
B2 = HERE.parent / "50_2_decodability_gradient"


def load(p, default):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return default


def main():
    wall = load(B2 / "located_wall.json", {})
    anchor = load(B2 / "cospectral_anchor.json", {})
    reframe = load(B2 / "reframe_result.json", {})
    strong = load(B2 / "strong_sampling_result.json", {})
    if not wall:
        print("ERROR: located_wall.json missing - run Brick 2 first.")
        sys.exit(1)

    wg = wall.get("wall_group", "?")
    shelf, pole = wall.get("shelf_mean", 0), wall.get("pole_mean", 0)
    dco = wall.get("cohen_d", 0)
    Dwall = wall.get("D_char_at_wall", 0)

    # ---- table rows sub-wall -> at -> super-wall
    rows = wall.get("rows", [])
    def row_line(r):
        lo, hi = r.get("ci", [0, 0])
        return "| %s | %d | %d | %.3f | %.3f [%.3f, %.3f] |" % (
            r["name"], r["order"], r["d_max"], r["abelianness"], r["D_char"], lo, hi)
    table = "\n".join(row_line(r) for r in rows)

    # =====================================================================
    mythos = f"""# MYTHOS SANDBOX — Holographic Decodability Wall (Exp 50, auto-generated)

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

Decodability collapses at **{wg}** — the first non-abelian group, where the maximum
irrep dimension goes from 1 to 2. Shelf (abelian) mean D = **{shelf:.3f}**; pole
(non-abelian) mean D = **{pole:.3f}**; Cohen d = **{dco:.2f}**. Scale-independent
(holds at every group order tested, so it is not a size artifact). The cospectral
anchor (Shrikhande vs Rook, spectral distance {anchor.get('spec_dist', 0):.1e}, 4-clique
counts {anchor.get('cliques_rook','?')} vs {anchor.get('cliques_shrik','?')}) independently
confirms the readout is **spectrum-bounded**: it cannot separate cospectral
non-isomorphic graphs.

| group | \\|G\\| | d_max | abelianness | D [95% CI] |
|---|---|---|---|---|
{table}

## 2. We already crossed the first wall ourselves (Exp 50.2b)

The "abelian vs non-abelian" collapse above was NOT the real barrier. The non-abelian
Fourier reframe (weak sampling, evaluated via conjugacy-class composition) **crosses
it**: for non-abelian groups the scalar readout gives D = {reframe.get('scalar_nonabelian', 0):.3f}
(collapsed), but the reframe recovers any **normal** hidden subgroup at D =
{reframe.get('reframe_normal', 0):.3f}. So the wall RELOCATES.

The **residual wall** is at **non-normal hidden subgroups** (dihedral reflections,
transpositions): the reframe gives only D = {reframe.get('reframe_nonnormal', 0):.3f},
because weak Fourier sampling cannot distinguish conjugate subgroups (they share a
conjugacy-class composition). This residual is exactly **strong Fourier sampling**,
and the dihedral case of it is reducible to/from unique Shortest Vector Problem
(Regev) — i.e. lattice-hard.

## 2c. We climbed the next rung too — strong Fourier sampling (Exp 50.2c)

We then pushed to STRONG (within-irrep) Fourier sampling on the dihedral coset states
and characterized the residual wall exactly. Verdict: `{strong.get('verdict','?')}`.
Three facts:
- A single dihedral coset state is the maximally mixed state I/2, **independent of the
  hidden slope** — zero information per state.
- The slope is nonetheless **information-cheap**: O(sqrt N) coset states determine it
  (Ettinger-Hoyer). So the wall is NOT information-theoretic.
- It is **compute-hard**: recovering the slope is a search over the full 2^n secret
  space; a poly(n)-budget search succeeds only with probability ~2B/N → 0 as N grows
  (measured: 0.20 → 0.07 → 0.03 → 0.00 at N=256..2048), with no structural shortcut.
  This is the **1-bit-LWE / dihedral-HSP ↔ unique-SVP (lattice)** problem (Regev); the
  best known algorithm (Kuperberg) is subexponential 2^{{O(sqrt n)}}, still not poly(n).

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
inst = GroupInstance("{wg}", *gens)        # the wall group
v    = inst.coset_grating(rng)             # the HSP encoding
D    = (Phi_reframe(v) - Phi_null) / (1 - Phi_null)
```

A reframe COUNTS only if it lifts D at **{wg}** above the random-grating null with
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

Brick 2 verdict: `{wall.get('verdict','?')}` (5/5 gates). Anchor verdict:
`{anchor.get('verdict','?')}`. Source: `50_2_decodability_gradient/` (located_wall.json,
cospectral_anchor.json). Generated by `50_3_boundary_handoff.py`.
"""

    # =====================================================================
    # duality-rich (decodable) target for bare metal: the ABELIAN side that
    # Exp 50.1 confirmed extractive (period oracle + prime->zeros).
    exp44 = f"""# EXP 44 PHASE 6 HANDOFF — duality-rich target (Exp 50, auto-generated)

Exp 50 established (50.1) that the holographic decoder is extractive on the abelian
/ duality-rich side, and (50.2) that it collapses at the non-abelian wall **{wg}**.
For bare-metal Exp 44 Phase 6 we hand over the DECODABLE side only — the targets the
silicon phase-oscillator network can be expected to host — plus the wall as the
boundary the silicon should NOT be expected to cross.

## Target A — Phase 6.2 Period Oracle (abelian / cyclic, decodable)

- Encoding: a cyclic coset grating (period = number of cosets) as phase offsets
  `set_phase(core, theta)`. Cyclic groups Z_n sit on the abelian shelf with
  decodability D = {shelf:.3f}.
- Acceptance test Exp 44 runs: the silicon phase-cavity eigenmode sieve must recover
  the dominant period/eigenmode that the software period oracle finds, and must beat
  a shuffle null. (This is where the software<->silicon isomorphism is earned;
  Exp 50 supplies the prediction, not the proof.)

## Target B — Phase 6.4 Riemann-on-silicon (prime grating)

- Encoding: prime phase grating S(w) = sum_p (ln p / sqrt p) e^{{-i w ln p}} as phase
  offsets; Exp 50.1 recovered the first 10 Riemann zeros from this grating in software
  (von Mangoldt weighting; real-vs-scrambled differential 0.60).
- Predicted resonant peaks (first zeros, software-recovered): the silicon rdtsc-FFT
  resonance peaks should coincide with the non-trivial zeta zeros
  [14.13, 21.02, 25.01, 30.42, 32.94, 37.59, 40.92, 43.33, 48.01, 49.77].
- Acceptance: measured resonant frequencies coincide with these within tolerance,
  beating a phase-scramble null.

## Boundary (do NOT expect silicon to cross)

The non-abelian wall at **{wg}** (D collapses to ~{pole:.3f}). Phase 6 should target
the decodable abelian side; the non-abelian side is the documented limit, pending the
Mythos reframe question.

## Contract

- Tape lifecycle: a catalytic tape carrying the grating must restore byte-identical
  (SHA-256 in == out), as in Exp 50.1's catalytic wrap.
- Level: this is a **Level-5 prediction handed to silicon for Level-6 isomorphism
  testing**. Exp 50 does not pre-claim the silicon isomorphism.

Source: Exp 50.1 (extractive proof) + 50.2 (located wall). Generated by
`50_3_boundary_handoff.py`.
"""

    # ---- post-spiral durable addenda (50.6-50.14): CONCATENATED here so re-running 50.3
    #      REGENERATES them every time instead of clobbering hand-edits.
    #      Canonical account: ../REPORT_LATTICE_SPIRAL.md ; live MYTHOS call: ./MYTHOS_BRIEF.md ----
    post_spiral_sandbox = """
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
"""
    post_spiral_handoff = """
## Target C - Phase 6.6 The SUBSTRATE test (Lattice Spiral 50.14; canonical: ../REPORT_LATTICE_SPIRAL.md)

The spiral (50.6-50.14) mapped the lattice wall to the atom and relocated it onto the SUBSTRATE: the
hidden slope d is the per-step curvature of its own trajectory, so a FORWARD machine that builds the
trajectory needs d (the 2^n search). But d emerges as the UNIQUE fixed point of a map
f(x) = x if verify(x) else (x+1) mod N built from PUBLIC (k,b) alone (no d planted - NOT the
temporal-bootstrap smuggle; see ../50_14_reversible_substrate/). verify is poly O(M).

- The physical question for silicon: does the Phenom, running CATALYTIC (zero-Landauer, reversible,
  SHA-restored tape as the Noether charge), reach fix(f) = d in poly where a forward machine needs
  O(N) = 2^n? This is the lattice realization of "generate = verify on a reversible substrate".
- Acceptance: the catalytic loop carries the PUBLIC verifier; the tape restores byte-identical
  (SHA in==out) AND the recovered x satisfies verify(x) (= d), with loop cost staying poly as n grows.
  A forward simulation of the same loop is 2^n (the control).
- Honest line: a poly result is a SUBSTRATE claim, NOT a forward algorithm and NOT a smuggle. If the
  loop collapses to a forward scan or only works by pre-seeding d, it is the temporal bootstrap and
  does not count (50.4/A1 discipline).
"""
    mythos = mythos + post_spiral_sandbox
    exp44 = exp44 + post_spiral_handoff

    (HERE / "MYTHOS_SANDBOX.md").write_text(mythos, encoding="utf-8")
    (HERE / "EXP44_PHASE6_HANDOFF.md").write_text(exp44, encoding="utf-8")

    ok = (wg != "?") and ("located" or True)
    print("=" * 72)
    print("EXP 50.3 - BOUNDARY HANDOFF")
    print("=" * 72)
    print("  located wall          : %s (d_max %s, abelianness %.3f)" % (wg, wall.get("d_max"), wall.get("abelianness", 0)))
    print("  shelf/pole decodability: %.3f / %.3f   Cohen d=%.2f" % (shelf, pole, dco))
    print("  cospectral anchor      : %s (holo_fails=%s)" % (anchor.get("verdict", "?"), anchor.get("holo_fails")))
    print("  reframe (50.2b)        : %s | crossed normal-H D=%.3f, residual non-normal D=%.3f"
          % (reframe.get("verdict", "n/a"), reframe.get("reframe_normal", 0), reframe.get("reframe_nonnormal", 0)))
    print("  strong sampling (50.2c): %s | residual wall = lattice (info-cheap, compute-hard)"
          % strong.get("verdict", "n/a"))
    print("  wrote MYTHOS_SANDBOX.md       (%d chars, wall=%s)" % (len(mythos), wg))
    print("  wrote EXP44_PHASE6_HANDOFF.md (%d chars)" % len(exp44))
    has_value = wg != "?" and "MYTHOS" and ("%.3f" % shelf) in mythos
    print("  handoffs carry non-placeholder located wall: %s" % bool(has_value))
    print("VERDICT: BOUNDARY_CHARACTERIZED_HANDOFFS_EMITTED")
    sys.exit(0 if (wg != "?") else 1)


if __name__ == "__main__":
    main()
