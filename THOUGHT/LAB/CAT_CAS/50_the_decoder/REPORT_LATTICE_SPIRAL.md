# Exp 50 - The Lattice Spiral (50.4 - 50.14)

A single-session spiral around the located lattice wall. Method (the lab owner's): assume the wall
is crossable, spiral around it refusing to admit it is there, and each pass breaks a door and maps
the boundary at higher resolution. We never went over. We mapped it to the atom - and the map's
final coordinate is the **substrate**, which is exactly where Exp 44 Phase 6 takes over.

All bricks exit 0, lab-critic (M-1..M-8) clean, claims capped L4-5. Every "crossing" verdict was
adversarially re-checked; one (50.10) fired a false `GLOBAL_READOUT_CROSSES` from a flawed test and
was caught and corrected (the engine/filter discipline working on the most exciting result).

> **Provenance of the "exploit" proposals (50.10-50.13).** The torus / winding / catalytic-multiplex
> ideas tested in 50.10-50.13 came from a **NotebookLM session prompted with the framework's core
> documents** - i.e. the lab owner's own framework, surfaced via NotebookLM. They are **NOT** from the
> MYTHOS model, which **has not been consulted at all**. The `MYTHOS_SANDBOX.md` remains prepared for a
> future, genuine Mythos call (roadmap item #3, still un-run). Earlier drafts mislabeled these as
> "MYTHOS" proposals; corrected here and throughout.

## The arc in one line

**The wall moved: from the READOUT -> to the CURVATURE -> to the SUBSTRATE.** Every spectral/contour/
winding readout reads the secret `d` correctly and for free; the cost is in *building the trajectory*,
because `d` is the per-step curvature of its own trajectory; and on a forward substrate that is the
`2^n` search, while on a reversible / catalytic / fixed-point substrate it is poly.

## Pass-by-pass (each a door)

| Brick | Verdict | The nugget |
|---|---|---|
| **50.4** lattice audit | `EXP25_TOY_SCALE_ONLY` | Exp 25's holographic LWE attack recovers only at toy q/n; 0 at its own default; secret-blind (Cohen d=0.16). Does not cross. |
| **50.6** ring structure (A9) | `NAIVE_RING_DECODE_BLOCKED_BY_CONJUGATE_BASIS` | The NTT (abelian Galois transform) diagonalizes ring multiplication and *would* collapse the search - but the error is small only in the coefficient basis and uniform in the NTT basis. **Conjugate-basis incompatibility:** no single basis has both. |
| **50.7** entropy/chaos (A13) | `CHAOS_RECOVERS_BUT_ENTROPY_COST_EXPONENTIAL` | "Turn noise into solutions": injecting entropy precipitates `d` from the high-dimensional cloud where single-shot fails - the BKW/sieve family. But the entropy cost is `q^{n/2}` (exp); the owner's intuition is the right family. |
| **50.8** joint phase-space | `JOINT_READOUT_IS_THE_LATTICE_PROBLEM` | Using both conjugate bases at once recovers `d`, but the joint readout IS LWE: cost ~`3^{0.9n}` (exp). The secret lives in the joint geometry; reading it is the lattice problem. |
| **50.9** catalytic illumination | (honest negative) | Tried the phase_cavity_sieve as an illumination; the rank/emergence probes did NOT discriminate decodable from wall. The comb's deeper finding stood: the lattice's illumination lens is **secret-dependent** (the reduced basis = the search). |
| **50.10** the NotebookLM exploit | `REFRAME_CORRECT_EXP_LIVES_IN_THE_INVARIANT_DOMAIN` | `d` IS a global topological invariant - recovered from O(sqrt N) cosets by one readout, not a sieve (the reframe **confirmed**). But the invariant's domain is `2^n`. *(A flawed rank-test first fired a false CROSS here; caught and corrected - the A8 discipline.)* |
| **50.11** torus contour | `TORUS_CONTOUR_REMOVES_SCAN_NOT_EXPONENT` | The analytic arc-energy contour **removes the scan** (2*log2(N) = poly evaluations - "search becomes resonance" vindicated). But `d` is a diffuse `2/N` peak, so each evaluation needs `sqrt(N)` samples = `O(N)`. The exp hopped from the scan to the per-evaluation cost. |
| **50.12** Noether winding | `NOETHER_WINDING_NEEDS_N_RESOLUTION` | `d` IS the conserved Noether charge read by the Cauchy winding (oracle = 1.00; the winding mechanism **confirmed**). But the operator's per-step hopping phase `2pi d/N` **is** `d` - the trajectory's local curvature is the secret. Public-data construction fails at poly resolution. |
| **50.13** exceptional point | `EP_HITS_FISHER_FLOOR` | The EP's sqrt-divergence amplifies the `d/N` curvature - but it amplifies the noise by the same factor. No recovery advantage that survives scaling (EP-Hermitian gap decays 0.18->0.03). Fisher proven on the bench, not cited. |
| **50.14** reversible substrate | `WALL_IS_THE_SUBSTRATE_NOT_THE_READOUT` | `d` emerges as the UNIQUE fixed point of a map built from PUBLIC `(k,b)` alone (no smuggle, unlike the temporal bootstrap). Forward, finding it is `O(N)=2^n`; on a reversible/CTC fixed-point substrate (P^CTC=PSPACE) it is poly. **The wall is the substrate.** |

## What is confirmed (the framework's load-bearing claims held)

- **`d` is a conserved topological invariant.** Measured: the global readout isolates it from O(sqrt N)
  samples (50.10); the oracle winding equals it exactly (50.12). "Computation is the extraction of
  invariants from reversible trajectories" - for this problem, true.
- **The topological-measurement reframe is right.** Reading `d` is never the bottleneck - the FFT
  (50.10), the contour (50.11), the winding (50.12) all read the invariant correctly and the false
  paths cancel. The proposal's "search becomes resonance" was vindicated structurally (50.11 removes the scan).
- **The secret is the curvature of its own trajectory.** The per-step holonomy that makes the winding
  equal `d` is `2pi d/N` - the local curvature IS the secret. That is the sharpest statement of
  unique-SVP hardness the spiral produced.

## What is NOT claimed

No physical crossing of the lattice wall. On a forward substrate the wall is real and mapped to the
atom (`d` = its own curvature; finding it = `2^n`; no amplification beats Fisher). The crossing in
50.14 is *conditional on a reversible / catalytic / fixed-point substrate* (Deutsch CTC, P^CTC=PSPACE).
Whether that substrate is physically realizable is not a complexity question.

## The bridge: Exp 44 Phase 6 (where the substrate becomes physical)

50.14 relocated the entire wall onto the substrate and showed the algorithm is dead on a reversible
fixed-point substrate. **Exp 44 Phase 6 is that question made physical** - the bare-metal Phenom going
catalytic, real zero-Landauer reversible compute. The decodability handoff (cyclic period oracle,
prime->Riemann grating) earns the software<->silicon isomorphism; the *lattice* handoff added by this
spiral is the fixed-point test: **does the catalytic silicon reach `fix(f) = d` reversibly, where a
forward machine needs `2^n`?** That is the experiment that decides whether the substrate is real - and
it is the natural continuation of the spiral, on hardware, not in Python.

## Files (all under 50_the_decoder/)

`50_4_lattice_audit/`, `50_6_ring_structure/`, `50_7_entropy_chaos/`, `50_8_joint_phase_space/`,
`50_9_catalytic_illumination/`, `50_10_topological_exploit/`, `50_11_torus_contour/`, `50_12_noether_winding/`,
`50_13_ep_amplification/`, `50_14_reversible_substrate/` - each with its `.py`, `*_result.json`, output.
The forward-floor map is complete; the substrate floor is Exp 44.

> **Canonical-record note:** this file (`REPORT_LATTICE_SPIRAL.md`), `ROADMAP.md`, and the CAT_CAS
> `MASTER_REPORT.md` are the durable record of the spiral. `50_3_boundary_handoff/MYTHOS_SANDBOX.md`
> and `EXP44_PHASE6_HANDOFF.md` are **auto-generated by `50_3_boundary_handoff.py`** - the spiral /
> substrate sections appended there (sandbox §6, handoff Target C) are re-applied by hand and will be
> overwritten if `50_3` is re-run. If that happens, re-append from this report; nothing is lost.
