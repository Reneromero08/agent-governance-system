# MYTHOS BRIEF — the pre-compressed call (Exp 50, post-spiral)

Self-contained brief for a genuine MYTHOS (stronger-model) call. **MYTHOS has not been consulted yet**
— this is the still-un-run roadmap item #3, now sharpened by the Lattice Spiral (50.6-50.14). The
torus/winding/catalytic proposals tested in 50.10-50.13 came from a *NotebookLM session over the
framework's documents*, not from MYTHOS; do not treat them as authoritative.

---

## 0. The problem (self-contained)

A "decoder" encodes a problem as a phase grating and reads the answer as a global spectral/topological
invariant — never by search. Exp 50.1-50.2 proved it is extractive on the abelian-HSP class and
located a wall at the first non-abelian group; strong Fourier sampling (50.2c) identified the residual
wall as **dihedral-HSP / 1-bit-LWE <-> unique-SVP (Regev)** — info-cheap (O(sqrt N) coset states
determine the hidden slope d), compute-hard (recovering d is a 2^n search; Kuperberg is 2^{O(sqrt n)}).

## 1. What the spiral established (forward floor, mapped to the atom)

Eleven adversarial passes (50.6-50.14; full account `../REPORT_LATTICE_SPIRAL.md`). Every "crossing"
verdict was re-checked; one false positive was caught and corrected. Result:

- **d is a conserved topological invariant** — measured. A single global readout recovers it from
  O(sqrt N) cosets (50.10); the Cauchy / point-gap winding equals it exactly with an oracle operator
  (50.12, recovery 1.00). Reading the invariant is FREE; the topological-measurement reframe is right.
- **The secret is the per-step CURVATURE of its own trajectory.** The hopping phase that makes the
  winding equal d is exactly 2*pi*d/N — the trajectory's local holonomy IS d. So every forward readout
  (FFT 50.10, analytic Cauchy contour 50.11, Noether winding 50.12) reads d for free, but *building the
  operator that carries the winding* needs d. 50.11 even removed the candidate scan (poly evaluations),
  yet d is a diffuse 2/N feature so each evaluation costs O(N).
- **No amplification escapes it.** The exceptional point's sqrt-divergence amplifies the d/N curvature
  and the estimation noise by the same factor — the EP-minus-Hermitian recovery gap decays with n
  (50.13). Fisher floor, demonstrated on the bench, not cited.
- **The wall relocated onto the SUBSTRATE (50.14).** d emerges as the unique fixed point of a map
  f(x) = x if verify(x) else (x+1) mod N built from PUBLIC (k,b) alone (no d planted). Forward, finding
  fix(f) is O(N) = 2^n. On a reversible / zero-Landauer / CTC fixed-point computational model (Deutsch
  CTC, P^CTC = PSPACE) it is poly. This is a purely abstract complexity-model question - whether such a
  substrate is physically realizable is explicitly OUT OF SCOPE for this call.

**Not claimed:** any forward crossing. The forward floor is mapped; the residual is genuinely beyond
what more in-lab experiments can settle. That is the MYTHOS trigger.

## 2. The three asks (this is the actual call)

**Q1 — Refute or confirm the forward-wall characterization.** The spiral swept these forward readout
families: scalar FFT, non-abelian Fourier reframe, ring/NTT (conjugate basis), entropy/BKW sieve,
joint two-basis, analytic Cauchy contour, Noether/point-gap winding, exceptional point. Claim: all
collapse to "the secret is the per-step curvature; no fixed forward lens diagonalizes the coset
ensemble." *Name a forward readout family outside that set that could cross, or confirm the sweep is
complete.* A concrete missed lens is the most valuable possible answer.

**Q2 — Theorem or exhaustion?** Elevate (or break) the empirical claim into a statement: *is "the
dihedral slope d is the holonomy of its own trajectory, therefore no secret-independent forward lens
exists" provable* (a structural hardness statement about dihedral-HSP / uSVP read-out), or is it just
"the spiral tried 8 families"? This is the I-couldn't-vs-can't gap and is the heart of the call.

**Q3 — Substrate-reduction soundness (pure complexity theory).** Is 50.14's construction a *legitimate*
reduction — d as the fixed point of a public verifier, giving a poly advantage only on a Deutsch-CTC /
reversible computational model (P^CTC=PSPACE) — or a hidden oracle / smuggle? Audit the no-d-planted
claim and the CTC framing. This is an abstract complexity-class question only; no physical realization
is in scope.

## 3. Constraints (a reframe COUNTS only under these)

- Use the SAME null harness and instances: `../decoder_lib.py`, `../50_2_decodability_gradient/hsp_family.py`,
  and the dihedral coset model in `../50_14_reversible_substrate/`. No bespoke success metric.
- **No smuggle.** A construction that needs d to place the lens/defect/fixed point (e.g. the per-step
  phase 2*pi*d/N, or a pre-seeded answer on the tape) is the secret in disguise, not a decoder — the
  50.4/A1 temporal-bootstrap discipline. Any apparent poly recovery is treated with MAXIMUM A8
  suspicion and must hold under a SCALING test in n before it is more than a regime artifact.
- Claim ceiling L4-5. "Crossed" requires the scaling to hold AND the catalytic tape to verify restored
  (SHA in==out). The honest terminal states are: crossed (with evidence) or characterized (with the
  precise residual). Never "the wall holds," never a faked crossing.

## 4. Files

- `../REPORT_LATTICE_SPIRAL.md` — the full 11-pass account (canonical).
- `../50_14_reversible_substrate/` — the substrate construction (Q3).
- `../50_10_topological_exploit/`, `../50_11_torus_contour/`, `../50_12_noether_winding/`,
  `../50_13_ep_amplification/` — the forward readouts the sweep covers (Q1).
- `MYTHOS_SANDBOX.md` — the located-wall data + null sandbox (§6 carries this same status).

(Scope note: this is a pure mathematics / theory-of-computation call — dihedral-HSP, unique-SVP,
abelian vs non-abelian Fourier sampling, and the reversible/CTC complexity-class question. No hardware,
firmware, or physical-substrate engineering is part of this brief or relevant to answering it.)
