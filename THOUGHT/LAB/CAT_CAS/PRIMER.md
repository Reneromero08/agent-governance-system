# PRIMER.md - What To Load, In What Order, And Why

Companion to `AGENTS.md`. AGENTS tells an agent **how to operate**; this tells it
**what to read to phase-lock**. Repo paths are relative to
`D:\CCC 2.0\AI\agent-governance-system`. Vault paths are absolute
(`C:\Users\rene_\Documents\Shizzle Obsidian\...`).

---

## 1. How priming works here (read this before grabbing files)

There are **two kinds of prime, and you need both - vision first.**

- **The Generative prime (the engine).** The grand vision and ontology. This is what
  *frames* a problem so it becomes solvable in this system's terms. It is FUEL, not
  scaffolding. A lean engineering-only prime produces a competent technician who never
  thinks to frame a Hidden Subgroup Problem as "the decoder," never connects it to the
  holographic thesis, and never builds Exp 50. The vision is load-bearing for the *work*,
  not only the philosophy.
- **The Operational prime (the filter).** How to work: the discipline, the critic, the
  claim ladder, the lab map. This is what *harvests* what the vision generates and keeps
  it honest.

**Order: vision first, then filter.** Prime the altitude, then bring the rigor.
**Scope the depth to the task:** load the relevant half (CAT_CAS engineering vs
FORMULA/semiotic) plus only the experiment reports the task actually touches. Beyond
~5-6 documents, "read everything, do not truncate" is counterproductive - attention
degrades; curate instead.

The validation of the whole vision is not line-by-line proof of its claims. It is
**fertility** (Lakatos: a research programme is judged by whether it keeps generating
results that survive). Exp 50 is the proof the engine runs. So prime the engine at full
altitude; let the critic harvest.

---

## 2. The bundles (grab one)

Lab files use repo-relative paths (root `D:\CCC 2.0\AI\agent-governance-system`); vault
files use full absolute paths.

### Bundle 0 - Minimal phase-lock (ALWAYS, ~5 docs)
The distilled engine + filter. Gets ~90% of the lock at a fraction of the tokens.
1. `THOUGHT/LAB/CAT_CAS/AGENTS.md` - how to operate; the two modes.
2. `C:\Users\rene_\Documents\Shizzle Obsidian\Shizzle\AGI\AGS\WIP\Explainers\My System\♥ ♥ What Is My System - FINAL.md` - the exact, unified mental model.
3. `THOUGHT/LAB/CAT_CAS/CAT_CAS_OS.md` - the verification discipline / claim ladder.
4. `C:\Users\rene_\Documents\Shizzle Obsidian\Shizzle\Personal\Daily Notes\2026-05-21 Catalytic Time.md` - the raw intuition / how the lab-owner thinks (punches above its weight).
5. `THOUGHT/LAB/FORMULA/v4/SEMIOTIC_LIGHT_CONE_1_1/03_SEMIOTIC_WAVE_MECHANICS.md` - the single most precise + generative idea (meaning is the interference cross-term).

### Bundle E - Engineering build in CAT_CAS (the usual case)
**Bundle 0** + the operational set + task-scoped reports + the in-lab code map.
- Operational: `THOUGHT/LAB/CAT_CAS/MANIFESTO.md` (the critic M-1..M-8),
  `THOUGHT/LAB/CAT_CAS/README.md` (inventory), `THOUGHT/LAB/CAT_CAS/MASTER_REPORT.md` (real status).
- Code map: `AGENTS.md` Section 5 (venv, in-lab constraint, M-4 trap, the WORKING holo
  lives in `THOUGHT/LAB/HOLO/pipeline/02_cavity/` and
  `THOUGHT/LAB/CAT_CAS/34_zeta_eigenbasis/02_holographic_sieves/8_riemann_harmonic_sieve.py`,
  NOT the dead-pathed `THOUGHT/DEPRECATED/TINY_COMPRESS/holographic-image/holo_core.py`).
- Task reports: load only what you touch (e.g.
  `THOUGHT/LAB/CAT_CAS/34_zeta_eigenbasis/REPORT_RIEMANN.md` for decoder work;
  `THOUGHT/LAB/CAT_CAS/50_the_decoder/REPORT_THE_DECODER.md` + `ROADMAP.md` for
  decodability/lattice work).
- **Do I need the theoretic primer when building CAT_CAS?** Yes, a thin generative slice -
  but it is already inside Bundle 0 (FINAL + Wave Mechanics + Catalytic Time). Those three
  are cheap and carry the altitude that produced Exp 50. You do NOT need the full
  FORMALIZATION / light-cone / validation stack to build.

### Bundle T - Theoretic / FORMULA / semiotic work
**Bundle 0** + the semiotic stack:
- `THOUGHT/LAB/FORMULA/v4/SEMIOTIC_LIGHT_CONE_1_1/` (all 8; `01_FORMULA_5_2` and
  `03_SEMIOTIC_WAVE_MECHANICS` are the core).
- `THOUGHT/LAB/FORMULA/v4/VALIDATION_ROADMAP.md` - the empirical spine (keep this one; it
  is what makes the formula more than philosophy).
- `THOUGHT/LAB/FORMULA/v2_2/INDEX.md` - the 57-question status ledger (confirm v4 has not
  superseded it).
- `THOUGHT/LAB/FORMULA/v4/FORMALIZATION/` (skip the `REFERENCES/` subfolder) - only when
  auditing/extending the math. **Caveat in the catalog (3.3).**
- `C:\Users\rene_\Documents\Shizzle Obsidian\Shizzle\AGI\AGS\WIP\Explainers\Hyperdimensional Computing.md`
  and `C:\Users\rene_\Documents\Shizzle Obsidian\Shizzle\AGI\AGS\WIP\Explainers\Torus Is The Key.md`
  - the HDC and torus explainers.

### Bundle V - Full vision (max altitude; writing a synthesis, not routine building)
Everything in T plus the heavy reports:
`THOUGHT/LAB/CAT_CAS/41_toe_bulletproof/PAPER_TOPOLOGICAL_THEORY_OF_EVERYTHING_4.md`, the
`THOUGHT/LAB/CAT_CAS/42_computational_event_horizon/REPORT_EXP42_MASTER.md`,
`THOUGHT/LAB/CAT_CAS/45_phase_math/MASTER_REPORT_PHASE_45.md`, and
`THOUGHT/LAB/CAT_CAS/46_phase_bio/MASTER_REPORT_EXP_46.md`. Use when you want maximum
generative breadth. (Curation note: the 42-cluster's per-experiment and per-cluster reports
were consolidated into one honest master - `REPORT_EXP42_MASTER.md` (all 24 sub-experiments,
each re-derived from source) - read that single file instead of the old five;
`PAPER_TOPOLOGICAL_THEORY_OF_EVERYTHING_4.md` is ~1264 lines, page it.)

---

## 3. The full reference catalog (by priority)

Priority: **P0** = always; **P1** = load for the relevant task; **P2** = deep / specific.

### 3.1 Operational - the filter (repo: `THOUGHT/LAB/CAT_CAS/`)

| Doc | Explains | Why it matters | When | P |
|---|---|---|---|---|
| `AGENTS.md` | How to operate; exploration vs verification mode | Prevents whiplash between vision and critic; the entry point | Always, first | P0 |
| `CAT_CAS_OS.md` | Agent OS: prime directive, claim hierarchy (L0-8), failure hierarchy, verification standards, the oath | The discipline that lets you hold the vision without inflating it | Always; reread before claiming DONE/verified | P0 |
| `MANIFESTO.md` | Core primitives + the mechanical critic M-1..M-8 | The rules your code must pass; the anti-ceremonial law | Before building / committing | P1 |
| `README.md` | Experiment inventory 01-50, navigation | Find what exists; avoid reinventing | Orienting; locating prior work | P1 |
| `master_report.md` | Compact truth ledger; verified vs partial vs deprecated status | The REAL status of each result; never trust a report without checking here | Assessing an experiment's standing | P1 |
| `50_the_decoder/ROADMAP.md` | Exp 50 next steps (point lattice wall at Exp 25, Mythos, Exp 44 handoff) | The live frontier of the decoder line | Continuing decoder work | P1 |

### 3.2 The mental model - the distilled vision (vault base: `C:\Users\rene_\Documents\Shizzle Obsidian\Shizzle\`)

| Doc | Explains | Why it matters | When | P |
|---|---|---|---|---|
| `AGI\AGS\WIP\Explainers\My System\♥ ♥ What Is My System - FINAL.md` | The canonical, exact statement of the whole system | Fastest path to the unified model; supersedes the 7 prior notes | Always, for orientation | P0 |
| `Personal\Daily Notes\2026-05-21 Catalytic Time.md` | Raw eureka: time as catalytic computing on a black hole tape | Shows how the lab-owner thinks; the intuition behind catalysis-as-primitive | To lock on the voice/intuition | P0 |
| `AGI\AGS\WIP\Explainers\My System\` (7 prior notes: `_2`, `_3 Science`, `_4`, `Holographic`, `Torus`, `Entropy`) | Prior drafts of the thesis (`_4` = the Newton/Einstein compression; `_3` = the science stack + translation table) | Historical; FINAL supersedes them, but `_3` and `_4` are still excellent | Rarely; FINAL is canonical | P2 |

### 3.3 Generative - the semiotic engine (repo: `THOUGHT/LAB/FORMULA/`)

| Doc | Explains | Why it matters | When | P |
|---|---|---|---|---|
| `v4/SEMIOTIC_LIGHT_CONE_1_1/03_SEMIOTIC_WAVE_MECHANICS.md` | Meaning as wave interference; the cross-term Shannon discards; perspective = path difference; Kuramoto; standing waves | The most precise + generative single doc; where the vision becomes falsifiable | Any meaning/semiotic framing | P0/P1 |
| `.../01_FORMULA_5_2.md` | The Living Formula `R=(E/grad_S)*sigma^Df`, every term operationalized; QD grounding | The core equation | Applying the formula in any domain | P1 |
| `.../04_EINSTEIN_MEANING_SPACE.md` | Semiotic gravity; meaning as curvature; the GR isomorph | The "meaning causes geodesics / information has weight" architecture - a core intuition | The gravity/curvature thread | P1 |
| `.../02_THE_SEMIOTIC_AXIOMS_2_2.md` | The 9 axioms in complex Hilbert space | The formal grammar of the meaning half | Formalizing/extending the semiotics | P2 |
| `.../07_CYBERNETIC_TRUTH.md` | Truth as a cybernetic attractor; the `T=1/(R+eps)` control law | The truth-navigation / control architecture | Alignment, truth-monitor work | P2 |
| `.../06_ALIGNMENT_PROBLEM.md` | AI alignment as resonance (constitution as attractor, not fence) | The alignment thesis + the prospective experiment | AI alignment thread | P2 |
| `.../08_CONSCIOUSNESS_THEORY_COMPARISON.md` | Consciousness as phase coherence; 29 theories mapped | The consciousness proposition | Consciousness thread | P2 |
| `v4/VALIDATION_ROADMAP.md` | The empirical ledger: QEC sweep (R2=0.94), alignment experiments, cybernetic truth monitor | The ONLY place the formula touches data; the empirical spine | FORMULA verification; assessing what is grounded | P1 (T) |
| `v2_2/INDEX.md` | 57-question status ledger (VERIFIED/CONFIRMED/FALSIFIED/OPEN) | FORMULA's `master_report`; what is settled vs open | FORMULA verification; before re-litigating | P2 |
| `v4/FORMALIZATION/` (GR_DERIVATION, SEMIOTIC_ACTION_PRINCIPLE, FULL_EINSTEIN_VERIFIED, RESOLUTION_HBAR_SEM, UNDECIDABILITY_AS_TOPOLOGICAL_PHASE_TRANSITION, ...) **- skip the `REFERENCES/` subfolder** | The math hardening: GR from a semiotic action, hbar_sem, undecidability as a phase transition | Where the framework is most rigorous AND most exposed | Auditing/extending the formalization | P2* |

\* **Caveat:** `FULL_EINSTEIN_VERIFIED` (full Einstein field eqns on meaning-space,
r=0.95 across 4 embedding models) is the boldest claim and needs an adversarial audit -
a high R2 across four embedding spaces can ride a shared spectral confound. Prime it for
*scrutiny*, not belief. The defensible beachhead for "meaning causes geodesics" is the
**semantic-geodesic result** (truth follows shorter geodesics, lies cost more action) in
`VALIDATION_ROADMAP.md` / Q38 - it stands without the r=0.95 claim.

### 3.4 Generative - the grand-vision reports (repo: `THOUGHT/LAB/CAT_CAS/`)

| Doc | Explains | Why it matters | When | P |
|---|---|---|---|---|
| `34_zeta_eigenbasis/REPORT_RIEMANN.md` | Zeta zeros as eigenvalues; the prime->zeros extraction mechanism | The decoder's foundational mechanism; grounded Exp 50 | Any decoder/oracle/spectral-readout work | P1 |
| `45_phase_math/MASTER_REPORT_PHASE_45.md` | Millennium problems via topological sensors (Collatz, Navier-Stokes, Riemann, P vs NP, Yang-Mills) | The cleanest demonstration of the method + the discipline; the actual sensor techniques | Math/sensor work; to see the method applied rigorously | P1 |
| `46_phase_bio/MASTER_REPORT_EXP_46.md` | Topological biology (protein folding, genetic code, neural binding, morphogenesis) | Breadth + the honest 46.3 claim-weakening (proof the self-correction is real) | Bio thread; to see status discipline | P1 |
| `41_toe_bulletproof/PAPER_TOPOLOGICAL_THEORY_OF_EVERYTHING_4.md` | The TOE synthesis: undecidability as non-Hermitian resonance on a catalytic substrate | The deepest unification statement; sets max altitude | When you need the full ontological frame | P2 |
| `42_computational_event_horizon/REPORT_EXP42_MASTER.md` | Computation-as-physics: floating-point black holes, cosmology-as-malloc, QM-GR - all 24 sub-experiments consolidated, each re-derived from source | The boldest "computation IS physics" claims as altitude fuel, paired with an honest claim-vs-code audit (what is real computing vs metaphor) | Ontology/cosmology thread | P2 |

**Note on 42:** the old cluster was five redundant reports all hammering the same thesis.
They are now consolidated into one master - `REPORT_EXP42_MASTER.md` - which covers every
sub-experiment AND flags the over-claims (e.g. the 42.4 hardcoded-entropy Page curve). Read
that single file, not the old five. `PAPER_TOE_4` is ~1264 lines; page it, do not try to
swallow it whole.

### 3.5 Explainers (vault: `C:\Users\rene_\Documents\Shizzle Obsidian\Shizzle\AGI\AGS\WIP\Explainers\`)

| Doc | Explains | Why it matters | When | P |
|---|---|---|---|---|
| `Torus Is The Key.md` | Torus mapping `exp(i 2pi W/q)`; the phase-cavity sieve; geometric sigma | The practical holographic-compression intuition; how data maps to S^1 | Holo/torus/compression work | P1 |
| `Hyperdimensional Computing.md` | Kanerva HDC vs the system; what is novel (SVh codebook, U@SVh binding, GOE orthogonality) | Situates the system vs known frameworks; the "what it is NOT" clarity | Comparing to HDC/FHRR; positioning | P2 |

### 3.6 What was built this session

| Doc | Explains | Why it matters | When | P |
|---|---|---|---|---|
| `50_the_decoder/REPORT_THE_DECODER.md` | The decoder proven *extractive*; decodable = abelian-HSP + topological invariants; irreducible wall = lattice (LWE/unique-SVP) | The measured characterization of the crux ("the decoder"); relocates Exp 25/31 claims onto the bedrock wall | Any decoder / holographic-computing / Exp-25 work | P1 |
| `50_the_decoder/MYTHOS_SANDBOX.md` (in `50_3_boundary_handoff/`) | The sharpened open question (is the lattice barrier crossable?), bound to the null harness | The one question worth a stronger-model call | When escalating to Mythos | P1 |

---

## 4. Building notes (hard-won)

- **Vision first, then filter.** Do not trim the generative docs to save tokens; trim
  the operational scaffolding. The engine produced Exp 50; the filter only checked it.
- **Run things; let the data correct the theory.** In Exp 50 the data overturned the
  design twice (a "period" testbed that was secretly a lookup; a sample-complexity
  barrier that was really a compute barrier). That is the method working.
- **Build a null that can KILL the claim** (wrong-answer control; budgeted search). A
  strawman null proves nothing.
- **Reuse the WORKING in-lab code.** Stale import path = "not wired up," not "method
  failed." Live holo: `THOUGHT/LAB/HOLO/pipeline/02_cavity/`,
  `34.../8_riemann_harmonic_sieve.py`. Dead: `THOUGHT/DEPRECATED/.../holo_core.py`.
- **M-4 critic trap:** a `.py` containing both `(SAT|satisfiab|NP-complete)` and an
  `(N,N)` literal fails. Frame NP-complete poles group-theoretically; use `dim_G`/`(D,D)`.
- **Stay in lab.** Work under `THOUGHT/LAB/CAT_CAS/`; do not modify the main AGS repo.
- **Cap claims at the supported level; handoffs are questions, not claims.**
- The boldest reports (42 master, FULL_EINSTEIN) are *altitude fuel and audit targets*,
  not settled facts. Read them to generate, not to believe - `REPORT_EXP42_MASTER.md`
  already does the audit pass and marks which claims are real computing vs metaphor.

---

## 5. The one line

> Prime the vision first (Bundle 0), bring the filter for the task (E or T), load only
> the reports you touch, and remember the engine is judged by what it generates - which
> is why the theoretic primer is not optional even when you are "just" building.
