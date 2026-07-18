# CAT_CAS Conventions

Single source of truth for how this lab is named and laid out. If a directory or
file disagrees with this document, the directory is wrong. Linked from
[PRIMER.md](../PRIMER.md) and [README.md](../README.md).

---

## 1. Directory layout

```
CAT_CAS/
  _lib/                  shared primitives (one copy each; see section 6)
  control/               machine-readable mission, state, branches, capabilities
  templates/             phase-lock, task, and experiment contract templates
  tools/                 phase-lock compiler and control-plane validators
  tests/control_plane/   cold-agent and structural control-plane tests
  docs/                  navigation + conventions; holds REPORTS/ (audit ledger)
  N_track_name/          thematic tracks (section 2) containing experiments
  workspace/             shared experiment fixtures (exp 02/03)
  MISSION.md CAPABILITY_GRAPH.md AGENTS.md CAT_CAS_OS.md MANIFESTO.md
  MASTER_REPORT.md PRIMER.md README.md
```

The leading-underscore folders (`_lib`) sort above the numbered tracks and signal
"infrastructure, not an experiment." `control/` is the phase-lock control plane, not
scientific evidence. Its current-state claims must point to experiment evidence. The
visualizer was promoted out to its own sibling lab at `THOUGHT/LAB/ORACLE/`.

## 1.1 Control-plane files

| File | Role |
|------|------|
| `MISSION.md` | Highest-level purpose and final architecture |
| `control/mission.json` | Machine-readable mission contract |
| `control/current_state.json` | Compact current frontier state and blockers |
| `control/capability_graph.json` | Compute-leverage lineage and exact code transfer |
| `control/branch_registry.json` | Long-lived branch purpose and context routing |
| `CAPABILITY_GRAPH.md` | Generated human view of the capability graph |
| `templates/PHASE_LOCK_RECEIPT.json` | Agent reconstruction handshake |
| `templates/TASK_CONTRACT.json` | Task freeze before implementation |
| `templates/EXPERIMENT.json` | Experiment-level compute-leverage contract |
| `tools/phase_lock.py` | Generates task-scoped context and receipts |
| `tools/validate_control_plane.py` | Fails closed on incomplete or inconsistent control state |

The authority order is declared in `control/canon_manifest.json`. Mission language may
not inflate evidence. Historical reports may not redefine the mission. Branch-local
state may refine the current state only within its recorded claim ceiling.

## 2. Tracks

Experiments live inside one of eight thematic tracks, named `N_snake_case`
(single digit, narrative order). Track membership follows the lab's own taxonomy
in [MASTER_REPORT.md](../MASTER_REPORT.md) and is a **contiguous experiment-number
range** so it is never ambiguous:

| Track | Range | Theme |
|-------|-------|-------|
| `1_foundations` | 01–05 | reversible computing, Landauer basics |
| `2_substrate_expansion` | 06–13 | catalytic memory, inference substrate |
| `3_physics_complexity` | 14–24 | Bekenstein, factorization, NP, temporal |
| `4_holographic` | 25–33 | lattice/crypto, graphs, wormholes, MERA |
| `5_topological_proofs` | 34–41 | zeta/RH, halting oracles, ToE |
| `6_frontier_phases` | 42–48 | limits (event-horizon) -> proof-power (math) -> emergence built atom-up (atom, energy, chem, bio) -> final boss (consciousness) |
| `7_decoder` | 49–50 | the decoder theory (49) + physical substrate crossing (50 bm_cpu) |
| `8_external_frontiers` | 51+ | externally adjudicated cryptanalysis, theorem frontiers, reconstruction, reasoning, compression, and cross-domain transfer |

New experiments append contiguously inside `8_external_frontiers` while that track
remains the active frontier. Open `9_*` only when a genuinely new thematic track is
needed — never renumber 01–50 or reuse 51+ identities.

## 3. Experiment directories

- `NN_snake_case_name` — two-digit zero-padded number + lowercase snake-case name.
- The **global number (01–50) is permanent and load-bearing** (the README,
  MASTER_REPORT, and the audit ledger reference experiments by number, e.g.
  "Exp 16", "Exp 50 Phase 6"). Moving an experiment into a track changes its
  *path*, never its number. The same permanence applies to new experiments 51+.
- **Collision suffix:** when one number forks into sibling *experiments*, append a
  letter to the number: `NNx_name`. Resolved live collisions:
  - `25_lattice_holography` — base experiment (keeps the bare number).
  - `25b_wigners_friend` — distinct experiment (was `25_wigners_friend`).
  - `26a_hawking_quantum`, `26b_optical_3sat` — two experiments under 26.
  - `25a_lattice_holography` was **not** a separate experiment (one infinity
    script); it is folded into `25_lattice_holography/infinity/` (section 5).

## 4. Sub-experiments

One scheme only: **`NN_M_topic`** (underscore, no dots). Nested levels add another
index: `NN_M_K_topic`.

- `20_1_base_eigen_shor` → `20_1_base_eigen_shor`
- `35_1_hermitian_oracle` → `35_1_hermitian_oracle`
- `34/34_1_spectral_foundations` → `34_1_spectral_foundations`
- `20.11a` (nested) → `20_11_1_...`

Flat experiments that encoded order via file prefixes (`1_x.py`, `2_x.py`) keep
those filenames — the `NN_M` rule governs **sub-directories**, not files inside a
single flat experiment.

## 5. Infinity / "pushed" variants

A scaled/pushed variant always lives in an **`infinity/` subdirectory** of its
parent experiment (matching exp 05, 06). Never a bare `NNa_` sibling directory,
never a loose `1_infinity_*.py` at the experiment root.

- `05_multibit_compiler/infinity/1_infinity_compiler.py` — canonical shape (already correct).
- `04_thermodynamic_cpu/1_infinity_thermo.py` → `04_.../infinity/1_infinity_thermo.py`
  (the `infinity/` subdir is the rule; the `1_infinity_*` filename matches 05/06).

## 6. Shared primitives

Core code is defined **once** in `_lib/` and imported, never copy-pasted:

| Module | Provides |
|--------|----------|
| `_lib/paths.py` | `repo_root()`, `cat_cas_root()`, `exp_dir()`, `eigen_buddy_rust()` |
| `_lib/catalytic_engine.py` | `MemoryTracker`, `CatalyticTape`, `OutOfMemoryError` |
| `_lib/tree_eval.py` | `TreeEval` |
| `_lib/reversible_cpu.py` | `ReversibleCPU`, `IrreversibleCPU`, `calculate_landauer_energy` |
| `_lib/catalytic_tape.py` | `CatalyticTape`, `BennettHistoryTape` (reconciled superset) |

- Import via `from _lib... import` (a root `conftest.py` puts `_lib/` on `sys.path`).
- **Never** count `../` hops to reach another experiment or the repo root — use
  `_lib/paths.py`. Hardcoded absolute paths are an M-7 critic violation.
- Track-specific shared code may live under that track's own `shared/` directory
  only when it is not a lab-wide primitive. Promote it to `_lib/` only after
  demonstrated cross-track reuse.

## 7. Report / doc filenames

| File | Meaning |
|------|---------|
| `REPORT.md` | the one primary report per experiment (UPPERCASE) |
| `REPORT_<TOPIC>.md` | a focused deep-dive within an experiment |
| `ROADMAP.md` / `ROADMAP_<TOPIC>.md` | plan for that experiment |
| `VERIFICATION_REPORT.md` | audit/verification record |
| `MASTER_REPORT.md` | exists **only** at the lab root |

Retired genres: `PUSHED_REPORT.md` (fold into a "Pushed / Infinity" section of
`REPORT.md`), lowercase `report.md` (rename to `REPORT.md`), per-experiment
`MASTER_REPORT_EXP_NN.md` (becomes that experiment's `REPORT.md`).

Preparation-only frontier directories may use `ROADMAP.md` as their primary doc
until execution begins. Create `REPORT.md` when evidence exists; do not create an
empty report merely to satisfy layout symmetry.

## 8. Status vocabulary

Use the six canonical states defined in [CAT_CAS_OS.md](../CAT_CAS_OS.md) §11 —
`DONE`, `DONE-UNVERIFIED`, `COSMETIC`, `OPEN`, `DEFERRED`, `UNKNOWN`. The canonical
per-experiment audit truth is
[REPORTS/VIOLATIONS/ROADMAP_3_VERIFIED.md](REPORTS/VIOLATIONS/ROADMAP_3_VERIFIED.md).

Workflow states inside a track may refine execution progress, but they do not
replace the canonical experiment status.

## 9. Large data (>100 MB)

Kept on disk **in the owning experiment**, gitignored, and documented as a
requirement in that experiment's `REPORT.md` and in [STORAGE.md](STORAGE.md)
(source, size, regeneration command, why untracked). Never committed, never moved
out of the lab.

## 10. Placement note (formerly the frozen exception)

`50_phase_bm_cpu` has been placed at `7_decoder/50_phase_bm_cpu` (the physical
test of crossing the decoder wall on bare-metal CPU substrate). It is no longer
frozen at the CAT_CAS root. Experiment number 44 belongs solely to
`6_frontier_phases/44_phase_atom`; the transient number collision is resolved.

Track 8 begins at experiment 51 and is outward-facing by role. It does not replace
or subordinate Exp 50; external frontiers and the physical Small Wall run in
parallel and meet only through explicit transfer records.
