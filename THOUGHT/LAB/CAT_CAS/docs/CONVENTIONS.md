# CAT_CAS Conventions

Single source of truth for how this lab is named and laid out. If a directory or
file disagrees with this document, the directory is wrong. Linked from
[PRIMER.md](../PRIMER.md) and [README.md](../README.md).

---

## 1. Directory layout

```
CAT_CAS/
  _lib/                  shared primitives (one copy each; see section 6)
  docs/                  navigation + conventions; holds REPORTS/ (audit ledger)
  N_track_name/          thematic tracks (section 2) containing experiments
  workspace/             shared experiment fixtures (exp 02/03)
  AGENTS.md CAT_CAS_OS.md MANIFESTO.md MASTER_REPORT.md PRIMER.md README.md
```

The leading-underscore folders (`_lib`) sort above the numbered tracks and signal
"infrastructure, not an experiment." The visualizer was promoted out to its own
sibling lab at `THOUGHT/LAB/ORACLE/`.

## 2. Tracks

Experiments live inside one of seven thematic tracks, named `N_snake_case`
(single digit, narrative order). Track membership follows the lab's own taxonomy
in [MASTER_REPORT.md](../MASTER_REPORT.md) and is a **contiguous experiment-number
range** so it is never ambiguous:

| Track | Range | Theme |
|-------|-------|-------|
| `1_foundations` | 01–05 | reversible computing, Landauer basics |
| `2_substrate_expansion` | 06–13 | catalytic memory, inference substrate |
| `3_physics_complexity` | 14–24 | Bekenstein, factorization, NP, temporal |
| `4_holographic` | 25–33 | lattice/crypto, graphs, wormholes, MERA |
| `5_topological_proofs` | 34–42 | zeta/RH, halting oracles, ToE, black holes |
| `6_frontier_phases` | 43–49 | consciousness, ssh, math, bio, atom, energy, chem |
| `7_decoder` | 50 | the decoder synthesis |

New experiments append to the highest track or open `8_*` — never renumber 01–50.

## 3. Experiment directories

- `NN_snake_case_name` — two-digit zero-padded number + lowercase snake-case name.
- The **global number (01–50) is permanent and load-bearing** (the README,
  MASTER_REPORT, and the audit ledger reference experiments by number, e.g.
  "Exp 16", "Exp 44 Phase 6"). Moving an experiment into a track changes its
  *path*, never its number.
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

## 8. Status vocabulary

Use the six canonical states defined in [CAT_CAS_OS.md](../CAT_CAS_OS.md) §11 —
`DONE`, `DONE-UNVERIFIED`, `COSMETIC`, `OPEN`, `DEFERRED`, `UNKNOWN`. The canonical
per-experiment audit truth is
[REPORTS/VIOLATIONS/ROADMAP_3_VERIFIED.md](REPORTS/VIOLATIONS/ROADMAP_3_VERIFIED.md).

## 9. Large data (>100 MB)

Kept on disk **in the owning experiment**, gitignored, and documented as a
requirement in that experiment's `REPORT.md` and in [STORAGE.md](STORAGE.md)
(source, size, regeneration command, why untracked). Never committed, never moved
out of the lab.

## 10. The frozen exception

`44_phase_ssh_linux/` is active work. While frozen it stays at the CAT_CAS root
and is exempt from every convention above. It moves into `6_frontier_phases/` only
when the owner declares it done.
