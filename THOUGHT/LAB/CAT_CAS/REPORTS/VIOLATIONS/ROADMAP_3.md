# CAT_CAS Remediation Roadmap — Actionable Checklist
*Last verified: 2026-06-02 | Critic status: 0 violations | Commits: 6 | Uncommitted: 5 files*

---

## 📋 Status Legend
| Symbol | Meaning |
|--------|---------|
| ✅ `DONE` | Fixed in git AND verified by audit |
| ⚠️ `DONE-UNVERIFIED` | Fixed in git, agent claims correct, not re-checked |
| 🎨 `COSMETIC` | Critic passes but fix is text labels, not real computation |
| 🔴 `OPEN` | Not fixed — requires action |
| ⏸️ `DEFERRED` | Known, intentionally not fixing now |
| ❓ `UNKNOWN` | Cannot classify by automated tool — needs manual review |

---

## 🚨 SECTION A: BLOCKER BUGS (4 items) — *Priority: CRITICAL*
*All 4 fixed in commit `97a96075` — verify before proceeding*

- [ ] **A-1** — Feistel swap produces `a^b` in both halves  
  📄 `15_hdd_native_inference/experiment.py` | Status: ⚠️ DONE-UNVERIFIED

- [ ] **A-2** — F16 weight loading uses uint16 not float16  
  📄 `16_catalytic_27b_inference/experiment.py` | Status: ⚠️ DONE-UNVERIFIED

- [ ] **A-3** — Undefined `k95_phase` variable  
  📄 `16_catalytic_27b_inference/_test_phase.py` | Status: ⚠️ DONE-UNVERIFIED

- [ ] **A-4** — 6 AttributeErrors on missing attrs  
  📄 `30_boundary_stress/1_memory_collision.py` | Status: ⚠️ DONE-UNVERIFIED

---

## 🔥 SECTION B: CRITICAL — NULL RESULTS & FALSE CLAIMS (6 items)

- [x] **B-1** — 47.4 palindrome = spin (null result)  
  Status: ✅ DONE | Notes: Refactored to baryon collision. Session 3 verified: mean=0.5228 vs random=0.5002. Underpowered at N=26 but signal real.

- [x] **B-2** — 47.5 Higgs mechanism (false claim)  
  Status: ✅ DONE | Notes: Corrected to mpmath normalization cost. 512-bit spike confirmed.

- [ ] **B-3** — PUSHED_REPORT inflated KV claims  
  Status: ⚠️ DONE-UNVERIFIED | Notes: Agent changed 3076.9x to 12.5x — verify calculation.

- [ ] **B-4** — Exp 13 cross-talk formula broken  
  Status: ⚠️ DONE-UNVERIFIED | Notes: Extraction formula produces 135K+ error — validate fix.

- [ ] **B-5** — Exp 13 snapshot drift wrong baseline  
  Status: ⚠️ DONE-UNVERIFIED | Notes: Now compares against same tape's initial state — confirm logic.

- [ ] **B-6** — Exp 7 non-deterministic measurement  
  Status: ⚠️ DONE-UNVERIFIED | Notes: `np.random.rand()` restored (quantum measurement IS probabilistic) — verify intent matches implementation.

---

## ⚙️ SECTION C: CRITIC RULES M-1 THROUGH M-4 (10 items)

### M-1: Hardcoded Invariants
- [x] **C-1** — M-1 hardcoded invariant  
  📄 `46/validation_mandate4_null_models.py` | Status: ✅ DONE | Notes: Fixed by Session 2 agent (dynamic computation).

### M-2: Ceremonial Tapes (genuine XOR-modifying required)
- [x] **C-2** — M-2 ceremonial tape  
  📄 `46/validation_mandate4_null_models.py` | Status: ✅ DONE | Notes: Genuine XOR tape with `was_modified` flag.

- [ ] **C-3** — M-2 ceremonial tape  
  📄 `46/validation_mandate5_conservation.py` | Status: ⚠️ DONE-UNVERIFIED

- [ ] **C-4** — M-2 ceremonial tape  
  📄 `46/validation_real_connectome.py` | Status: ⚠️ DONE-UNVERIFIED

- [ ] **C-5** — M-2 ceremonial tape  
  📄 `46/validation_real_morphogenesis.py` | Status: ⚠️ DONE-UNVERIFIED

### M-3: Arbitrary Thresholds
- [x] **C-6** — M-3 arbitrary threshold 0.55  
  📄 `47_4_lhc_overflow_exploit.py` | Status: ✅ DONE | Notes: Resolved by B-1 refactor.

### M-4: NxN SAT Claims (annotation-only fixes)
- [x] **C-7** — M-4 NxN SAT  
  📄 `40_sub_1_temporal_sat.py` | Status: 🎨 COSMETIC | Notes: Annotated as "proven impossible" — no code change.

- [x] **C-8** — M-4 NxN SAT  
  📄 `40_sub_2_floquet_swarm.py` | Status: 🎨 COSMETIC | Notes: Same — annotation only.

- [x] **C-9** — M-4 NxN SAT  
  📄 `40_sub_4_sat_swarm.py` | Status: 🎨 COSMETIC | Notes: Same — annotation only.

- [x] **C-10** — M-4 NxN SAT  
  📄 `45_5_p_vs_np_catalytic.py` | Status: 🎨 COSMETIC | Notes: Same — annotation only. Phase 45.5 report already admits "UNIVERSAL FAILURE."

---

## 🎭 SECTION D: PHASE 47 CEREMONIAL TAPE CRISIS (6 items)
*All 6 fixed. Shared `47_phase_atom/catalytic_tape.py` created with genuine XOR-modifying tape.*

- [x] **D-1** — `47_1_nucleus_memory_knot.py` | Status: ✅ DONE
- [x] **D-2** — `47_2_electron_edge_states.py` | Status: ✅ DONE
- [x] **D-3** — `47_3_pauli_exclusion.py` | Status: ✅ DONE
- [x] **D-4** — `47_4_lhc_overflow_exploit.py` | Status: ✅ DONE
- [x] **D-5** — `47_5_higgs_mechanism.py` | Status: ✅ DONE
- [x] **D-6** — `47_6_quark_confinement.py` | Status: ✅ DONE

---

## 🧪 SECTION E: MISSING NULL MODELS (23 files)
*Audited with `classify_ef.py` on 2026-06-02. Results are ground truth from actual files.*

### ✅ REAL (genuine computation) — 10 files — *No action needed*
`33/20_tuneable`, `41/41d_transfer_clock`, `45.2`, `45.3`, `45.3spatial`, `45.5_time_crystal`, `46.2`, `47.1`, `47.2`, `47.6`

### 🏷️ TEXT_ONLY (labels on existing controls) — 8 files — *Add real null models or document rationale*
- [ ] `04` — Add null model or justify control-as-null
- [ ] `05/compiler` — Add null model or justify control-as-null
- [ ] `46.3` — Add null model or justify control-as-null
- [ ] `46.5` — Add null model or justify control-as-null
- [ ] `46.6` — Add null model or justify control-as-null
- [ ] `47.3` — Add null model or justify control-as-null
- [ ] `47.5` — Add null model or justify control-as-null
- [ ] `05/reversible` (from 04) — Add null model or justify control-as-null

### ❓ UNKNOWN (can't classify by regex) — 5 files — *Manual inspection required*
- [ ] `05/reversible_cpu` — Inspect for implicit null (e.g., U(1) as null for SU(2))
- [ ] `40/sub3_quantum` — Inspect for implicit null
- [ ] `45.4` — Inspect for implicit null
- [ ] `45.6_gribov` — Inspect for implicit null
- [ ] `45.6_mass_gap` — Inspect for implicit null

> 📝 Assessment: Phase 45 and 47 have REAL null models. Phase 46 mostly has TEXT_ONLY (existing control groups that got labels). The 5 UNKNOWN files need manual inspection.

---

## 📊 SECTION F: MISSING STATISTICS (50 files)
*Audited with `classify_ef.py` on 2026-06-02.*

### ✅ REAL (np.std, t-test, CI, bootstrap) — 35 files — *No action needed*
Most of Phase 19, 23, 24, 33, 34(telescope), 35, 40(most), 42(most), 45.1, 47(most)

### 🎭 FAKE (text "std=0" without computation) — 10 files — *Add real stats or justify exact invariants*
> ⚠️ Note: Some report exact topological invariants (winding numbers, Chern numbers) where "std=0" is physically correct. Needs manual review to distinguish legitimate exactness from missing computation.

- [ ] `07_quantum_simulator/1_infinity_quantum.py` — Add statistical computation or document exact invariant
- [ ] `11_grail_calorimeter/1_infinity_calorimeter.py` — Add statistical computation or document exact invariant
- [ ] `34_zeta_eigenbasis/18_googol_zero_telescope.py` — Add statistical computation or document exact invariant
- [ ] `34_zeta_eigenbasis/19_topological_zeta_winding.py` — Add statistical computation or document exact invariant
- [ ] `34_zeta_eigenbasis/20_transcendent_winding_oracle.py` — Add statistical computation or document exact invariant
- [ ] `34_zeta_eigenbasis/21_absolute_infinity_collapse.py` — Add statistical computation or document exact invariant
- [ ] `36_bekenstein_godel/36_bekenstein_godel_singularity_catalytic.py` — Add statistical computation or document exact invariant
- [ ] `40_sub/40_sub_3_quantum/40_sub_3_quantum.py` — Add statistical computation or document exact invariant
- [ ] `42/BLACK_HOLES/exp_20_amps_firewall/20_amps_firewall.py` — Add statistical computation or document exact invariant

### ❓ UNKNOWN (Rust or can't classify) — 3 files — *Manual inspection*
- [ ] `08` — Inspect for statistical implementation
- [ ] `40/sub13_rust` — Inspect for statistical implementation
- [ ] `42/15_unification` — Inspect for statistical implementation

### 📦 UNCOMMITTED (modified but not in git) — 2 files — *Commit after verification*
- [ ] `42.10` — Verify stats implementation, then commit
- [ ] `42.22` — Verify stats implementation, then commit

---

## 🗂️ SECTION G: HARDCODED OUTPUT PATHS (17 items)
*All fixed except 2. Uses `os.path.dirname(os.path.abspath(__file__))` now.*

### Phase 46 Oracles — G-1 to G-7
- [ ] G-1 to G-7 — Verify path fixes in Phase 46 oracle files | Status: ⚠️ DONE-UNVERIFIED

### Phase 47 Experiments — G-9 to G-14
- [ ] G-9 to G-14 — Verify path fixes in Phase 47 experiment files | Status: ⚠️ DONE-UNVERIFIED

### 🔴 OPEN Items
- [ ] **G-8** — `46/validation_real_morphogenesis.py` input CSV path still hardcoded | Status: 🔴 OPEN

---

## 🐛 SECTION H: CODEBASE BUGS (10 items)

- [ ] **H-1** — lm_head overwrite in 16  
  Status: ⚠️ DONE-UNVERIFIED

- [ ] **H-2** — ground_truth side-effect in 11  
  Status: ⚠️ DONE-UNVERIFIED

- [ ] **H-3** — 41b = 41a exact duplicate  
  Status: ⚠️ DONE-UNVERIFIED | Notes: Annotated as duplicate

- [ ] **H-4** — Float equality in 04 infinity thermo  
  Status: ⚠️ DONE-UNVERIFIED

- [ ] **H-5** — Float equality in 07 experiment  
  Status: ⚠️ DONE-UNVERIFIED

- [ ] **H-6** — torch.svd → torch.linalg.svd (2 files)  
  Status: ⚠️ DONE-UNVERIFIED

- [ ] **H-7** — np.random.RandomState → Generator (25 files)  
  Status: ⏸️ DEFERRED | Notes: Breaking change, requires re-running all experiments

- [ ] **H-8** — mmap try/finally in 14 hdd_scale  
  Status: ⚠️ DONE-UNVERIFIED

- [ ] **H-9** — Dead code in 33 _infinity_engine  
  Status: ⚠️ DONE-UNVERIFIED

- [ ] **H-10** — Dead rope() in 33 _tape_engine  
  Status: ⚠️ DONE-UNVERIFIED

---

## 💳 SECTION I: TECHNICAL DEBT (3 items)

- [ ] **I-1** — 46 bare `except:` clauses  
  Status: ⚠️ DONE-UNVERIFIED | Notes: 35 replaced with specific types

- [ ] **I-2** — torch.load without weights_only (2 files)  
  Status: ⚠️ DONE-UNVERIFIED

- [ ] **I-3** — 6 hardcoded Windows paths  
  Status: 🔴 OPEN

---

## 📚 SECTION J: DOCUMENTATION (6 items)

- [ ] **J-1** — Spelling "Haydeng-Preskill"  
  Status: ⚠️ DONE-UNVERIFIED

- [ ] **J-2** — Spelling "Assesment"  
  Status: ⚠️ DONE-UNVERIFIED

- [ ] **J-3** — Missing files in README  
  Status: ⚠️ DONE-UNVERIFIED | Notes: Agent says false positive — verify manually

- [ ] **J-4** — master_report.md covers 9 of 41+ experiments  
  Status: 🔴 OPEN | Notes: Session 3 updated it but still incomplete — expand coverage

- [ ] **J-5** — Unused imports (5 files)  
  Status: ⚠️ DONE-UNVERIFIED

- [ ] **J-6** — Duplicate reversible_cpu.py  
  Status: ⚠️ DONE-UNVERIFIED

---

## 🔄 SECTION K: PROCESS IMPROVEMENTS (4 items)

- [ ] **K-1** — Zero-violation pre-commit enforcement  
  Status: 🔴 OPEN | Notes: pre-commit hook exists but not tested — add CI test

- [x] **K-2** — BennettHistoryTape fail-safe (bytes_written check)  
  Status: ✅ DONE

- [x] **K-3** — Isomorphism audit per phase  
  Status: ✅ DONE | Notes: Phases 42, 45, 46, 47 audited (see Isomorphism Audits section below)

- [ ] **K-4** — M-8 critic check for ceremonial tapes  
  Status: ⚠️ DONE-UNVERIFIED

---

## 📌 SECTION L: PHASE 46 LIMITATIONS (4 items)
*All documented. No code changes needed — track for future work.*

- [x] **L-1** — IPR signal degrades at large L | Status: ⚠️ DONE-UNVERIFIED | Notes: Documented
- [x] **L-2** — Prion doesn't propagate in static lattice | Status: ⚠️ DONE-UNVERIFIED | Notes: Documented
- [x] **L-3** — Connectome validation (synthetic vs real) | Status: ⚠️ DONE-UNVERIFIED | Notes: C. elegans validation added
- [x] **L-4** — Bott Index fails at Exceptional Points | Status: ⚠️ DONE-UNVERIFIED | Notes: 1D slice IPR workaround documented

---

## 🔍 ISOMORPHISM AUDITS SUMMARY (Session 3 — 33 experiments)
*Full reports in `PHASE_*_ISOMORPHISM_AUDIT.md`*

### Phase 42 (15 experiments) — 9 VALID, 5 WEAK, 1 not audited
| Exp | Claim | Verdict | Action |
|-----|-------|---------|--------|
| 42.1 | Hawking evaporation | ✅ VALID | None |
| 42.2 | Wormhole exploit | ✅ VALID | None |
| 42.3 | Quantum tunneling | ✅ VALID | None |
| 42.4 | Page curve | ✅ VALID | None |
| 42.5 | Gravitational waves | ⚠️ WEAK | Exponent shift ≠ propagating wave — document limitation |
| 42.6 | Holographic principle | ✅ VALID | None |
| 42.7 | Einstein-Rosen bridge | ✅ VALID | None |
| 42.8 | White holes | ✅ VALID | None |
| 42.9 | Quantum superposition | ⚠️ WEAK | Race conditions ≠ quantum states — document limitation |
| 42.10 | Information paradox | ✅ VALID | None |
| 42.11 | Photon sphere | ⚠️ WEAK | Riemann zeros ≠ orbital resonances — document limitation |
| 42.21 | Bekenstein-Hawking | ⚠️ WEAK | Shannon entropy ≠ black hole entropy — document limitation |
| 42.24 | Dark matter | ✅ VALID | None |
| 42.27 | Arrow of time | ✅ VALID | None |
| 42.15 | QM-GR unification | ⚠️ WEAK | Pearson r ≠ unification — document limitation |

### Phase 45 (6 experiments) — 4 VALID, 2 WEAK
| Exp | Claim | Verdict | Action |
|-----|-------|---------|--------|
| 45.1 | Collatz conjecture | ✅ VALID | None |
| 45.2 | Navier-Stokes smoothness | ⚠️ WEAK | Specific Weyl model ≠ general PDE — document scope |
| 45.3 | Erdos discrepancy | ✅ VALID | None |
| 45.4 | Riemann hypothesis | ✅ VALID | None |
| 45.5 | P vs NP | ✅ VALID procedure, ⚠️ LOOSE claim | Dual answer ≠ proof — clarify claim language |
| 45.6 | Yang-Mills mass gap | ✅ VALID | None |

### Phase 46 (6 experiments) — 3 VALID, 1 WEAK, 1 PARTIAL, 1 WEAKENED
| Exp | Claim | Verdict | Action |
|-----|-------|---------|--------|
| 46.1 | Protein foldability | ✅ VALID | None |
| 46.2 | Folding pathway | ⚠️ WEAK | Parameter sweep ≠ dynamical pathway — document limitation |
| 46.3 | Prion contagion | ⚠️ PARTIAL | Detects impurity, no propagation shown — add propagation test or clarify scope |
| 46.4 | Genetic code | ✅ VALID structure, ⚠️ WEAKENED claim | Mitochondrial codes are superior — revise claim or add comparative analysis |
| 46.5 | Neural binding | ✅ VALID | None |
| 46.6 | Morphogenesis | ✅ VALID | None |

### Phase 47 (6 experiments) — 4 VALID, 2 WEAK
| Exp | Claim | Verdict | Action |
|-----|-------|---------|--------|
| 47.1 | GC cycle = strong force | ✅ VALID | None |
| 47.2 | Edge states = orbitals | ✅ VALID | None |
| 47.3 | TRS breaking = Pauli | ✅ VALID | None |
| 47.4 | LHC overflow = particles | ⚠️ WEAK | Shattering real, particle mapping forced — document as metaphor or refine mapping |
| 47.5 | Higgs = normalization | ⚠️ WEAK | Latency real, Higgs mapping metaphorical — document as analogy or refine claim |
| 47.6 | Confinement = string tension | ✅ VALID | None |

> 📊 Overall: 20/33 valid (61%), 13/33 weak/forced (39%)

---

## 🔴 OPEN ITEMS SUMMARY — *Action Required*
*Consolidated list of items that are actually not done*

### Path/Portability Issues
- [ ] **G-8** — `validation_real_morphogenesis.py` input CSV path still hardcoded
- [ ] **I-3** — 6 hardcoded Windows paths

### Deferred/Large-Scale Changes
- [ ] **H-7** — np.random.RandomState → Generator migration (25 files) — *DEFERRED: breaking change, requires re-running all experiments*

### Documentation Gaps
- [ ] **J-4** — master_report.md still incomplete (covers ~9 of 41+ experiments) — expand to cover all phases

### Statistical/Null Model Ambiguities — *Manual Review Queue*
- [ ] **F-FAKE** — 10 files with text-only "statistics" — *Review each to determine if "std=0" is legitimate (exact invariant) or missing computation*
- [ ] **E-UNKNOWN** — 5 null model files that couldn't be classified by regex — *Manual inspection for implicit nulls*

### Uncommitted Work — *Verify and Commit*
- [ ] `41d_transfer_clock` — modified but not committed
- [ ] `42.10` — modified but not committed
- [ ] `42.2` — modified but not committed
- [ ] `42.3` — modified but not committed
- [ ] `42.22` — modified but not committed

---

## 📦 UNCOMMITTED FILES TRACKING
*Files modified locally but not yet in git — verify before committing*

| File | Phase | Action |
|------|-------|--------|
| `41d_transfer_clock` | 41 | Verify changes, run tests, commit |
| `42.10` | 42 | Verify changes, run tests, commit |
| `42.2` | 42 | Verify changes, run tests, commit |
| `42.3` | 42 | Verify changes, run tests, commit |
| `42.22` | 42 | Verify changes, run tests, commit |

---

## 🚫 OUT OF SCOPE / NOT TRACKED IN THIS ROADMAP
*For awareness — not part of current remediation effort*

- ❌ **Phase 48 (Energy Extraction)** — Roadmap exists at `ROADMAP_48_ENERGY_EXTRACTION.md`, not yet started
- ❌ **Phases 01-33, 35-40** — Original experiments, not part of remediation effort
- ❌ **AGENT_BULLSHIT_LOG.md** — Historical record of agent failures, kept for accountability
- ❌ **USER_DIRECTIVES_LOG.md** — What the user told agents to do (and what they did wrong)

---

## 🎯 QUICK-START PRIORITIZATION
*If you're overwhelmed, tackle in this order:*

1. 🔴 **Verify A-1 through A-4** (Blocker bugs) — ensure critical fixes actually work
2. 🔴 **Close G-8 and I-3** (Hardcoded paths) — quick wins for portability
3. 🔴 **Expand J-4** (master_report.md) — improves visibility for all other work
4. ❓ **Manual review queue**: E-UNKNOWN (5 files) + F-FAKE (10 files) — 15 files total to classify
5. 📦 **Commit uncommitted files** — 5 files pending git integration
6. ⏸️ **Plan H-7 migration** — large breaking change, schedule deliberately

---

> 💡 **Pro Tip**: Use your project tracker to create sub-tasks for each unchecked item. Tag with `#remediation`, `#phase-XX`, and priority (`P0`/`P1`/`P2`). Update status in this doc as you complete items to maintain a single source of truth.

*Document integrity preserved: All original details, file paths, status codes, and notes retained. Reorganized for actionability and trackability.*