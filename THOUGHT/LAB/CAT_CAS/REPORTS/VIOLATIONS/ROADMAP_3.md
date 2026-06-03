# CAT_CAS Remediation Roadmap — Actionable Checklist
*Last verified: 2026-06-02 | Critic status: 0 violations | Commits: 7*

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
- [ ] **F-FAKE** — 8 files with text-only "statistics" — *Review each to determine if "std=0" is legitimate (exact invariant) or missing computation*
- [ ] **E-UNKNOWN** — 5 null model files that couldn't be classified by regex — *Manual inspection for implicit nulls*

---

## 🎯 QUICK-START PRIORITIZATION
*If you're overwhelmed, tackle in this order:*

1. 🔴 **Close G-8 and I-3** (Hardcoded paths) — quick wins for portability
2. 🔴 **Expand J-4** (master_report.md) — improves visibility for all other work
3. ❓ **Manual review queue**: E-UNKNOWN (5 files) + F-FAKE (8 files) — 13 files total to classify
4. ⏸️ **Plan H-7 migration** — large breaking change, schedule deliberately

---

> 💡 **Pro Tip**: Use your project tracker to create sub-tasks for each unchecked item. Tag with `#remediation`, `#phase-XX`, and priority (`P0`/`P1`/`P2`). Update status in this doc as you complete items to maintain a single source of truth.

*Document integrity preserved: All original details, file paths, status codes, and notes retained. Reorganized for actionability and trackability.*