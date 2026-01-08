# Living Formula Changelog

All notable changes to the Living Formula research.

---

## [v3.7.24] - 2026-01-08

### Answered
- **Q1: Why grad_S?** - grad_S measures potential surprise. R = E/grad_S implements Free Energy Principle.
- **Q9: Free Energy Principle** - Confirmed R ~ 1/F with 97.7% free energy reduction via R-gating.

### Key Findings
- E = amount of truth (must be measured against reality)
- R-gating reduces free energy by 97.7%
- R-gating is 99.7% more efficient (Least Action)
- Echo chambers have low E despite low grad_S, so R correctly penalizes them

### Tests Added
- `open_questions/q1/q1_adversarial_test.py` - attack vectors on grad_S
- `open_questions/q1/q1_essence_is_truth_test.py` - E definition validation
- `open_questions/q6/q6_free_energy_test.py` - R vs F correlation

### Structure
- Created `experiments/formula/open_questions/` with q1/, q2/, q4/, q6/ subdirectories
- Organized tests by question number

---

## [v3.7.23] - 2026-01-08

### Answered
- **Q1-Q5** initial answers (later refined in v3.7.24)
- Quantum Darwinism validation

### Key Findings
- R_single = 0.5 at full decoherence (gate CLOSED)
- R_joint = 18.1 at full decoherence (gate OPEN)
- 36x context improvement ratio
- Formula validated across 7 domains

### Tests Added
- `passed/quantum_darwinism_test.py`
- `passed/quantum_darwinism_test_v2.py`

---

## [v3.7.22] - 2026-01-07

### Established
- Formula is GATE not COMPASS
- Gate mechanism provides +24% improvement
- grad_S is critical component
- Time/history is scaffolding, not signal

### Tests Added
- `passed/opus_ablation_test.py`
- `passed/elo_gate_test.py`
- `passed/navigation_trap_test.py`
- `passed/navigation_hardening_test.py`

---

## Open Questions Status

| Q# | Question | Status | R-Score |
|----|----------|--------|---------|
| 1 | Why grad_S? | ANSWERED | 1800 |
| 2 | Falsification criteria | ANSWERED | 1750 |
| 3 | Why generalize? | ANSWERED | 1720 |
| 4 | Novel predictions | ANSWERED | 1700 |
| 5 | Agreement vs truth | ANSWERED | 1680 |
| 6 | IIT connection | PARTIAL | 1650 |
| 7 | Multi-scale composition | OPEN | 1620 |
| 8 | Topology classification | OPEN | 1600 |
| 9 | Free Energy Principle | ANSWERED | 1580 |
| 10 | Alignment detection | OPEN | 1560 |
| ... | ... | ... | ... |

**6/30 questions answered** (20%)
