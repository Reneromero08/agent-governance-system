# Exp 50 L4A Class B W_B Carrier Report

**Status:** L4A_CLASS_B_WB_CARRIER_PASS. Claim L2.
**Commit:** Pending.

---

## Executive Verdict

W_B workload and lock-in measurement layer run on Phenom II. Dual-sender alu_burst
at 200 Hz detected by receiver lock-in. 3 modes: normal, label-swap, carrier-off.
.holo records written with PhaseRelation data. Label-swap reveals Q_diff sign does
NOT flip -- indicates core asymmetry, not fold-odd residue. Carrier-off null.

---

## Run Matrix

| Mode | branch_plus (core 4) | branch_minus (core 5) | I+ | I- | Q+ | Q- | q_diff |
|---|---|---|---|---|---|---|---|
| normal | a=125 | Na=131 | -0.059 | 0.137 | -0.482 | -0.858 | +0.376 |
| label_swap | Na=131 | a=125 | 0.021 | 0.105 | -0.136 | -0.355 | +0.219 |
| carrier_off | 0 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

---

## Key Finding

Q_diff sign does not flip under label-swap (+0.376 vs +0.219, both positive).
This indicates the Q difference is CORE-DEPENDENT (PDN coupling asymmetry between
cores 4 and 5), not VALUE-DEPENDENT. A genuine fold-odd residue would flip sign
when branch values are swapped between cores.

---

## Interpretation

- Carrier excitation works: lock-in detects PDN activity.
- Measurement path is live: .holo records capture PhaseRelation data.
- Core asymmetry dominates: Q_diff driven by core identity, not branch values.
- No fold-odd residue detected at this validation level.
- This does NOT close all substrate routes. It only says Class B PDN/common-mode
  shows core asymmetry rather than value-driven fold-odd residue.
