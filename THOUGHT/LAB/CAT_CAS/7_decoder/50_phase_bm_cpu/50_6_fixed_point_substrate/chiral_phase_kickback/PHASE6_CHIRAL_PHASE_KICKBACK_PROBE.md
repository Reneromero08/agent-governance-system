# Phase 6 Chiral Phase-Kickback Probe

Verdict: `CHIRAL_PREP_PUBLIC_NO_CROSSING__HIDDEN_PREP_GATE_LIVE`

## Question

Can a chiral pre-projection tape preparation expose a fold-odd carrier before the public cosine boundary is read?

## Gate

The probe reuses `fold_audit/stage3/hardened_gate.py`: orientation AUC, random-private-fold AUC, and exact fold-invariance delta. Public candidates read only `k`, `b`, `N`, and `n`. The hidden control deliberately binds the phase-walk direction to `d` to verify the carrier/gate lights up when an orientation lane exists.

## Results

| candidate | n | verdict | orient auc/null95 | random-fold auc/null95 | delta |
|---|---:|---|---:|---:|---:|
| chiral_phase_kickback_PUBLIC | 8 | FAIL_CHANCE | 0.492/0.538 | 0.562/0.567 | 0 |
| dual_lane_even_cancel_PUBLIC | 8 | FAIL_CHANCE | 0.504/0.582 | 0.398/0.559 | 0 |
| chiral_shuffle_null_PUBLIC | 8 | FAIL_CHANCE | 0.477/0.572 | 0.489/0.560 | 0 |
| hidden_chiral_prep_CONTROL | 8 | FAIL_SMUGGLE | 1.000/0.555 | 1.000/0.563 | 48.3 |
| chiral_phase_kickback_PUBLIC | 10 | FAIL_CHANCE | 0.551/0.559 | 0.564/0.572 | 0 |
| dual_lane_even_cancel_PUBLIC | 10 | FAIL_CHANCE | 0.399/0.557 | 0.501/0.572 | 0 |
| chiral_shuffle_null_PUBLIC | 10 | FAIL_CHANCE | 0.474/0.594 | 0.519/0.564 | 0 |
| hidden_chiral_prep_CONTROL | 10 | FAIL_SMUGGLE | 1.000/0.572 | 1.000/0.551 | 41.4 |
| chiral_phase_kickback_PUBLIC | 12 | FAIL_CHANCE | 0.447/0.559 | 0.433/0.538 | 0 |
| dual_lane_even_cancel_PUBLIC | 12 | FAIL_CHANCE | 0.527/0.555 | 0.451/0.579 | 0 |
| chiral_shuffle_null_PUBLIC | 12 | FAIL_CHANCE | 0.504/0.565 | 0.545/0.572 | 0 |
| hidden_chiral_prep_CONTROL | 12 | FAIL_SMUGGLE | 1.000/0.592 | 1.000/0.553 | 48.2 |

## Interpretation

The public chiral preparation, dual-lane even cancellation, and public shuffled schedule did not produce a no-smuggle crossing in this local model. The hidden pre-projection control is caught at every size, which means the carrier features and gate are live: if the missing orientation lane is bound before projection, the instrument detects it.

The physical PDN version was not launched because the Phenom is currently running the long 300-trial matrix job on `/root/slot2_pdn/run_t300.sh`. The clean next step is a PDN sideband run after that job exits, using the same logic: same public cosine stream, opposite chiral preparation lanes, even cancellation, and victim lock-in phase readout.

## Artifacts

- `chiral_phase_kickback_probe.py`
- `results/chiral_phase_kickback_result.json`
- `rust_baremetal/chiral_pdn_native.rs`
- `rust_baremetal/PHASE6_CHIRAL_PDN_NATIVE_RUST.md`
- `rust_baremetal/results/chiral_pdn_native_result.json`
