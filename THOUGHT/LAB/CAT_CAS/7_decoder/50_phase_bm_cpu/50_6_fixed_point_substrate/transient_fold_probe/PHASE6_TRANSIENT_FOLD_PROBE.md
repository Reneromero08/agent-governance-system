# Phase 6 Transient Fold Probe

**Verdict:** `TRANSIENT_PUBLIC_FAIL_CHANCE__NO_FOLD_ODD_FUNCTIONAL_FOUND`

## Question

Does the public transient of `f(x)` carry a fold-odd orientation functional that static/global Phase 6 sensors missed?

## Result

| candidate | n | verdict | auc | null95 | random_fold_auc | random_fold_null95 | delta |
|---|---:|---|---:|---:|---:|---:|---:|
| transient_public | 8 | `FAIL_CHANCE` | 0.429 | 0.574 | 0.519 | 0.575 | 0 |
| transient_public | 10 | `FAIL_CHANCE` | 0.433 | 0.556 | 0.505 | 0.561 | 0 |
| transient_public | 12 | `FAIL_CHANCE` | 0.511 | 0.586 | 0.440 | 0.562 | 0 |
| transient_smuggle_control | 8 | `FAIL_SMUGGLE` | 1.000 | 0.537 | 1.000 | 0.541 | 1 |
| transient_smuggle_control | 10 | `FAIL_SMUGGLE` | 1.000 | 0.568 | 1.000 | 0.547 | 1 |

## Interpretation

The public transient features remain fold-even under the hardened random-private-fold gate. The hidden-orientation control is caught as a smuggle, so the instrument is live. This closes the specific `REPORT_SESSION_LATTICE_CLIMB.md` open crack about local/transient invariants for the current feature family.

This does not change the formal dihedral lower-bound status. It only says this concrete public transient route did not recover the orientation bit.
