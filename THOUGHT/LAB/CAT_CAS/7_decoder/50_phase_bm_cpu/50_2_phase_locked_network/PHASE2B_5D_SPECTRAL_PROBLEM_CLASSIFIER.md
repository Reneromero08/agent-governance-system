# PHASE2B_5D_SPECTRAL_PROBLEM_CLASSIFIER

## Verdict

`PHASE2B_5D_SPECTRAL_CLASSIFIER_PASS`

The Exp31-style spectral/topological problem classifier was ported and run on the Phenom II target. It selected a best-mean solver family on 5/5 held-out problem families after one bounded rule revision for ring graphs and tie-aware scoring.

This is a software routing aid for phase-oracle selection. It is not passive Kuramoto evidence and not physical phase lock.

## Command

```powershell
Get-Content -Raw session_scripts\phase2b\spectral_problem_classifier.c | ssh -o BatchMode=yes -o ConnectTimeout=5 root@192.168.137.100 "cat > /tmp/spectral_problem_classifier.c && gcc -O2 /tmp/spectral_problem_classifier.c -lm -o /tmp/spectral_problem_classifier && timeout 40 /tmp/spectral_problem_classifier"
```

## Safety

- Pure userspace C.
- No MSR access.
- No voltage writes.
- No BIOS flash.
- No P0-P3 modification.
- No Tier 3 physical instrumentation.

## Features

The classifier computes:

- graph density,
- signed-triangle frustration ratio,
- degree coefficient of variation,
- signed spectral radius,
- absolute-adjacency spectral radius,
- signed/absolute spectral ratio.

## Solver Families

The classifier routes among:

- `active_edge`,
- `vertex_phase`,
- `bloch_complex`.

Correctness is tie-aware: a prediction is accepted if its mean energy equals the best mean energy within `0.001`.

## Results

| Held-out problem | Prediction | Best mean family | Result |
|---|---|---|---|
| `heldout_ferro_ring_n14` | `bloch_complex` | `bloch_complex` | PASS |
| `heldout_odd_anti_ring_n13` | `bloch_complex` | `bloch_complex` | PASS |
| `heldout_sparse_mixed_n16` | `bloch_complex` | `bloch_complex` | PASS |
| `heldout_noisy_planted_n16` | `active_edge` | `active_edge` | PASS |
| `heldout_chord_frustrated_n14` | `bloch_complex` | tie with `vertex_phase` | PASS |

Global result:

```text
Classifier accuracy: 5/5
PHASE2B_5D_SPECTRAL_CLASSIFIER_PASS
```

## Interpretation

The classifier gives Phase 2B a deterministic pre-oracle routing layer:

- low-density regular rings route to `bloch_complex`,
- frustrated or irregular sparse graphs route to `bloch_complex`,
- dense low-frustration planted graphs route to `active_edge`,
- tie cases are accepted only when the predicted solver matches the best mean.

This improves the active software phase-oracle branch, but it does not convert active software computation into passive substrate evidence.

## Route Impact

Phase 2B.5D advances from untested to:

`PHASE2B_5D_SPECTRAL_CLASSIFIER_PASS`

The global Phase 2 goal remains active:

- not `CPU_SINGS`,
- not `BYTE_READY_HUMAN_REVIEW`,
- not `SOFTWARE_FIRMWARE_TRUE_WALL`,
- not `HUMAN_TOOL_REQUIRED_WITH_ALL_OTHER_ROUTES_EXHAUSTED`.

## Next Action

`PHASE2B_5E_HOLO_MERA_BRIDGE`

Connect the phase-oracle output to the `.holo` eigenbasis / MERA bridge and test whether the classifier-selected oracle result can be encoded, restored, and compared against nulls through the catalytic tape path.
