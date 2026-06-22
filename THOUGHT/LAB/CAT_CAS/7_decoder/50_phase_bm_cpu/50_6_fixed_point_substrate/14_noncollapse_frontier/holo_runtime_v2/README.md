# Phase 6 V2 Executor

This is a forward-only executor. The historical V1 runtime remains in
`../holo_runtime` and is not modified by V2.

V2 implements:

- an eight-state waveform whose full period is the requested fundamental;
- `pi/4` theta steps and a four-step code-sign offset;
- the verbatim historical Slot2/Exp 5.10 drive primitive, guarded by a
  source-identity regression test;
- distinct receiver and sender codeword/theta fields for deterministic logical
  sender-field separation;
- exact runtime, executor binary, campaign plan, source bundle, session
  manifest, session ID, and route/core binding to a V2 authorization;
- a frequency-settling gate before the first capture origin;
- total run-directory manifest closure;
- strict C-to-Python waveform equivalence fixtures.

Real execution requires a separate
`CAT_CAS_PHASE6_V2_SPECTRAL_CALIBRATION_AUTHORIZATION_V1` artifact. V2 output
is always engineering calibration evidence, never scientific acquisition
evidence. These fields remain false:

Each hardware invocation also requires a singleton source bundle and a
singleton authorization containing exactly the current session. A full
campaign bundle cannot authorize subset execution.

```text
acquisition_authorized=false
restoration_authorized=false
target_coupling_authorized=false
small_wall_authorized=false
```

No hardware calibration or acquisition is authorized by this source tree.
The sender mapping is serialized in the full schedule and is reconstructible.
The receiver projection is not a blinded scramble null and makes no hidden-gate
or unreconstructibility claim.
