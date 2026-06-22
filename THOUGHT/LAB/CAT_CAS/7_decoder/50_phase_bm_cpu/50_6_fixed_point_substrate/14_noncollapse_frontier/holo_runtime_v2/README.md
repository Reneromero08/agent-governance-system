# Phase 6 V2 Executor

This is a forward-only executor. The historical V1 runtime remains in
`../holo_runtime` and is not modified by V2.

V2 implements:

- an eight-state waveform whose full period is the requested fundamental;
- `pi/4` theta steps and a four-step code-sign offset;
- the verbatim historical Slot2/Exp 5.10 drive primitive, guarded by a
  source-identity regression test;
- distinct receiver and sender codeword/theta fields for physical scramble;
- exact runtime, executor binary, campaign plan, source bundle, session
  manifest, session ID, and route/core binding to a V2 authorization;
- a frequency-settling gate before the first capture origin;
- total run-directory manifest closure;
- strict C-to-Python waveform equivalence fixtures.

Real execution requires a separate
`CAT_CAS_PHASE6_V2_SPECTRAL_CALIBRATION_AUTHORIZATION_V1` artifact. V2 output
is always engineering calibration evidence, never scientific acquisition
evidence. These fields remain false:

```text
acquisition_authorized=false
restoration_authorized=false
target_coupling_authorized=false
small_wall_authorized=false
```

No hardware calibration or acquisition is authorized by this source tree.
