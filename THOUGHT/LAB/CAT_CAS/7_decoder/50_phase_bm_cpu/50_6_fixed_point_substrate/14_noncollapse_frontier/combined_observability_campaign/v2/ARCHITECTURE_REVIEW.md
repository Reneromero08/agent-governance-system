# Phase 6 V2 Architectural Review

This change is forward-only. It adds `holo_runtime_v2`, a V1 frozen-artifact
and recorded-output binding audit, and V2 calibration contracts.

Review invariants:

```text
V1 historical paths remain untouched
external_frontiers remains untouched
calibration_authorized=false by default
acquisition_authorized=false
restoration_authorized=false
target_coupling_authorized=false
small_wall_authorized=false
```

Real V2 calibration cannot start without a new authorization artifact bound to
the executor commit, executor binary SHA-256, source bundle, campaign plan,
session IDs, route cores, runtime parameters, output root, and authorizer.

V2 run objects use execution class
`AUTHORIZED_V2_SPECTRAL_CALIBRATION_NOT_ACQUISITION`. Validation and mock
outputs also set every scientific authorization field false. The calibration
analyzer rejects any evidence object that is labeled as acquisition or enables
restoration, target coupling, or the small-wall path.

No V2 hardware execution was performed. No V2 rerun is authorized.
