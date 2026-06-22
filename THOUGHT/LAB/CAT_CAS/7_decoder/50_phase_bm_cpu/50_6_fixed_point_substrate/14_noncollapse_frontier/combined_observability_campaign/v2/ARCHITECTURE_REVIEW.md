# Phase 6 V2 Architectural Review

This change is forward-only. It adds `holo_runtime_v2`, an honest V1
artifact-and-recorded-output binding audit, and exact V2 calibration contracts.

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

The executable plan contains four exact sessions: pre- and post-reboot
repetitions for each of `v4s5` and `v2s3`. Each session has a mechanically
derived 588 windows:

```text
12 tones * 3 amplitudes * 8 theta states * 2 signs + 12 sender-off controls
= 588 windows/session
= 1,176 windows/route
= 2,352 windows/campaign
```

The plan binds the ordered windows, runtime parameters, scramble fields, and
predeclared spectral thresholds. Exact session manifests are members of the
source-bundle manifest. The analyzer can issue a campaign pass only after both
reboot partitions and both routes satisfy the frozen rules.

```text
CALIBRATION_PLAN_V2.json SHA-256:
2809e7341e54ac7c94b74501d9a1c773d791b75ca953cddde76ac7bb39eb2797

SOURCE_BUNDLE_MANIFEST_V2.json SHA-256:
70b92cf8b155537519af80b6f0ace677ba8ebee0612996b2f615747f168e0596
```

V2 run objects use execution class
`AUTHORIZED_V2_SPECTRAL_CALIBRATION_NOT_ACQUISITION`. Validation and mock
outputs also set every scientific authorization field false. The calibration
analyzer rejects any evidence object that is labeled as acquisition or enables
restoration, target coupling, or the small-wall path.

The V2 drive source is protected by a source-identity regression against the
historical Slot2 primitive containing `Lifted verbatim from
phase5_10_driven_lockin.c`.

No V2 hardware execution was performed. No authorization artifact was created.
No V2 rerun is authorized.
