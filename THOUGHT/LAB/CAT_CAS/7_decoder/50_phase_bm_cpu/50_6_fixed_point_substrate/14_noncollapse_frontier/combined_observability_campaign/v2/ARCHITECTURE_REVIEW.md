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
derived 672 windows:

```text
12 tones * (8 theta blocks * (3 amplitudes * 2 signs + 1 sender-off control))
= 672 windows/session
= 1,344 windows/route
= 2,688 windows/campaign
```

The plan binds the ordered windows, runtime parameters, logically separated
sender fields, and
predeclared spectral thresholds. Exact session manifests are members of the
source-bundle manifest. The analyzer can issue a campaign pass only after both
reboot partitions and both routes satisfy the frozen rules.

The committed four-session source bundle closes the campaign design, but it is
not directly executable one session at a time. The C hardware gate requires a
distinct singleton subset bundle and authorization whose complete session sets
both equal the current session. This prevents partial execution under the full
campaign bundle.

The current sender/receiver field projection is deterministic logical field
separation only. The full schedule serializes both mappings, so the sender gate
is neither hidden nor unreconstructible. This is not a valid blinded scramble
null and cannot support a scramble-null scientific claim.

```text
CALIBRATION_PLAN_V2.json SHA-256:
b4b9d1bd50eb1f246f109a1508f7ac68f42c0ba7e4462fe53442e678c4009c3c

SOURCE_BUNDLE_MANIFEST_V2.json SHA-256:
c165c753b70f8e5aff49bff31ad4c3c5db5c0f2d8715594171324b187df1342a
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
