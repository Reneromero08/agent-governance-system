# Phase 6 V2 Architectural Review

**Status:** `DRAFT_ENGINEERING_QUALIFICATION__FINAL_HEAD_CLOSURE_PENDING`  
**Authority:** `../../PHASE6_V2_ENGINEERING_QUALIFICATION_ADDENDUM_2026-06-22.md`  
**Scientific acquisition authorized:** false

This change is forward-only. It adds `holo_runtime_v2`, an honest V1 artifact-and-recorded-output binding audit, exact V2 calibration contracts, and a corrected spectral-analysis lane.

It does not alter the historical acquisition object, accept Gate R, enter Phase 6B.6, establish a predictive physical operator, authorize restoration, or execute the selected reversed/randomized tone-order control.

## Review invariants

```text
V1 historical paths remain untouched
external_frontiers remains untouched
calibration_authorized=false by default
acquisition_authorized=false
restoration_authorized=false
target_coupling_authorized=false
small_wall_authorized=false
```

Real V2 calibration cannot start without a new authorization artifact bound to the executor commit, executor binary SHA-256, campaign source commit, singleton source bundle, campaign plan, session ID, route cores, runtime parameters, output root, and authorizer.

## Current generated object

The executable plan contains four exact sessions: pre- and post-reboot repetitions for each of `v4s5` and `v2s3`. Each session has a mechanically derived 672 windows:

```text
12 tones * (8 theta blocks * (3 amplitudes * 2 signs + 1 sender-off control))
= 672 windows/session
= 1,344 windows/route
= 2,688 windows/campaign
```

The plan binds ordered windows, runtime parameters, logically separated sender fields, sender-off controls, and predeclared spectral thresholds. Exact session manifests are members of the source-bundle manifest. The analyzer can issue a campaign pass only after both reboot partitions and both routes satisfy the frozen rules.

Current generated digests after source repair `d68a4f8ac53068ac68b83aa6e0e404c94a0f3b22`:

```text
CALIBRATION_PLAN_V2.json SHA-256:
d30f7e55fad4cf7ae2cc853d4d15c9a484b02a9713ed6aa737e2869b5fde43c1

SOURCE_BUNDLE_MANIFEST_V2.json SHA-256:
e041718aca33a734e441d59469440b9b6ddf1c32a56608264251c191ef7f5c24
```

These digests are provisional engineering artifacts. Any additional source repair requires deterministic regeneration before qualification evidence is final.

## Execution boundary

The committed four-session source bundle closes the campaign design, but it is not directly executable one session at a time. The C hardware gate requires a distinct singleton subset bundle and authorization whose complete session sets both equal the current session. This prevents partial execution under the full campaign bundle.

V2 run objects use execution class:

```text
AUTHORIZED_V2_SPECTRAL_CALIBRATION_NOT_ACQUISITION
```

Validation and mock outputs also set every scientific authorization field false. The calibration analyzer rejects evidence objects labeled as acquisition or enabling restoration, target coupling, or the Small Wall path.

The V2 drive source is protected by a source-identity regression against the historical Slot2 primitive containing `Lifted verbatim from phase5_10_driven_lockin.c`.

## Logical sender-field boundary

The current sender/receiver field projection is deterministic logical field separation only. The full schedule serializes both mappings, so the sender gate is neither hidden nor unreconstructible.

This is not a blinded scramble null and cannot support a scramble-null scientific claim.

## Tone-order boundary

The V2 calibration schedule uses ascending tone order. It qualifies waveform, runtime, custody, and analyzer behavior.

It is not the reversed/randomized tone-order physical control selected by Phase 6B.5D. That control requires a separate versioned design, review decision, authorization artifact, source bundle, and evidence chain.

## Final-head qualification still required

The focused Windows suite at historical head `38e15e9b11869f955270b65678089ff9b3a077b1` reported 24 passed and 0 errors. That result does not qualify later source or documentation heads.

Before merge readiness, the exact final head must close:

1. strict command-line numeric parsing;
2. complete fail-closed authorization document parsing;
3. mechanical capture-quality threshold binding;
4. analyzer enforcement of capture coverage, empirical sample rate, Nyquist margin, and timestamp-gap thresholds;
5. exact plan and evidence schema validation;
6. rejection of extra directories and symlinks in immutable run roots;
7. same-byte parse and digest binding;
8. V2 GitHub strict, equivalence, analyzer, ASan, and UBSan lanes;
9. exact final-head Linux target qualification;
10. regenerated contracts, evidence inventory, logs, and authority documents.

No V2 hardware execution was performed. No authorization artifact was created. No V2 rerun is authorized.
