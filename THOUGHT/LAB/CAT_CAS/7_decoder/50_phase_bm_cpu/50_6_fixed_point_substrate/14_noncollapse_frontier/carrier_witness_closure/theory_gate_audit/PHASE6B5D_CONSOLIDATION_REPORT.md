# Phase 6B.5D Carrier Claim Consolidation

**Status:** `COMPLETE__CLAIM_FROZEN_PENDING_GATE_R`  
**Official strict carrier closure:** `PARTIAL` (unchanged)  
**Input:** committed, manifest-bound Phase 6B.5C packet  
**New physical acquisition:** none  
**Open-ended analysis authorized:** no

## Deterministic bindings

```text
Phase 6B.5C campaign = phase6b5_t48_d32b1bed_20260619
Phase 6B.5C campaign manifest = cbcd2a19d6dd3bc478244f77888aa87eb043003a7685caa17ff13fe4d47e6487
Phase 6B.5C analysis manifest = 93ccb5fb5d9cbc96c25c52797ea0dd0693810997a369e714cfe57109af35ff2b
Phase 6B.5D deterministic manifest = d11bf9d41c1b9a9195d79d5ba1ab8b591f9c364b3f57435fded958d5a0861f31
Workflow run = 27848260395
Workflow artifact ID = 7758297990
Workflow artifact digest = defb733ba811e435068514fabb0e024f1a2492d3151b30491518e68197dd2e92
```

The workflow generated the packet twice from identical committed inputs and required a recursive byte-for-byte diff to pass. Tests, output-manifest verification, and the double-generation comparison all passed.

The stable binding is committed in `PHASE6B5D_DETERMINISTIC_MANIFEST.json`. Detailed JSON outputs remain reproducible through `run_carrier_consolidation.py` and the read-only workflow. The earlier wall-clock timestamp variant is not retained as evidence.

---

## 1. Scalar calibration cannot rescue the old gates

All twelve Phase 6B.5C selected charts are nonzero scalar identity maps:

```text
T(z) = alpha z
```

The historical analyzer first computes:

```text
zhat(z) = z exp(-i arg(sum z)) / ||z||
```

For any nonzero complex scalar `alpha`:

```text
zhat(z / alpha) = zhat(z)
```

Therefore scalar charting leaves these quantities exactly unchanged:

```text
fvec
rho
mhat
centroid prediction
real accuracy
real-mode floor
real-vs-pseudo floor
pseudo-reject floor
pseudo-declared match
wrong-actual match
wrong-declared match
differential phase delta
```

The old strict conjunction cannot be repaired by route/session scalar calibration. Its failures instead occupy:

```text
residual structure
sparse 0.95 threshold geometry
gate semantics
```

The successful transfer-aware result and frozen partial gate test different properties.

---

## 2. Cross-session and cross-route generalization

The Phase 6B.5C packet contains 132 source-chart to target-session tests:

| Transfer class | Records | Positive-margin fraction | Median residual | Median mode margin |
|---|---:|---:|---:|---:|
| Same route, different seed | 60 | median `1.000`, minimum `0.9792` | `0.3873` | `1.0490` |
| Different route, same seed | 12 | median `1.000`, minimum `0.9792` | `0.3740` | `1.0854` |
| Different route, different seed | 60 | median `1.000`, minimum `0.9792` | `0.3866` | `1.0685` |

Every transfer record remains above the frozen `0.95` positive-margin criterion.

```text
relational mode ordering generalizes across sessions and routes
```

This does not identify a universal physical operator. Absolute residual scale remains session-sensitive even while relational ordering generalizes.

### Fitted scalar coordinates

```text
route 2:3 |alpha| = 0.04713–0.05064
route 2:3 arg(alpha) = -1.3189 to -1.2812 rad

normal route 4:5 |alpha| = 0.04911–0.05103
normal route 4:5 arg(alpha) = -1.3259 to -1.3035 rad

route 4:5 seed 4 |alpha| = 0.025851
route 4:5 seed 4 arg(alpha) = -1.221254 rad
```

---

## 3. Residual structure

The compact packet contains 576 held-out real/wrong residual records.

```text
median normalized residual = 0.37417
median phase-aligned residual = 0.36976
median phase-removable component = 0.000932
95th-percentile phase-removable component = 0.01571
```

Global phase mismatch explains almost none of the residual.

| Factor | eta-squared |
|---|---:|
| Seed/session | `0.53628` |
| Theta index | `0.03487` |
| Mode | `0.00424` |
| Trial block | `0.00121` |
| Family (`real`/`wrong`) | `0.00092` |
| Route | `0.00034` |

The dominant compact-packet factor is **seed/session**. Route, mode, family, and elapsed trial block are minor by comparison.

```text
seed 0 median residual = 0.3335
seed 1 median residual = 0.3458
seed 2 median residual = 0.3578
seed 3 median residual = 0.3674
seed 4 median residual = 0.6080
seed 5 median residual = 0.3632
```

The result does not prove that full complex residual vectors are unstructured noise. A bin-level residual Gram or cross-spectrum would require another raw-field pass, which is not authorized because this bounded consolidation already localizes the dominant issue sufficiently for the next control.

---

## 4. Seed 4 localization

Route `4:5`, seed `4` is refined from generic `CHART_FAILURE` to:

```text
SCALAR_GAIN_OUTLIER_WITH_RELATIONAL_INVARIANTS_PRESERVED
```

```text
|alpha| = 0.025851
peer route median |alpha| = 0.049883
alpha magnitude z-score = -30.57
alpha phase z-score = +10.73
held-out residual median = 0.77674
peer route residual median = 0.31857
residual excess ratio = 2.438
phase-removable residual median = 0.00195
residual-vs-trial rank = 0.0161
residual-vs-symbol-index rank = 0.0104
real positive margin = 0.9583
wrong actual-over-declared = 1.000
phase MAE = 0.09135 rad
pseudo covariance = preserved
```

Seed 4 is neither monotonic late-run drift nor a missed global-phase alignment. Its scalar carrier coordinate is approximately half normal route amplitude and phase-shifted, while mode ordering, execution-over-declaration, phase relation, and permutation covariance survive.

The correct interpretation is a session-level carrier-coordinate excursion, not relational carrier collapse.

---

## 5. Frozen carrier claim

Supported:

```text
On the retained T48 campaign and tested routes, sender-owned mode, phase, and
exact pseudo-permutation relations survive held-out evaluation under
calibration-only minimal scalar complex charts.

Wrong-family receiver geometry follows physically executed mode rather than
false declared metadata.

Relational mode ordering generalizes across seeds and routes under cross-session
scalar charts.

Silent carrier-off and scramble unshared-schedule controls remain null.

Route 4:5 seed 4 preserves relational invariants despite a large session-level
scalar gain/phase excursion and degraded absolute fit.
```

Not supported:

```text
scalar calibration rescues the historical normalized strict gates
a complete physical route operator has been identified
all complex residual vectors are unstructured noise
tone order has a causal physical effect
physical HoloGeometry
physical restoration
target coupling
orientation recovery
Small Wall crossing
```

The official historical strict conjunction remains `PARTIAL` because it is a separate frozen gate.

---

## 6. Hard stop and next boundary

Further open-ended analysis of this campaign is not authorized.

```text
Immediate boundary = Gate R external L4B.5B0 human review
Project-owner integration decision = required
```

Proposed next physical control:

```text
REVERSED_RANDOMIZED_TONE_ORDER
status = PREREGISTERED_NOT_AUTHORIZED
```

Its purpose is to separate tone identity from within-symbol path position while preserving routes, codewords, phase actions, wrong/pseudo controls, and raw provenance.
