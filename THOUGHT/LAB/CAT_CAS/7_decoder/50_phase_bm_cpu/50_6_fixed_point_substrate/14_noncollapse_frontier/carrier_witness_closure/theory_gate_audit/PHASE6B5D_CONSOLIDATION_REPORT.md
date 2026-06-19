# Phase 6B.5D Carrier Claim Consolidation

**Status:** `COMPLETE__CLAIM_FROZEN_PENDING_GATE_R`  
**Official strict carrier closure:** `PARTIAL` (unchanged)  
**Input:** committed, manifest-bound Phase 6B.5C packet  
**New physical acquisition:** none  
**Open-ended analysis authorized:** no

## Bindings

```text
Phase 6B.5C campaign = phase6b5_t48_d32b1bed_20260619
Phase 6B.5C campaign manifest = cbcd2a19d6dd3bc478244f77888aa87eb043003a7685caa17ff13fe4d47e6487
Phase 6B.5C analysis manifest = 93ccb5fb5d9cbc96c25c52797ea0dd0693810997a369e714cfe57109af35ff2b
Phase 6B.5D analysis manifest = 07aa4bbe683ebfd43dff15bdad68d87f6d417109ce9909448c97b8d0516a18cb
Workflow run = 27847845516
Workflow artifact digest = 57d5411928a50210ac297997d4b4f4bad4ec9294cab1022686bbeab3b13e20ad
```

The Phase 6B.5D workflow executed its regression suite, generated the bounded packet, verified every generated SHA-256, uploaded the packet, and committed the exact JSON outputs to the PR branch.

---

## 1. Scalar calibration cannot rescue the old gates

All twelve Phase 6B.5C selected charts are nonzero scalar identity maps:

```text
T(z) = alpha z
```

The old analyzer first computes:

```text
zhat(z) = z exp(-i arg(sum z)) / ||z||
```

For any nonzero complex scalar `alpha`:

```text
zhat(z / alpha) = zhat(z)
```

Therefore the following are exactly unchanged by applying the selected C0 calibration:

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

The old strict conjunction cannot be repaired by scalar route/session calibration. Its failures live in:

```text
residual structure
sparse 0.95 threshold geometry
gate semantics
```

—not in an unremoved scalar gain or phase.

This narrows the earlier theory audit: the old scorer assumed ideal codeword concentration, but the minimal chart discovered by Phase 6B.5C was already quotiented out by its feature normalization. The successful transfer-aware result comes from testing relational fit and covariance rather than demanding a zero-error sparse concentration floor.

---

## 2. Cross-session and cross-route generalization

The Phase 6B.5C cross-chart packet contains 132 source-chart → target-session tests:

| Transfer class | Records | Positive-margin fraction | Median residual | Median mode margin |
|---|---:|---:|---:|---:|
| Same route, different seed | 60 | median `1.000`, minimum `0.9792` | `0.3873` | `1.0490` |
| Different route, same seed | 12 | median `1.000`, minimum `0.9792` | `0.3740` | `1.0854` |
| Different route, different seed | 60 | median `1.000`, minimum `0.9792` | `0.3866` | `1.0685` |

Every transfer record remains above the predeclared `0.95` positive-margin criterion.

Conclusion:

```text
relational mode ordering generalizes across sessions and routes
```

Absolute residual scale remains session-sensitive, especially when seed 4 is the target. This does not identify a universal physical operator; it shows that the transported relational ordering is substantially more stable than the per-session scalar coordinate.

### Fitted scalar coordinates

Route `2:3`:

```text
|alpha| = 0.04713–0.05064
arg(alpha) = -1.3189 to -1.2812 rad
```

Normal route `4:5` sessions:

```text
|alpha| = 0.04911–0.05103
arg(alpha) = -1.3259 to -1.3035 rad
```

Route `4:5`, seed `4`:

```text
|alpha| = 0.025851
arg(alpha) = -1.221254 rad
```

---

## 3. Residual structure

The compact packet contains 576 held-out real/wrong residual records.

Overall:

```text
median normalized residual = 0.37417
median phase-aligned residual = 0.36976
median phase-removable component = 0.000932
95th-percentile phase-removable component = 0.01571
```

Global phase mismatch therefore explains almost none of the residual.

### Variance attribution over compact observables

| Factor | eta-squared |
|---|---:|
| Seed/session | `0.53628` |
| Theta index | `0.03487` |
| Mode | `0.00424` |
| Trial block | `0.00121` |
| Family (`real`/`wrong`) | `0.00092` |
| Route | `0.00034` |

The dominant compact-packet factor is **seed/session**. Route, mode, family, and elapsed trial block are minor by comparison.

Median residual by seed, pooling routes:

```text
seed 0 = 0.3335
seed 1 = 0.3458
seed 2 = 0.3578
seed 3 = 0.3674
seed 4 = 0.6080
seed 5 = 0.3632
```

Mode medians remain close (`0.3631–0.3870`), as do phase-index medians (`0.3551–0.3972`). Real and wrong families are almost identical (`0.37395` versus `0.37640`).

The result is not evidence that all complex residual vectors are unstructured noise. The compact packet retains residual magnitudes and relational alternatives, not the full complex residual vectors needed for bin-level residual Gram or cross-spectral decomposition. That raw-field extension is not authorized because the bounded consolidation already localizes the dominant issue sufficiently for the next control.

---

## 4. Seed 4 localization

Route `4:5`, seed `4` is refined from generic `CHART_FAILURE` to:

```text
SCALAR_GAIN_OUTLIER_WITH_RELATIONAL_INVARIANTS_PRESERVED
```

Evidence:

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
pseudo covariance = preserved in Phase 6B.5C
```

Seed 4 is not a monotonic late-run drift and is not repaired by global phase alignment. Its scalar carrier coordinate is approximately half the normal route amplitude and phase-shifted, while mode ordering, execution-over-declaration, phase relation, and pseudo permutation covariance survive.

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

The official historical strict conjunction remains `PARTIAL` because it is a separate, frozen gate.

---

## 6. Hard stop and next boundary

The bounded consolidation is complete. Further open-ended analysis of this campaign is not authorized.

Immediate boundary:

```text
Gate R — external L4B.5B0 human review and project-owner integration decision
```

Proposed next physical control:

```text
REVERSED_RANDOMIZED_TONE_ORDER
status = PREREGISTERED_NOT_AUTHORIZED
```

Its purpose is to separate tone identity from within-symbol path position while preserving routes, codewords, phase actions, wrong/pseudo controls, and raw provenance.
