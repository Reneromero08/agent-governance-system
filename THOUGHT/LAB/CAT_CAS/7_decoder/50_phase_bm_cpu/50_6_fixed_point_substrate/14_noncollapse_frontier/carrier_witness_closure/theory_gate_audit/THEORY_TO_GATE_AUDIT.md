# Phase 6B.5B Theory-to-Gate Audit

**Status:** `COMPLETE__NO_NEW_PHYSICAL_ACQUISITION`  
**Scope:** historical T300 gate, raw-reconstructable T48 evidence, CAT_CAS non-collapse doctrine  
**Official carrier-closure status:** unchanged at `PARTIAL`  
**Immediate conclusion:** the old pass conjunction is not a single physical invariant

---

## 1. Why this audit exists

The prior replication-discrepancy analysis was statistically careful but began from the historical scorer as the fixed object. CAT_CAS requires the inverse direction:

```text
mechanism
→ physical carrier
→ observable relation
→ extraction
→ boundary projection
→ gate
```

The question is not only whether the old gate reproduced. The question is whether each gate is a faithful witness for the physical claim assigned to it.

The governing architecture is:

```text
relational geometry
→ carrier-mediated evolution
→ preserved ordered path
→ declared closure
→ explicit CollapseBoundary
→ invariant extraction
```

A reduction that occurs before its declared boundary may still be a useful local diagnostic, but it cannot silently become the ontology of the experiment.

---

## 2. Strongest coherent physical object

For one sender mode `m`, relational phase `theta`, route `r`, session `s`, and tone bin `b`, the receiver-side complex response is more faithfully represented as:

```text
z[r,s,m,theta,b]
    = H[r,s,b] * (T[r,s] c[m])[b] * exp(i theta)
      + drift[r,s,b,t]
      + noise[r,s,b,t]
```

where:

- `c[m]` is the declared sender codeword;
- `theta` is the sender-owned relational phase;
- `H[r,s,b]` is frequency-dependent route/session response;
- `T[r,s]` is the unidentified physical transfer between sender codebook coordinates and receiver coordinates;
- `z` is the measured I/Q field assembled from ordered tone windows.

The old analyzer effectively assumes that after one global phase removal and L2 normalization:

```text
T[r,s] approximately equals scalar * identity
```

because it compares the received vector directly against the ideal sender codewords. That assumption has not been physically identified. The repository's own physical mapping correctly classifies the relation basis / PDN transfer operator as proposed and unsupported.

Therefore failure of direct ideal-codeword concentration is not equivalent to failure of carrier transport.

---

## 3. The old gates are different scientific objects

The historical scorer contains nine booleans. The frozen closure contract names seven. They should not be spoken of as one homogeneous invariant family.

### 3.1 Protocol-integrity gate

```text
all_rows_restore
```

What it actually witnesses:

```text
local symbol-metadata tape was reversibly returned
```

What it does not witness:

```text
physical carrier restoration
physical substrate restoration
path closure
holonomy
```

Disposition: retain as protocol hygiene; do not count it as physical carrier evidence.

### 3.2 Mode-transport gates

```text
real_accuracy
real_mode_floor
wrong_actual_match
wrong_declared_match
```

Intended mechanism:

```text
receiver response follows the physically executed mode rather than declared metadata
```

The `wrong` family is the strongest no-smuggle component because the declared label is intentionally false while the physical drive uses the actual mode. Strong actual-mode recovery with low declared-mode recovery is direct evidence that the result follows execution rather than metadata.

Disposition: elevate the actual-versus-declared contrast as a primary carrier witness. Report global and per-mode behavior separately.

### 3.3 Phase-transport gate

```text
phase_corr_true - phase_corr_null
```

Intended mechanism:

```text
ordered differential sender phase survives into receiver phase relations
```

This is the most theory-aligned old gate because it tests a relation between consecutive states rather than a static winner label. It is also robust across the T48 matrix sessions.

Disposition: retain as primary phase-carrier evidence, while later replacing the shuffled-only null with richer ordered-path controls.

### 3.4 Shared-schedule specificity gate

```text
pseudo_reject_floor
```

The pseudo family is not carrier-off. It still contains:

- physical drive;
- sender-owned phase;
- signed codeword energy;
- matched timing;
- the same substrate route.

What is broken is the shared canonical bin relation through an unshared permutation.

Disposition: interpret pseudo rejection as shared-schedule / canonical-basis specificity, not carrier existence.

### 3.5 Canonical-basis fidelity gate

```text
real_vs_pseudo_floor
```

This gate combines:

```text
real rows above a global rho threshold
+
pseudo rows below the same threshold
```

It does not merely ask whether pseudo rows are rejected. `pseudo_reject_floor` already asks that. It additionally requires every real row in the weakest sparse mode bucket to remain concentrated in the ideal sender codebook basis.

Example: `v4s5_matrix_seed0` has:

```text
real accuracy = 1.0
pseudo reject floor = 1.0
pseudo false accepts in floor bucket = 0
real-vs-pseudo floor = 0.7
```

The failure comes from three correctly classified real `mini` rows falling below the global concentration threshold—not from pseudo overlap and not from lost mode identity.

Disposition: rename semantically as `ideal_codebook_chart_fidelity`; do not use it as the sole veto on the more general channel-transport claim until a route/session transfer operator has been identified or factored out.

### 3.6 Metadata-leakage sanity gates

```text
pseudo_declared_match
wrong_declared_match
```

These test whether arbitrary or deliberately false declared labels control the result.

Disposition: retain as no-smuggle diagnostics. They are not interchangeable with physical mode transport or codebook fidelity.

---

## 4. Concrete pass-namespace defect

The closure contract freezes seven scientific gates:

```text
all_rows_restore
real_accuracy
real_vs_pseudo_floor
pseudo_reject_floor
wrong_actual_match
wrong_declared_match
phase_delta
```

The analyzer emits nine gates, adding:

```text
real_mode_floor
pseudo_declared_match
```

The validator intentionally computes `scientific_pass` from the seven contract keys while retaining the analyzer's nine-gate verdict string.

Consequently `v4s5_matrix_seed3` is recorded as:

```text
contract seven-gate pass = true
analyzer nine-gate pass = false
scientific_pass = true
verdict = PHASE4B_PDN_PARTIAL
```

This is not a physical contradiction. It is a reporting namespace defect.

Required correction:

```text
contract_seven_gate_pass
analyzer_nine_gate_pass
```

must always be separate fields. Neither historical artifact nor official verdict is rewritten by this audit.

Derived route counts:

| Route | Contract seven-gate | Analyzer nine-gate |
|---|---:|---:|
| `4:5` | 1/6 | 0/6 |
| `2:3` | 2/6 | 2/6 |

---

## 5. What the T48 campaign says when gates are separated by role

Derived diagnostics, not replacement verdicts:

| Layer | Route `4:5` | Route `2:3` |
|---|---:|---:|
| Phase transport | 6/6 | 6/6 |
| Core carrier transport | 5/6 | 6/6 |
| Canonical sender-basis fidelity | 1/6 | 2/6 |
| Seven-gate closure | 1/6 | 2/6 |
| Nine-gate analyzer pass | 0/6 | 2/6 |

`core carrier transport` here means the conjunction of:

```text
mode follows executed drive
phase follows declared relational phase
pseudo schedule is rejected
```

It excludes local metadata restoration and the direct ideal-codebook concentration requirement.

This re-read does not close Phase 6B.5. It identifies where the Wall currently lies:

```text
carrier transport is strong
canonical receiver chart is unstable or unidentified
```

Seed `4` on route `4:5` remains a real anomaly because it also degrades mode transport. It is not explained away by this decomposition.

---

## 6. Finite-sample geometry of the 0.95 floor

The T48 per-mode combined real/pseudo denominators are approximately `9–16`.

For a threshold of `>= 0.95`:

```text
n < 20  →  zero errors allowed
n = 20  →  one error allowed
```

Therefore the same nominal percentage gate has different discrete semantics at T48 and T300.

At T48 a single false rejection can force failure in many buckets. At T300 the same underlying error rate can pass. This is not merely lower statistical power; it is a different acceptance geometry.

Required conclusion:

```text
T48 is not an equivalent strict replication of the T300 percentage gate.
```

The accurate statement is:

```text
the historical uncalibrated codebook-fidelity closure was not reproduced
under the T48 discrete gate geometry
```

not:

```text
the physical carrier hypothesis was reproduced only once
```

---

## 7. Premature boundaries in the old analyzer

### 7.1 Static-vector collapse

Each twelve-bin symbol is physically acquired as twelve sequential `0.5 s` tone windows. The analyzer treats the assembled result as one simultaneous exchangeable vector.

This collapses:

```text
ordered path
frequency progression
elapsed-time drift
route response
```

into one point before testing path order or route transfer.

Tone and time are confounded because tone order is fixed. The raw evidence supports a future order-reversal control, but the existing campaign cannot disentangle them causally.

### 7.2 Global-phase gauge

The feature vector uses:

```text
g = arg(sum(z))
zr = z * exp(-i g)
```

This gauge can become ill-conditioned for balanced codewords whose ideal signed sum is near zero. The rho magnitude is mostly gauge-insensitive, but centroid coordinates and mode classification can rotate unpredictably when the sum is small or noise-dominated.

### 7.3 L2 normalization

L2 normalization removes route/session amplitude and may help classification, but it also discards potentially meaningful cross-bin amplitude ratios and carrier-strength information before the boundary is fully characterized.

### 7.4 Winner projection

Nearest-centroid mode prediction converts a complex field into one label. This is permissible as a declared projection, but it cannot substitute for testing whether the received field transforms equivariantly under the sender operation.

---

## 8. Old controls reinterpreted in their strongest coherent form

| Family/control | Physical meaning | Correct evidentiary role |
|---|---|---|
| `real` | declared and executed mode agree | positive transported geometry |
| `wrong` | declared label differs; physical drive uses actual mode | execution-over-metadata / no-smuggle witness |
| `pseudo` | physical drive and phase remain; bin relation is privately permuted | canonical-basis/shared-schedule control |
| `scramble` | receiver lacks the executed relation | shared-relational-key control |
| `silent` | sender does not drive | carrier-off null |

The old conjunction erases these distinctions. The corrected audit retains all of them without relaxing any official gate.

---

## 9. Theory-aligned next analysis

No new physical acquisition should occur until the existing raw campaign is analyzed under an explicitly route-aware, order-aware geometry contract.

The next derived analysis must test:

1. **Transfer-calibrated equivariance**  
   Estimate a route/session transfer chart from preamble and even real rows only. Test odd real and wrong rows held out. Do not use pseudo or test labels to select the chart.

2. **Execution-over-declaration likelihood**  
   Compare the received field's likelihood under actual versus declared physical mode for wrong rows without reducing first to a winner label.

3. **Phase equivariance**  
   Test whether phase shifts induce the predicted group action within mode and route, including phase closure around declared loops.

4. **Permutation covariance**  
   Compare canonical and exact pseudo-permuted sender models after the same transfer chart. Ask whether pseudo is off-manifold or merely transformed within an allowed equivalence class.

5. **Route conjugacy**  
   Ask whether route-specific transfer charts are related by a stable map while preserving relational invariants.

6. **Path-order sensitivity**  
   Analyze sequential windows as an ordered path. Quantify what is lost by vector assembly and specify future reversed/randomized tone-order controls.

7. **Seed-4 chart-versus-carrier diagnosis**  
   Determine whether seed 4 loses phase, executable mode geometry, transfer stability, or only the old coordinate chart.

All results remain derived diagnostics until separately preregistered physical controls are run.

---

## 10. Audit verdict

```text
OFFICIAL STRICT CARRIER CLOSURE: PARTIAL
RAW PROVENANCE: COMPLETE
PHASE CARRIER: STRONGLY SUPPORTED
EXECUTED MODE TRANSPORT: STRONGLY SUPPORTED EXCEPT ROUTE 4:5 SEED 4
IDEAL SENDER-BASIS FIDELITY: NOT REPRODUCIBLE AT T48
IDENTIFIED PHYSICAL RELATION BASIS: NOT ESTABLISHED
```

Primary audit finding:

```text
The old strict gate conflates carrier transport with fidelity to an
unidentified receiver coordinate chart.
```

Secondary findings:

```text
- seven-gate and nine-gate pass namespaces were mixed;
- the 0.95 floor becomes a zero-error gate at T48;
- RvP duplicates pseudo rejection while adding hidden real-concentration failure;
- ordered physical acquisition is collapsed into static vector classification;
- wrong-family actual-over-declared recovery is underweighted;
- local byte restoration is incorrectly adjacent to physical evidence.
```

This audit does not lower thresholds, replace the official verdict, or claim physical HoloGeometry. It identifies the fastest no-smuggle analysis needed to test the unique mechanism before more machine time is spent.
