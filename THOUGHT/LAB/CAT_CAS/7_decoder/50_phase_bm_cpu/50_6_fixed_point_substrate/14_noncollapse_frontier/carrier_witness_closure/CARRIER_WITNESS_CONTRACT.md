# Phase 6B.5 Strict PDN Carrier-Witness Contract

**Contract ID:** `phase6b5_pdn_carrier_witness_v1`  
**Status:** `FROZEN_FOR_IMPLEMENTATION_REVIEW`  
**Primary route:** victim `4`, sender `5`  
**Comparator route:** victim `2`, sender `3`  
**Claim ceiling:** reconstructable sender-owned mode/phase transport through the tested PDN channel

---

## 1. Scientific question

Does a declared sender-owned mode/phase schedule produce a reproducible, route-scoped complex lock-in response on the Phenom PDN channel, such that every published score can be regenerated from retained raw timing samples and frozen source/configuration?

This is a carrier and provenance question.

It is not a question about:

```text
which fold coordinate is true
public orientation recovery
physical relational memory
physical restoration
target coupling
wall crossing
```

---

## 2. Predecessor evidence

The existing T300 campaign reported:

- two isolated userspace processes;
- absolute shared TSC origin;
- register/L1-only square-wave drive;
- victim ring-period lock-in;
- twelve non-harmonic tones;
- six seeds on routes `2:3` and `4:5`;
- silent and scramble controls;
- route `4:5` passing the frozen scored gates on all six seeds.

The imported repository artifacts begin at the per-symbol complex-vector/summary layer. The original receiver arrays `(t_tsc, ro_period)` were discarded after scoring each window. This contract moves the witness boundary back to those arrays.

---

## 3. Frozen route scope

### Required closure route

```text
victim core = 4
sender core = 5
route label = v4s5
```

Required seeds:

```text
0 1 2 3 4 5
```

Required conditions:

```text
matrix
silent
scramble-drive
```

The full matrix may retain `real`, `pseudo`, and `wrong` families inside each run as the existing deterministic schedule defines them.

### Comparator route

```text
victim core = 2
sender core = 3
route label = v2s3
```

The comparator is required for topology context, but `CLOSED_ROUTE_4_5` does not require route `2:3` to satisfy the primary scored gate. It does require comparator raw data to be structurally valid when reacquired as part of the same campaign.

---

## 4. Required raw observables

For every `(run, symbol, bin)` capture window retain:

```text
absolute sample TSC values: uint64
ring-period values: float64 ticks per inner iteration
absolute slot start TSC
absolute capture deadline TSC
shared run t0 TSC
capture sample count
victim core
sender core(s)
family
declared mode
actual physical mode
trial index
theta index
tone frequency
drive sign
phase fraction
control condition
temperature before and after window
current-frequency proxy before and after window
COFVID/P-state proxy before and after window
```

Missing thermal telemetry invalidates a new full acquisition unless the project owner explicitly approves a replacement sensor and the replacement is recorded before the run. `-999`, empty, or fabricated temperature values are invalid.

---

## 5. Deterministic configuration

Each run must preserve:

```text
campaign ID
run ID
UTC start/end
source Git commit
source-file SHA-256 list
compiler identity and flags
binary SHA-256
kernel and CPU identity
isolcpus/affinity configuration
TSC feature flags and measured TSC rate
P-state target and restoration log
k10temp or approved thermal source
victim/sender cores
seed
trials
nbin
f_lo/f_hi
resolved tone list
slot_s
gap_s
read_hz
temperature veto
control flags
complete deterministic symbol schedule
complete codebook
```

The schedule is evidence, not something to regenerate later from a remembered algorithm version. Regeneration may be checked against it, but the exact schedule used must be serialized before or during acquisition.

---

## 6. Raw bundle structure

Every run is one immutable directory:

```text
runs/<run_id>/
  run.json
  schedule.json
  windows.csv
  raw_samples.bin
  summary.csv
  analysis.json
  stdout.log
  stderr.log
  run_manifest.json
```

The campaign root contains:

```text
campaign.json
source_manifest.json
runs/
aggregate/
  aggregate.json
  closure_report.json
campaign_manifest.json
```

Large `raw_samples.bin` files may remain on the evidence host or approved durable storage rather than Git, but their exact paths, sizes, SHA-256 values, and storage media identifiers must be committed in the compact campaign manifest.

---

## 7. Reconstruction requirement

For every window, the validator must recompute the original Hann-windowed, mean-removed lock-in:

```text
mean = average(ro_period)
win_i = 0.5 * (1 - cos(2*pi*i/(n-1)))
dt_i = (t_tsc_i - slot_start_tsc) / tsc_hz
phase_i = 2*pi*f_ref*dt_i
I = 2 * sum((x_i-mean)*win_i*cos(phase_i)) / sum(win_i)
Q = 2 * sum((x_i-mean)*win_i*sin(phase_i)) / sum(win_i)
```

The off-bin floor reference is frozen as:

```text
f_floor = 1.37 * f_drive + 0.071
```

Recomputed window values must match the values in `windows.csv` within:

```text
absolute tolerance = 1e-9
relative tolerance = 5e-6
```

The validator must then regenerate `summary.csv`, rerun the existing matched-null analyzer, rerun the route/seed aggregate, and compare all retained metrics and gate booleans.

No hand-edited summary is admissible.

---

## 8. Structural validity gates

Each run must satisfy:

- manifest hashes verify;
- source and binary hashes exist;
- run ID is unique;
- route/seed/control matches campaign plan;
- schedule and window counts match;
- raw record count equals the sum of window sample counts;
- binary record offsets are contiguous and non-overlapping;
- each window has at least four samples;
- TSC samples are strictly increasing within each window;
- first sample occurs after the declared slot start;
- last sample does not exceed the deadline by more than two nominal read intervals;
- all values are finite;
- temperature remains below the frozen veto;
- P-state restoration is recorded after every invocation;
- receiver/sender affinity failures are absent;
- summary and analysis are regenerated successfully.

A run failing any structural gate is `INVALID`, not a negative scientific result.

---

## 9. Scientific gates

The scored gates remain the historical T300 gates for continuity:

```text
all_rows_restore
real_accuracy >= 0.60
real_vs_pseudo floor >= 0.95
pseudo_reject floor >= 0.95
wrong_actual_match >= 0.60
wrong_declared_match <= 0.20
phase_corr_true - phase_corr_null > 0.30
```

Additional closure gates:

```text
raw_reconstruction_pass = true
all_run_manifests_valid = true
all_required_runs_present = true
all_control_runs_structurally_valid = true
silent and scramble do not pass witness gates
route 4:5 passes all six seeds
```

The closure validator must report structural validity separately from scientific gate outcomes.

---

## 10. Allowed closure verdicts

### `CLOSED_ROUTE_4_5`

All required route `4:5` raw bundles, seeds, and controls are complete and reconstructable; the frozen scored gate passes 6/6 seeds; campaign provenance is valid. Comparator route may be partial scientifically but must be honestly reported.

### `CLOSED_MULTI_ROUTE`

At least two routes independently meet the complete raw and scored closure gates.

### `PARTIAL`

Raw reconstructability is complete for only part of the frozen campaign, or route `4:5` remains scientifically incomplete despite structurally valid raw evidence.

### `PENDING`

Acquisition or raw recovery has not completed.

### `INVALID`

Provenance, raw structure, thermal safety, source binding, or reconstruction failed.

---

## 11. Safety and stop conditions

Stop before full acquisition when:

- no valid thermal sensor is available;
- the temperature source returns sentinel/invalid values;
- current temperature is at or above veto;
- core affinity does not hold;
- TSC assumptions are false;
- P-state pin/restore cannot be verified;
- disk space is insufficient for raw bundles plus duplicate manifest generation;
- a raw smoke run cannot reproduce its own I/Q summary;
- source or binary hashes change after campaign freeze.

Do not bypass thermal refusal with `--no-temp`, a dummy value, or a raised veto.

---

## 12. Claim language after closure

Allowed:

> On the frozen Phenom route `4:5`, the declared sender-owned mode and relational phase were transported reproducibly through the measured PDN/ring-period channel, and the reported complex vectors and scored summaries were regenerated from retained raw TSC/ring-period samples under matched controls.

Forbidden:

```text
physical HoloGeometry exists
physical memory was restored
public fold orientation was recovered
target state traversed the carrier
Small Wall was crossed
```

---

## 13. Exit and next gate

After a valid closure verdict:

1. freeze the compact closure report and campaign manifest;
2. update the physical mapping support only at channel/reconstructability scope;
3. bind the external L4B.5B0 human design review;
4. separately decide whether observability acquisition is authorized.

Carrier closure itself does not authorize that acquisition.
