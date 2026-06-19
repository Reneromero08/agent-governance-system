# Phase 6B.5C Transfer-Aware Carrier Geometry Analysis Contract

**Status:** `FROZEN_FOR_RAW_ONLY_IMPLEMENTATION`  
**Physical acquisition authorized:** false  
**Input:** immutable T48 campaign only  
**Official carrier-closure status:** remains `PARTIAL`

## Scientific question

Does the measured complex PDN field preserve declared sender transformations after allowing a route/session-specific receiver coordinate chart?

The object under test is:

```text
sender codeword + phase + ordered schedule
→ route/session transfer
→ receiver complex field
```

It is not a winner label.

## Frozen partition

Calibration uses only:

```text
four preamble rows
even-trial real rows
```

Evaluation uses only:

```text
odd-trial real rows
wrong rows
pseudo rows
```

Evaluation labels, controls, and result summaries may not influence chart estimation. Seed 4 remains included and separately reported.

## Receiver chart ladder

Admit increasing complexity only:

```text
C0 = scalar complex gain
C1 = diagonal complex gain over bins
C2 = low-rank complex linear transfer
C3 = regularized full complex linear transfer
```

A more complex chart is admitted only after the simpler chart fails held-out equivariance and calibration rank supports the added degrees of freedom.

The chart is a coordinate relation, not a claim of complete physical operator identification.

## Held-out tests

### Real-mode equivariance

Compare held-out received fields against transferred codeword orbits before reducing to a class label. Report complex residuals, phase-aligned residuals, subspace angles, and actual-mode margin over alternatives.

### Execution over declaration

For every wrong row, compare fit to the physically executed mode against fit to the declared decoy mode. This is the primary metadata-independence witness.

### Phase equivariance

Within each mode and route, test the declared phase action. Report circular residual, pairwise phase-difference error, loop-closure error, and shuffled-schedule control.

### Pseudo permutation covariance

For every pseudo row, compare the exact executed permuted codeword model against canonical and unrelated-permutation models. This asks whether the carrier persists while the basis relation changes.

### Silent and scramble controls

The relational tests must collapse under carrier-off and unshared-schedule controls. No chart may be fitted using control rows.

## Route relation

Estimate charts independently for routes `2:3` and `4:5`, then test stable correspondence using subspace angles, Gram matrices, mode ordering, phase action, and held-out transfer residuals. Raw vectors need not match directly across routes.

## Ordered path

The twelve tone windows are a sequential physical path. Report elapsed position, local phase/magnitude drift, prefix/suffix stability, and path-composition residuals. Because only one tone order was acquired, causal order claims remain blocked and future reversed/randomized-order controls must be specified.

## Seed 4

For route `4:5`, seed `4`, test phase equivariance, execution-over-declaration margin, chart conditioning, held-out transfer residual, bin-wise gain stability, peer-session subspace rotation, and ordered-window drift.

Allowed classifications:

```text
CARRIER_FAILURE
CHART_FAILURE
TRANSFER_REGIME_SHIFT
MIXED_FAILURE
UNRESOLVED
```

## Required derived controls

```text
shuffled phase schedule
alternative mode assignment
matched-weight sign control
alternative bin permutation
cross-seed chart transfer
cross-route chart transfer
silent
scramble
```

Generation must be deterministic and recorded.

## Required outputs

Write under `theory_gate_audit/results/`:

```text
gate_layer_reconciliation.json
chart_calibration.json
heldout_equivariance.json
execution_relation.json
phase_equivariance.json
pseudo_permutation_covariance.json
route_conjugacy.json
ordered_path_analysis.json
seed4_transfer_report.json
analysis_manifest.json
```

Each output binds the campaign manifest, raw hashes, source commit, analysis source hash, data partition, chart complexity, parameters, and control seeds.

## Decision outcomes

Choose one primary result:

```text
TRANSFER_EQUIVARIANCE_SUPPORTED
PHASE_ONLY_TRANSPORT_SUPPORTED
CANONICAL_BASIS_ONLY_SUPPORTED
ROUTE_SESSION_CHART_UNSTABLE
NO_RELATIONAL_TRANSPORT_BEYOND_CONTROLS
INCONCLUSIVE
```

This analysis does not establish physical restoration, complete relation-basis identification, target coupling, orientation, or a Small Wall crossing.

## Acquisition decision

Only after this analysis may the project choose among an independent T48 session, tone-order controls, explicit route calibration, a versioned higher-N campaign, or a narrower claim. The next physical run must target the unresolved mechanism rather than merely increase sample count until a scalar conjunction passes.
