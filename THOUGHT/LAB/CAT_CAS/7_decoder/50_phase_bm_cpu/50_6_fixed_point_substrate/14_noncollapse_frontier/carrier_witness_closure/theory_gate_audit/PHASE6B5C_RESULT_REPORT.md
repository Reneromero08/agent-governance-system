# Phase 6B.5C Transfer-Aware T48 Result

**Primary outcome:** `TRANSFER_EQUIVARIANCE_SUPPORTED`
**Official strict carrier closure:** `PARTIAL` (unchanged)
**Execution branch/head:** `phase6b/carrier-witness-closure` at `3d45766dc5fef090272c6a05c384117ebe1a4b52`
**Execution host:** `root@192.168.137.100` (`catcas`)
**New physical acquisition:** none

## Bindings

```text
campaign_id = phase6b5_t48_d32b1bed_20260619
campaign_source_commit = d32b1bed0deae1b907a07eeed018b924244e9ea2
campaign_manifest_sha256 = cbcd2a19d6dd3bc478244f77888aa87eb043003a7685caa17ff13fe4d47e6487
analysis_source_sha256 = 6892be9beed370b8f99b18e1eee5b4f3e66c265fb996aa5d9545575907b00515
complex_geometry_source_sha256 = 13ea91fc9b000062fd840ce0b07b3909cb570eaf6e1b214dcc115d2cf5a9b6a3
analysis_manifest_sha256 = 93ccb5fb5d9cbc96c25c52797ea0dd0693810997a369e714cfe57109af35ff2b
```

All 14 run manifests were verified and every `raw_samples.bin` SHA-256 was
recomputed. The ten-file output manifest verifies without error.

## Execution

```bash
./run_phase6b5c.sh \
  /root/catcas_evidence/phase6b5_t48_d32b1bed_20260619 \
  /root/catcas_phase6b5c_75e841f0/14_noncollapse_frontier/carrier_witness_closure/theory_gate_audit/results/phase6b5c_t48_d32b1bed_20260619
```

The first execution exposed a portable-path defect in the gate-layer binding:
it resolved the repository-derived official decomposition relative to the
external campaign root. The binding now resolves from the analysis source tree,
records a repository-relative path, size, and SHA-256, and has regression
coverage. The first generated output was deleted and the complete analysis was
rerun from immutable inputs.

## Chart Selection

Every matrix run selected `C0_scalar`. Eleven runs satisfied the frozen
calibration-only criterion. Route `4:5`, seed `4` selected C0 diagnostically with
`CALIBRATION_CHART_UNSTABLE`.

| Run | Status | Fit rows | Validation rows | Validation residual | Positive margin |
|---|---|---:|---:|---:|---:|
| v2s3 seed 0 | stable | 16 | 12 | 0.3611 | 1.000 |
| v2s3 seed 1 | stable | 16 | 12 | 0.3767 | 1.000 |
| v2s3 seed 2 | stable | 16 | 12 | 0.4295 | 1.000 |
| v2s3 seed 3 | stable | 16 | 12 | 0.4342 | 1.000 |
| v2s3 seed 4 | stable | 16 | 12 | 0.4876 | 1.000 |
| v2s3 seed 5 | stable | 16 | 12 | 0.4056 | 1.000 |
| v4s5 seed 0 | stable | 16 | 12 | 0.3105 | 1.000 |
| v4s5 seed 1 | stable | 16 | 12 | 0.3226 | 1.000 |
| v4s5 seed 2 | stable | 16 | 12 | 0.3174 | 1.000 |
| v4s5 seed 3 | stable | 16 | 12 | 0.3225 | 1.000 |
| v4s5 seed 4 | unstable | 16 | 12 | 0.7187 | 1.000 |
| v4s5 seed 5 | stable | 16 | 12 | 0.3148 | 1.000 |

C1, C2, and C3 were selected zero times. This is evidence for a minimal scalar
complex receiver chart on these data, not a complete physical operator.

## Held-Out Relations

| Run | Real positive margin | Real residual | Phase-aligned residual | Wrong actual over declared | Phase MAE | Phase resultant |
|---|---:|---:|---:|---:|---:|---:|
| v2s3 seed 0 | 1.000 | 0.3606 | 0.3578 | 1.000 | 0.0430 | 0.9987 |
| v2s3 seed 1 | 1.000 | 0.3810 | 0.3765 | 1.000 | 0.0441 | 0.9985 |
| v2s3 seed 2 | 1.000 | 0.4031 | 0.4003 | 1.000 | 0.0637 | 0.9977 |
| v2s3 seed 3 | 1.000 | 0.3995 | 0.3944 | 1.000 | 0.0674 | 0.9966 |
| v2s3 seed 4 | 1.000 | 0.4658 | 0.4567 | 1.000 | 0.0860 | 0.9961 |
| v2s3 seed 5 | 1.000 | 0.3855 | 0.3835 | 1.000 | 0.0461 | 0.9985 |
| v4s5 seed 0 | 1.000 | 0.3122 | 0.3121 | 1.000 | 0.0172 | 0.9997 |
| v4s5 seed 1 | 1.000 | 0.3180 | 0.3180 | 1.000 | 0.0237 | 0.9995 |
| v4s5 seed 2 | 1.000 | 0.3178 | 0.3177 | 1.000 | 0.0199 | 0.9996 |
| v4s5 seed 3 | 1.000 | 0.3219 | 0.3210 | 1.000 | 0.0223 | 0.9996 |
| v4s5 seed 4 | 0.958 | 0.8013 | 0.8006 | 1.000 | 0.0913 | 0.9934 |
| v4s5 seed 5 | 1.000 | 0.3203 | 0.3202 | 1.000 | 0.0213 | 0.9995 |

Wrong execution-over-declaration succeeds on 24/24 held-out wrong rows in every
run. Median wrong margins are positive in every run (`0.493` for route `4:5`,
seed `4`; approximately `0.994-1.209` otherwise). This is the primary no-smuggle
witness.

Pairwise phase MAE is `0.0259-0.1649` radians. Shuffled-null MAE is
`1.2425-1.7953` radians. Phase equivariance is therefore supported in all 12
matrix runs.

## Pseudo Permutation Covariance

For every route and seed, the exact executed permutation beats both the canonical
codeword and a deterministic unrelated permutation on 24/24 held-out pseudo
rows. Median exact-over-canonical margins are `0.369-1.062`; median
exact-over-unrelated margins are `0.352-1.078`. Pseudo is a changed basis
relation, not carrier-off.

## Controls

The silent median vector norm is `0.003354`, versus matrix reference `0.183936`
(ratio `0.0182`). Scramble canonical positive-margin fractions are
`0.292-0.306` under the six primary-route charts. No chart was fit on a control.
Both controls remain null.

## Route and Session Relation

Same-route and cross-route transferred charts both have median positive-margin
fraction `1.000`; median residuals are `0.3873` and `0.3866`, respectively.
Matched-seed normalized Gram differences are approximately `5e-14` to `1e-11`.
Because all selected charts are scalar, normalized Gram equality and the
diagnostic route map are structurally degenerate; they do not identify a
complete route operator or prove route-specific conjugacy.

## Ordered Path

Elapsed-position versus magnitude rank associations are small (`-0.0469` to
`0.0342`). Bin-position associations are larger and generally negative
(`-0.4006` to `-0.0508`). Median capture lateness is approximately `0.000258 s`;
median sample count is `1520`; temperatures span `40.375-44.0 C`; and the
frequency proxy remains `1,600,000 kHz` throughout. Prefix/suffix records are
retained in `ordered_path_analysis.json`.

Tone and within-symbol position were never independently varied. No causal
ordered-path claim is allowed from this fixed tone order.

## Seed 4

Route `4:5`, seed `4` is classified `CHART_FAILURE`. Its calibration chart is
unstable and its held-out residual is elevated, but real positive margin is
`0.958`, wrong execution-over-declaration is `1.000`, phase MAE is `0.0913`,
pseudo covariance is `1.000`, telemetry is stable, and no monotonic elapsed drift
explains the result. The carrier relations survive while the frozen calibration
fit criterion fails.

## Claim and Next Control

Supported claim:

```text
On the retained T48 campaign and tested routes, sender-owned mode, phase, and
exact pseudo-permutation relations survive a calibration-only minimal scalar
complex receiver chart, including execution-over-declaration controls.
```

The recommended next physical control is a separately frozen
reversed/randomized tone-order campaign. It must separate tone identity from
within-symbol path position. It is proposed, not authorized.

Not proven: physical `HoloGeometry`, a complete physical operator, physical
restoration, target coupling, orientation recovery, or a Small Wall crossing.
