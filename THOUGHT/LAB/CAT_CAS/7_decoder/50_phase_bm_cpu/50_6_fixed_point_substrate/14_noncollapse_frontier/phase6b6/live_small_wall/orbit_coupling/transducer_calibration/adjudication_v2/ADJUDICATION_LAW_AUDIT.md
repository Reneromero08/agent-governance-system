# Balanced Transducer Adjudication Law Audit V2

## Scope

This is an offline retrospective audit of retained V1 evidence from:

`phase6b6/live_small_wall/orbit_coupling/transducer_calibration/runs/balanced_transducer_calibration_1/`

No lab device was contacted. No SSH, SCP, hardware process, or live run was used.
The historical V1 result remains immutable:

`BALANCED_PHYSICAL_TRANSDUCER_PARTIAL`

The V2 program reads only the retained schedule, raw capture, and restoration
sentinels. It reproduces the V1 classification from those retained inputs before
applying corrected adjudication laws.

## Retained Evidence Custody

| retained file | expected sha256 | observed by V2 |
|---|---:|---:|
| `PUBLIC_TRIAL_SCHEDULE.json` | `3c6d499c6085ad9e9168a238ca30c63d0f642f8d26a3af13d25fd2b8f12adff1` | match |
| `RAW_TRANSDUCER_CAPTURE.jsonl` | `942328fde50ed2fca6d0fb620e97e3ebe15c1bd2b7de887e6b04e73d7ba96dab` | match |
| `RESTORATION_SENTINELS.jsonl` | `b516983c5440ab58c7ea73e043d91234bff40f4a1c7e2e2eea4f5e42dcb14299` | match |

V2 packet:

`TRANSDUCER_ADJUDICATION_AUDIT_V2.json`

Packet SHA-256:

`104851ece0dc9d8ff8fcc5d1e443620d77f45022a5116e8f08d60693bd138f42`

V2 audit SHA-256:

`3284bd167dd6139ab143d6076f665b4b0e80074d5b82bb4e34b836e14164cad6`

V2 manifest SHA-256:

`68020db108cbacc6123714cdee070259dc70d894fc3615b5edfb312d0c113c57`

Self-test SHA-256:

`834b74c27fc6690d619671828103a8f971257669b66bec3b03f9d2aa797b0878`

V1 reproduction from schedule/raw/sentinels:

| field | value |
|---|---|
| reproduced V1 status | `BALANCED_PHYSICAL_TRANSDUCER_PARTIAL` |
| expected V1 status | `BALANCED_PHYSICAL_TRANSDUCER_PARTIAL` |
| status match | `true` |
| reproduced internal SHA-256 | `3468a20d0991148ce608ca0391afeffb0e539e8cc7db806e896571b16a4ba58b` |

The V2 tool does not read the retained V1 adjudication JSON when reproducing the V1
classification.

## Result

V2 emits:

`V1_PARTIAL_V2_TRANSFER_CANDIDATE`

Primary V2 coordinate:

`change_to_dirty`

Eligible nondiagnostic coordinate:

`change_to_dirty`

Partial candidate coordinates:

`probe_dirty`, `cycles`, `duration_ns`, `change_to_dirty_per_cycle`,
`probe_dirty_per_cycle`

This is not a calibrated transducer claim. It is not an OrbitState coupling claim,
and it is not `SMALL_WALL_CROSSED`. Because V2 is retrospective, the result requires
a fresh prospective confirmation run under the frozen V2 contract.

## V1 Defect A: q=0 Relative-Error Singularity

V1 used `q = 0` in relative-error invariance/reversal laws even though the expected
logical response is zero at `q = 0`. That makes small absolute residuals dominate
relative-error maxima. V1 already accounted for `q = 0` through null-building and
null-testing components, so V2 treats `q = 0` as a null-region requirement instead
of as a relative-error test.

V1 failed-control maxima:

| coordinate | law | max error | q | abs residual | denominator | nonzero max | q0 dominated |
|---|---|---:|---:|---:|---:|---:|---|
| `change_to_dirty` | logical pointer swap | 1.425287 | 0 | 7.75 | 5.4375 | 0.025838 | yes |
| `change_to_dirty` | physical pointer reversal | 1.425287 | 0 | 7.75 | 5.4375 | 0.025838 | yes |
| `change_to_dirty` | source-order equality | 0.529412 | 0 | 1.125 | 2.125 | 0.019179 | yes |
| `change_to_dirty` | receiver-order equality | 1.691358 | 0 | 17.125 | 10.125 | 0.043235 | yes |
| `probe_dirty` | logical pointer swap | 1.131944 | 0 | 10.1875 | 9.0 | 0.004834 | yes |
| `probe_dirty` | physical pointer reversal | 1.131944 | 0 | 10.1875 | 9.0 | 0.004834 | yes |
| `probe_dirty` | source-order equality | 1.396135 | 0 | 18.0625 | 12.9375 | 0.007321 | yes |
| `probe_dirty` | receiver-order equality | 0.670213 | 0 | 3.9375 | 5.875 | 0.017403 | yes |
| `cycles` | logical pointer swap | 0.627995 | 0 | 899.25 | 1431.9375 | 0.071177 | yes |
| `cycles` | physical pointer reversal | 0.627995 | 0 | 899.25 | 1431.9375 | 0.071177 | yes |
| `cycles` | source-order equality | 1.251161 | 0 | 3282.5 | 2623.5625 | 0.043162 | yes |
| `cycles` | receiver-order equality | 1.767207 | 0 | 14914.125 | 8439.375 | 0.27293 | no |
| `duration_ns` | logical pointer swap | 1.307802 | 0 | 1370.25 | 1047.75 | 0.075812 | yes |
| `duration_ns` | physical pointer reversal | 1.307802 | 0 | 1370.25 | 1047.75 | 0.075812 | yes |
| `duration_ns` | source-order equality | 1.484816 | 0 | 2090.25 | 1407.75 | 0.028199 | yes |
| `duration_ns` | receiver-order equality | 1.762198 | 0 | 5374.375 | 3049.8125 | 0.236739 | yes |
| `change_to_dirty_per_cycle` | logical pointer swap | 0.000218 | 1536 | 0.000218 | 1.0 | 0.000218 | no |
| `change_to_dirty_per_cycle` | physical pointer reversal | 0.000218 | 1536 | 0.000218 | 1.0 | 0.000218 | no |
| `change_to_dirty_per_cycle` | source-order equality | 0.000151 | 0 | 0.000151 | 1.0 | 0.000125 | yes |
| `change_to_dirty_per_cycle` | receiver-order equality | 0.000856 | 0 | 0.000856 | 1.0 | 0.000806 | yes |
| `probe_dirty_per_cycle` | logical pointer swap | 0.000645 | 1024 | 0.000645 | 1.0 | 0.000645 | no |
| `probe_dirty_per_cycle` | physical pointer reversal | 0.000645 | 1024 | 0.000645 | 1.0 | 0.000645 | no |
| `probe_dirty_per_cycle` | source-order equality | 0.00048 | -1024 | 0.00048 | 1.0 | 0.00048 | no |
| `probe_dirty_per_cycle` | receiver-order equality | 0.002059 | 512 | 0.002059 | 1.0 | 0.002059 | no |

For `change_to_dirty`, all four V1 failed controls were produced at `q = 0`; the
largest nonzero-q relative error was 0.043235, below the frozen 0.25 tolerance.

## V2 q=0 Semantics

V2 keeps `q = 0` as a null custody test:

- logical mappings must remain inside the retained null ceiling;
- source-order, receiver-order, and mapping residuals still contribute to the null
  ceiling;
- `q = 0` physical A-minus-B reversal is report-only, because equal work was applied
  to both banks;
- no near-zero relative-error test can determine calibration.

For the primary `change_to_dirty` coordinate:

| q0 field | value |
|---|---:|
| null ceiling | 77.0 |
| max held-out q0 null-test abs | 63.0 |
| max q0 logical mapping abs | 77.0 |
| q0 logical mappings inside null | `true` |
| q0 physical A-minus-B pair sum | 108.0 |
| q0 physical pair-sum ceiling | 154.0 |
| q0 physical pair sum inside pair null bound | `true` |
| q0 physical reversal applicable | `false` |

`probe_dirty` remains partial because its q0 logical mapping maximum is outside its
null ceiling even though the nonzero paired crossover errors are small.

## V2 Defect B Repair: Paired Crossover Analysis

V2 evaluates the frozen crossover pairs by `pair_index` instead of by marginal mapping
means. For every nonzero-q pair:

```text
logical_pair_residual = logical_F_mapping0 - logical_F_mapping1
physical_pair_residual = physical_A_minus_B_mapping0 + physical_A_minus_B_mapping1
```

The tolerance remains 0.25.

| coordinate | max logical pair error | max physical pair error | paired laws pass |
|---|---:|---:|---|
| `change_to_dirty` | 0.204403 | 0.204403 | yes |
| `probe_dirty` | 0.092369 | 0.092369 | logical q0 null fails |
| `cycles` | 0.513099 | 0.513099 | no |
| `duration_ns` | 0.502528 | 0.502528 | no |
| `change_to_dirty_per_cycle` | 0.426052 | 0.426052 | no |
| `probe_dirty_per_cycle` | 0.497825 | 0.497825 | no |

The primary coordinate passes pairwise nonzero crossover. Several other coordinates
show visible transfer shape but fail genuine nonzero-q crossover or receiver-order
controls.

## V2 Defect C Repair: Source and Receiver Order

V2 excludes `q = 0` from source-order and receiver-order relative invariance tests and
adds gain-normalized checks over nonzero q.

| coordinate | nonzero source max | nonzero receiver max | diagnosis |
|---|---:|---:|---|
| `change_to_dirty` | 0.019179 | 0.043235 | V1 failure was q0 singularity dominated |
| `probe_dirty` | 0.007321 | 0.017403 | q0 logical null still fails |
| `cycles` | 0.043162 | 0.27293 | receiver-order nonzero instability and restoration failure |
| `duration_ns` | 0.028199 | 0.236739 | diagnostic-only, restoration and gain-normalized controls fail |
| `change_to_dirty_per_cycle` | 0.037127 | 0.402507 | receiver-order nonzero instability after unit floor repair |
| `probe_dirty_per_cycle` | 0.049928 | 0.391531 | receiver-order nonzero instability after unit floor repair |

Mechanism classification:

| coordinate | mechanism diagnosis |
|---|---|
| `change_to_dirty` | candidate; V1 failed controls were q0 singularity dominated |
| `probe_dirty` | visible transfer but q0 logical null fails |
| `cycles` | visible transfer but nonzero receiver/crossover instability and restoration failure |
| `duration_ns` | visible transfer but diagnostic-only and restoration/gain controls fail |
| `change_to_dirty_per_cycle` | visible transfer but receiver/crossover instability remains |
| `probe_dirty_per_cycle` | visible transfer but receiver/crossover instability remains |

## V2 Defect D Repair: Coordinate-Specific Floors

V2 freezes unit-specific absolute floors:

| coordinate | absolute floor |
|---|---:|
| `change_to_dirty` | 1.0 |
| `probe_dirty` | 1.0 |
| `cycles` | 1.0 |
| `duration_ns` | 1.0 |
| `change_to_dirty_per_cycle` | 0.000001 |
| `probe_dirty_per_cycle` | 0.000001 |

The normalized-coordinate floors are therefore dimensional rather than inherited from
count/cycle/ns units.

| coordinate | null ceiling | gain floor | resolved nonzero q |
|---|---:|---:|---|
| `change_to_dirty` | 77.0 | 231.0 | all six nonzero q |
| `probe_dirty` | 75.0 | 225.0 | all six nonzero q |
| `cycles` | 20135.0 | 60405.0 | all six nonzero q |
| `duration_ns` | 10686.0 | 32058.0 | all six nonzero q |
| `change_to_dirty_per_cycle` | 0.0008601578166163625 | 0.0025804734498490875 | all six nonzero q |
| `probe_dirty_per_cycle` | 0.002292948798212395 | 0.006878846394637185 | all six nonzero q |

This repairs the V1 under-resolution of normalized coordinates, but it does not make
the normalized coordinates eligible because their paired crossover and receiver-order
controls still fail.

Sol identified one additional unit issue in the draft V2 code: gain-normalized
agreement compares values in `coordinate/q`, so its floor must also be in
`coordinate/q`. The final V2 packet uses `coordinate_floor / 512` as the gain-space
floor:

| coordinate | gain-space floor |
|---|---:|
| `change_to_dirty` | 0.001953125 |
| `probe_dirty` | 0.001953125 |
| `cycles` | 0.001953125 |
| `duration_ns` | 0.001953125 |
| `change_to_dirty_per_cycle` | 0.000000001953125 |
| `probe_dirty_per_cycle` | 0.000000001953125 |

## Restoration Interpretation

Restoration remains coordinate-specific. Cycles and duration failures do not invalidate
separate dirty-count coordinates whose own restoration laws passed.

| coordinate | restoration passed | max error | max mode | trial | q | pre | post |
|---|---|---:|---|---:|---:|---:|---:|
| `change_to_dirty` | yes | 0.0 | bank A | 0 | 0 | 0.0 | 0.0 |
| `probe_dirty` | yes | 0.0 | bank A | 0 | 0 | 0.0 | 0.0 |
| `cycles` | no | 1.9941002949852507 | differential | 99 | 512 | 3729.0 | -3707.0 |
| `duration_ns` | no | 1.8901209677419355 | differential | 43 | -1024 | 1984.0 | -1766.0 |
| `change_to_dirty_per_cycle` | yes | 0.0 | bank A | 0 | 0 | 0.0 | 0.0 |
| `probe_dirty_per_cycle` | yes | 0.0 | bank A | 0 | 0 | 0.0 | 0.0 |

Cycles restoration detail:

| mode | max error | trial | q | pre | post |
|---|---:|---:|---:|---:|---:|
| bank A | 0.21747845259571055 | 68 | 512 | 34923.0 | 27328.0 |
| bank B | 0.1941884575509734 | 0 | 0 | 34724.0 | 27981.0 |
| common mode | 0.18077532708271768 | 0 | 0 | 74446.0 | 60988.0 |
| differential | 1.9941002949852507 | 99 | 512 | 3729.0 | -3707.0 |

Duration restoration detail:

| mode | max error | trial | q | pre | post |
|---|---:|---:|---:|---:|---:|
| bank A | 0.4277230699842141 | 103 | 512 | 26606.0 | 15226.0 |
| bank B | 0.16347623998532199 | 0 | 0 | 16351.0 | 13678.0 |
| common mode | 0.2911642712452976 | 103 | 512 | 41468.0 | 29394.0 |
| differential | 1.8901209677419355 | 43 | -1024 | 1984.0 | -1766.0 |

## Primary Coordinate Summary

`change_to_dirty` aggregate means:

| q | F(q) |
|---:|---:|
| -1536 | -1881.09375 |
| -1024 | -1274.125 |
| -512 | -633.625 |
| 0 | -1.5625 |
| 512 | 642.28125 |
| 1024 | 1267.0625 |
| 1536 | 1911.125 |

The coordinate passes:

- held-out q0 null;
- sign;
- oddness;
- gain;
- monotonicity;
- paired logical pointer invariance;
- paired physical pointer reversal;
- nonzero-q source-order invariance;
- nonzero-q receiver-order invariance;
- gain-normalized source and receiver checks;
- stratum transfer;
- replicate consistency;
- coordinate-specific restoration;
- retained-evidence hash custody.

## Read-Only Sol Audit Outcome

GPT-5.6 Sol Extra High ran a read-only audit under:

```text
CUSTODY: READ_ONLY
NO_CHECKOUT_MUTATION: true
NO_LAB_DEVICE_CONTACT: true
NO_GIT_WRITE: true
```

Sol's conclusion was that `V1_PARTIAL_V2_TRANSFER_CANDIDATE` is mathematically
defensible as a retrospective interpretation of the retained evidence, but it blocks
promotion to `V1_PARTIAL_CONFIRMED` or any custody-final claim without fresh
preregistered evidence.

Material Sol findings and repairs:

| finding | disposition |
|---|---|
| q0 must be excluded from relative transfer laws | accepted; V2 tests q0 as null membership only |
| q0 physical A-minus-B pair sum should be bounded by `2C`, not left report-only | repaired; final packet records `108 <= 154` for `change_to_dirty` |
| pairwise crossover is superior to marginal mapping means | accepted; V2 uses `pair_index` |
| gain-normalized agreement needs a gain-space floor | repaired; final packet uses `coordinate_floor / 512` |
| substantive V1 safeguards are preserved | recorded; V2 retains null, sign, oddness, gain, monotonicity, order, stratum, restoration, replicate, diagnostic-coordinate, and claim-ceiling gates |
| V2 alone does not provide false-positive rate or multiplicity control | recorded; final status remains retrospective candidate only |
| packet should bind governing hashes and self-test result | repaired; final packet includes manifest SHA `68020db108cbacc6123714cdee070259dc70d894fc3615b5edfb312d0c113c57` |

Sol's fresh-confirmation recommendation is reflected in
`CONFIRMATION_CONTRACT_V2.md`: predeclare `change_to_dirty` as the sole confirmatory
coordinate, freeze the repaired V2 law first, collect two fresh processes under a new
frozen schedule, analyze the fresh run alone, do not revise thresholds after
observation, and keep any permutation/slope statistics as confirmatory support rather
than as a replacement for the deterministic V2 laws.
