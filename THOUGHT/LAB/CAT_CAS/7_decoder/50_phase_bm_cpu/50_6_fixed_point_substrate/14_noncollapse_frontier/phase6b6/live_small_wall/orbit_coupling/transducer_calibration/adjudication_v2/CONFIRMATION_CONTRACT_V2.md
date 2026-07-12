# Balanced Transducer Confirmation Contract V2

## Authority

This contract is prospective only.

No live run is authorized by this file. No lab device contact, SSH, SCP, hardware
process, CPU-frequency change, or PMU capture may occur until a later user directive
explicitly authorizes this exact confirmation.

## Purpose

The retained V1 evidence remains:

`BALANCED_PHYSICAL_TRANSDUCER_PARTIAL`

The offline V2 audit classifies that retained evidence as:

`V1_PARTIAL_V2_TRANSFER_CANDIDATE`

This confirmation contract defines the smaller fresh run required to remove the
retrospective adjudication-law concern without widening the scientific claim.

The sole confirmatory coordinate is:

`change_to_dirty`

Other coordinates may be reported as diagnostics or secondary evidence, but they may
not replace `change_to_dirty` after observation.

## Frozen Result Classes

The confirmation adjudicator may emit only:

- `V1_PARTIAL_CONFIRMED`
- `V1_PARTIAL_V2_TRANSFER_CANDIDATE`
- `V1_PARTIAL_V2_NOT_ESTABLISHED`

It may not emit:

- `BALANCED_PHYSICAL_TRANSDUCER_CALIBRATED`
- `ORBITSTATE_PHYSICAL_QUERY_COUPLING_CANDIDATE`
- `SMALL_WALL_CROSSED`

## Confirmation Geometry

Run id:

`balanced_transducer_confirmation_v2_0`

Evidence root:

`phase6b6/live_small_wall/orbit_coupling/transducer_calibration/runs/balanced_transducer_confirmation_v2_0/`

Required q ladder:

```text
-1536, -1024, -512, 0, +512, +1024, +1536
```

Fresh processes:

- replicate 0: fresh stimulus and measurement process pair;
- replicate 1: fresh stimulus and measurement process pair.

Schedule size:

- 112 raw trials;
- two fresh replicates;
- for every replicate, q, source order, and receiver order, execute one true mapping
  crossover pair;
- each pair has the same q, source order, receiver order, allocated A/B banks, and
  receiver baseline re-established between mapping legs;
- mapping order is counterbalanced across q/source/receiver strata.

This is smaller than the V1 calibration schedule because it removes the doubled
mapping-order cross while preserving true paired crossover and balanced order factors.

## Required Custody

The confirmation must preserve:

- frozen V2 source, threshold, class-vocabulary, and self-test manifest before any
  live execution;
- two fresh processes;
- the full q ladder above;
- true paired pointer crossover by `pair_index`;
- balanced source order;
- balanced receiver order;
- held-out q0 null;
- PMU custody through ordinary Linux PMU reads only;
- no voltage writes;
- no direct MSR reads or writes;
- no physical-address access;
- no cache-set mapping;
- restoration sentinels for every trial;
- exact source bundle hash;
- exact schedule hash;
- exact raw capture hash;
- exact restoration-sentinel hash;
- exact post-copy local evidence hashes;
- CPU-frequency policy snapshot and restoration if a later live directive authorizes
  frequency preparation.

The repaired V2 law must be hashed before the future run. The fresh run must be
analyzed alone; it may not pool with retained V1 evidence to satisfy a confirmation
law.

## q0 Null Split

The q0 trials must be split into null-build and null-test roles before execution.

Null-build q0 data may set:

- q0 absolute ceiling;
- q0 mapping residual;
- q0 source-order residual;
- q0 receiver-order residual;
- q0 mapping-order residual;
- sentinel variation.

Null-test q0 data must remain held out for the zero law.

No q0 relative-error invariance, pointer-reversal, source-order, or receiver-order
failure may determine the candidate class.

## Frozen V2 Laws

For at least one nondiagnostic coordinate, the fresh confirmation must pass:

- held-out q0 null;
- sign;
- oddness;
- gain;
- monotonicity;
- paired logical pointer invariance over nonzero q;
- paired physical pointer reversal over nonzero q;
- direct nonzero-q source-order invariance;
- direct nonzero-q receiver-order invariance;
- gain-normalized source-order invariance;
- gain-normalized receiver-order invariance;
- stratum transfer;
- replicate consistency;
- coordinate-specific restoration;
- hard custody and schedule integrity.

For promotion to `V1_PARTIAL_CONFIRMED`, that nondiagnostic coordinate must be the
predeclared `change_to_dirty` coordinate in aggregate and in both fresh processes.

The tolerance values remain frozen:

| law | tolerance |
|---|---:|
| oddness | V1 tolerance |
| pointer/crossover | 0.25 |
| source/receiver order | 0.25 |
| replicate consistency | V1 tolerance |
| restoration | V1 tolerance |
| gain multiplier | V1 multiplier |

Coordinate floors remain:

| coordinate | absolute floor |
|---|---:|
| `change_to_dirty` | 1.0 |
| `probe_dirty` | 1.0 |
| `cycles` | 1.0 |
| `duration_ns` | 1.0 |
| `change_to_dirty_per_cycle` | 0.000001 |
| `probe_dirty_per_cycle` | 0.000001 |

Duration remains diagnostic-only.

No threshold revision, coordinate substitution, or retry may occur after observing the
fresh capture. A pair-block q-label permutation test and slope confidence interval may
be included as statistical support, but they do not replace the deterministic V2 laws
and may not be used to tune thresholds.

## Future Command

The exact future command, when explicitly authorized later, is:

```powershell
.\.venv\Scripts\python.exe THOUGHT\LAB\CAT_CAS\7_decoder\50_phase_bm_cpu\50_6_fixed_point_substrate\14_noncollapse_frontier\phase6b6\live_small_wall\orbit_coupling\transducer_calibration\adjudication_v2\run_confirmation_v2.py --run-id balanced_transducer_confirmation_v2_0 --contract THOUGHT\LAB\CAT_CAS\7_decoder\50_phase_bm_cpu\50_6_fixed_point_substrate\14_noncollapse_frontier\phase6b6\live_small_wall\orbit_coupling\transducer_calibration\adjudication_v2\CONFIRMATION_CONTRACT_V2.md
```

The command is not executed by this offline task.

## Expected Evidence Files

The future confirmation must produce:

- `CONFIRMATION_V2_MANIFEST.json`
- `CONFIRMATION_PUBLIC_TRIAL_SCHEDULE.json`
- `CONFIRMATION_PUBLIC_TRIAL_SCHEDULE.tsv`
- `CONFIRMATION_SOURCE_BUNDLE.tar.gz`
- `CONFIRMATION_SOURCE_HASHES.json`
- `CONFIRMATION_RUNTIME_STDOUT_REPLICATE_0.txt`
- `CONFIRMATION_RUNTIME_STDERR_REPLICATE_0.txt`
- `CONFIRMATION_RUNTIME_STDOUT_REPLICATE_1.txt`
- `CONFIRMATION_RUNTIME_STDERR_REPLICATE_1.txt`
- `RAW_TRANSDUCER_CAPTURE.jsonl`
- `RESTORATION_SENTINELS.jsonl`
- `TRANSDUCER_FEATURES_V2.json`
- `TRANSDUCER_ADJUDICATION_CONFIRMATION_V2.json`
- `FINAL_RESULT_CONFIRMATION_V2.json`
- `COPYBACK_MANIFEST.json`
- `LIVE_CUSTODY_LOG.json`

The final result must bind every hash needed to reproduce the schedule, raw capture,
restoration sentinels, V2 features, and V2 adjudication packet.
