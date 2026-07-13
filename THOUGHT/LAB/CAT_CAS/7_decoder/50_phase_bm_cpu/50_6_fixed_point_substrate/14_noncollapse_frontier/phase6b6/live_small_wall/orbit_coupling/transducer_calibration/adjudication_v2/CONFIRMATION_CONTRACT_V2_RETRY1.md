# Balanced Transducer Confirmation Contract V2 Retry 1

## Authority

This retry contract is prospective only.

No live run is authorized by this file. No lab device contact, SSH, SCP, ping,
remote cleanup, target process inspection, hardware process, sysctl write,
CPU-frequency change, or PMU capture may occur until a later user directive
explicitly authorizes this exact retry-one confirmation.

The original scientific contract is incorporated by hash:

`CONFIRMATION_CONTRACT_V2.md`

`6a0dee41851f417d6572facad7880d2bb92a364e46ceb4f9c5bc6b00890bead4`

This retry changes only:

- attempt identity;
- failed-attempt provenance;
- PMU platform custody semantics;
- future command;
- future evidence roots.

It does not weaken any scientific law, threshold, restoration rule, schedule
geometry, class vocabulary, or fresh-only adjudication rule from the original
Confirmation V2 contract.

## Failed Attempt Provenance

Attempt zero:

`balanced_transducer_confirmation_v2_0`

Attempt-zero local evidence root:

`phase6b6/live_small_wall/orbit_coupling/transducer_calibration/runs/balanced_transducer_confirmation_v2_0/`

Attempt-zero retained remote root:

`/root/catcas_live_small_wall/balanced_transducer_confirmation_v2_0`

Attempt zero ended during `pmu_platform` preflight before hardware execution,
runtime compilation, PMU event-open, replicate execution, raw capture,
restoration sentinel capture, feature extraction, adjudication, or scientific
classification. No live measurements were revealed.

Because no measurement data was produced in attempt zero, reusing the same trial
geometry does not create adaptive scientific selection. Attempt one remains a
fresh prospective confirmation.

## Retry Identity

Run id:

`balanced_transducer_confirmation_v2_1`

Future local evidence root:

`phase6b6/live_small_wall/orbit_coupling/transducer_calibration/runs/balanced_transducer_confirmation_v2_1/`

Future remote root:

`/root/catcas_live_small_wall/balanced_transducer_confirmation_v2_1`

Attempt-zero roots are immutable and are not cleanup targets for retry-one
controller logic.

## Preserved Scientific Law

The sole confirmatory coordinate remains:

`change_to_dirty`

The q ladder remains:

```text
-1536, -1024, -512, 0, +512, +1024, +1536
```

The schedule remains two fresh replicates and 112 total trial legs with the same
paired crossover geometry, source-order balance, receiver-order balance, q0
null-build/null-test split, line permutation, physical bank pairing, PMU raw
event encodings, coordinate floors, thresholds, restoration laws, and fresh-only
adjudication.

Allowed classifications remain:

- `V1_PARTIAL_CONFIRMED`
- `V1_PARTIAL_V2_TRANSFER_CANDIDATE`
- `V1_PARTIAL_V2_NOT_ESTABLISHED`

Forbidden classifications remain:

- `BALANCED_PHYSICAL_TRANSDUCER_CALIBRATED`
- `ORBITSTATE_PHYSICAL_QUERY_COUPLING_CANDIDATE`
- `SMALL_WALL_CROSSED`

Fresh retry-one evidence may not pool with V1 retained evidence or attempt-zero
failure evidence to satisfy any confirmation law.

## PMU Platform Custody Semantics

`perf_event_paranoid` is an unprivileged access-control setting. The
confirmation target is administered through `root@192.168.137.100`, and the
target process must record and require effective UID zero.

When effective UID is zero, the observed `perf_event_paranoid` value is
diagnostic-only. The controller and target must not fail merely because the
numeric value is `3` or greater, and they must not write any sysctl.

When effective UID is nonzero, the target fails closed unless a separate audited
CAP_PERFMON path is explicitly authorized. This retry does not introduce that
path.

The definitive permission gate is the exact-event runtime PMU preflight:

1. compile the frozen runtime on the target with the strict command;
2. run `--self-test`;
3. run `--pmu-preflight`;
4. require a complete unmultiplexed three-event group with event IDs, receiver
   core 5 before and after, positive cycles, unchanged bytes, and no scientific
   classification.

The preflight is custody and permission evidence only. It performs no experiment
trial and emits no scientific evidence.

The retained empirical fact is:

`balanced_transducer_calibration_1` previously completed 224 PMU records on the
same host while `perf_event_paranoid` was `3`.

## Future Command

The future live command is:

```powershell
$env:CONFIRMATION_V2_RETRY1_COMMIT_BINDING = "<authorized-final-commit>"
$env:CONFIRMATION_V2_RETRY1_LIVE_AUTHORITY = "balanced_transducer_confirmation_v2_1"
.\.venv\Scripts\python.exe THOUGHT\LAB\CAT_CAS\7_decoder\50_phase_bm_cpu\50_6_fixed_point_substrate\14_noncollapse_frontier\phase6b6\live_small_wall\orbit_coupling\transducer_calibration\adjudication_v2\run_confirmation_v2.py --run-id balanced_transducer_confirmation_v2_1 --contract THOUGHT\LAB\CAT_CAS\7_decoder\50_phase_bm_cpu\50_6_fixed_point_substrate\14_noncollapse_frontier\phase6b6\live_small_wall\orbit_coupling\transducer_calibration\adjudication_v2\CONFIRMATION_CONTRACT_V2_RETRY1.md --execute-authorized
```

This command is not authorized by this file. It requires a later explicit live
authorization bound to the frozen commit and retry-one hashes.
