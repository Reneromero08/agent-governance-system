# Independent-Window Public Transducer V3 Contract

## Authority

This package freezes an offline, public, independent-window V3 protocol for the
live-small-wall orbit-coupling transducer lane. It starts from repository head
`524b9e580cb13f547fd7e0638dcf25b3b66e1112` on `main`. It does not authorize
SSH, SCP, ping, remote commands, PMU execution, remote cleanup, or target
inspection.

The retained V1/V2 evidence roots remain immutable:

- `runs/balanced_transducer_calibration_0/`
- `runs/balanced_transducer_calibration_1/`
- `runs/balanced_transducer_confirmation_v2_0/`
- `runs/balanced_transducer_confirmation_v2_1/`

The prior classifications remain preserved exactly:

- `BALANCED_PHYSICAL_TRANSDUCER_PARTIAL`
- `V1_PARTIAL_V2_TRANSFER_CANDIDATE`

The retry-one result remains a transfer candidate. This contract does not
retroactively alter its adjudication.

## V3 Measurement Law

Each mapping leg contains two independent subcaptures:

1. Positive subcapture.
2. Negative subcapture.

Each subcapture must perform, in order:

1. Full receiver baseline.
2. Pre-sentinel on both physical banks.
3. Rebaseline.
4. Complete source encoding with `positive_work = M + q`,
   `negative_work = M - q`, and total source work `4096`.
5. Measurement of exactly one logical bank.
6. Restoration of both physical banks.
7. Post-sentinel on both physical banks.

The positive and negative measured components must each begin from independent
receiver baseline and source encoding. The derived V3 response is:

`F(q) = positive_subcapture_change_to_dirty - negative_subcapture_change_to_dirty`

Raw evidence must record the component windows. Python adjudication derives all
differentials from raw components and rejects runtime-derived delta fields.

## Frozen Geometry

- Q ladder: `[-1536, -1024, -512, 0, 512, 1024, 1536]`
- `M = 2048`
- Source work per subcapture: `4096`
- Source work per mapping leg: `8192`
- Bank lines: `4096`
- Line bytes: `64`
- Line permutation: `A = 257`, `B = 43`
- Source core: `4`
- Receiver core: `5`
- Replicates: `2`
- Mapping-leg records per replicate: `64`
- Total mapping-leg records: `128`
- Total component measurement windows: `256`

Frozen factors:

- `q`
- `mapping`
- `mapping_order`
- `source_order`
- `fresh_process_replicate`
- `subcapture_order`

The V3 `subcapture_order` levels are:

- `positive_subcapture_first`
- `negative_subcapture_first`

The old receiver-order factor is not reused.

## Q0 Null Split

For each replicate and each `source_order x subcapture_order` cell:

- repeat `0` is `null_build`
- repeat `1` is held-out `null_test`

Each replicate therefore has four q0 build crossover pairs and four held-out q0
test crossover pairs. Null-test data must not construct its own ceiling. A
single held-out q0 violation prevents confirmation.

## Allowed And Forbidden Emissions

Allowed V3 scientific classes:

- `PUBLIC_INDEPENDENT_WINDOW_TRANSDUCER_CONFIRMED`
- `PUBLIC_INDEPENDENT_WINDOW_TRANSDUCER_CANDIDATE`
- `PUBLIC_INDEPENDENT_WINDOW_TRANSDUCER_NOT_ESTABLISHED`

Forbidden emissions:

- `BALANCED_PHYSICAL_TRANSDUCER_CALIBRATED`
- `V1_PARTIAL_CONFIRMED`
- `ORBITSTATE_PHYSICAL_QUERY_COUPLING_CANDIDATE`
- `SMALL_WALL_CROSSED`

## Offline Closure

The freeze package must provide schedule artifacts, strict C runtime
self-checks, controller and target self-tests, q0 split validation,
source-work/component-window reconstruction, rejection mocks, source bundle
hashing, JSON parsing, disassembly inspection where a local compiler exists,
and local governance gates. The package must report zero live contact.

Future live work requires a new explicit live authorization and must bind:

- `INDEPENDENT_WINDOW_TRANSDUCER_V3_COMMIT_BINDING=<final_commit>`
- `INDEPENDENT_WINDOW_TRANSDUCER_V3_LIVE_AUTHORITY=independent_window_transducer_v3_0`
