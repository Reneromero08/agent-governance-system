# OrbitState Independent-Window Contract V2

## Status

This package freezes a private OrbitState independent-window wall-crossing
experiment. It is live-capable, but this freeze performs zero live target
contact. A later live execution requires all three exact controller authority
environment variables and a matching committed manifest.

Allowed target result classes:

- `ORBITSTATE_INDEPENDENT_COUPLING_CONFIRMED`
- `ORBITSTATE_INDEPENDENT_COUPLING_CANDIDATE`
- `ORBITSTATE_INDEPENDENT_COUPLING_NOT_ESTABLISHED`

The target adjudicator may not emit `SMALL_WALL_CROSSED`.

## Retained Public Transducer

The public carrier is the retained V3 independently reconstructed
Change-to-Dirty transducer at commit
`4762b5b49b308ae4aca8e141113e4fafe4b0f81e`.

Retained evidence hashes are bound in `PUBLIC_TRANSDUCER_REFERENCE.json`.
The historical V3 class remains
`PUBLIC_INDEPENDENT_WINDOW_TRANSDUCER_CANDIDATE`; this contract does not
rewrite that adjudication.

Prospective constants:

- `PUBLIC_Q0_ABSOLUTE_BOUND = 152.0`
- `PRIVATE_ODD_SIGNAL_FLOOR = 456.0`
- `RELATIONAL_TOLERANCE = 0.25`

The `152` bound is frozen before private evidence from the V3 aggregate build
ceiling `76`, the predeclared physical pair bound `2 * 76`, the largest retained
held-out mapping `100`, and the largest retained held-out pair residual `127`.
It is never refit after private data exists.

## OrbitState Source Formula

The source-side private structure is:

```c
typedef struct {
    uint32_t modulus;
    uint32_t member;
} OrbitState;
```

Constants:

- `N = 256`
- `d = 23`
- `fold(d) = 233`
- `quantization scale = 1536`
- `base work M = 2048`
- public decoder phases: `0`, `pi/2`, `pi`, `3pi/2`

For preprojection:

```text
phi = 2*pi*member/N
r_theta(member) = cos(phi - theta)
q_theta = round(1536 * r_theta)
```

Clamp only to `[-1536, +1536]`.

Balanced physical work:

```text
positive_work = 2048 + q_theta
negative_work = 2048 - q_theta
total_work = 4096
```

The source computes `q_theta` from the live `OrbitState`; the receiver schedule
does not contain q or work values.

## Frozen Conditions

The nine private conditions are:

- `pre_projection_d`
- `pre_projection_fold`
- `source_off`
- `query_off`
- `post_projection`
- `declaration_sham`
- `query_scramble`
- `equal_orbit_odd_zero`
- `source_polarity_inversion_d`

The public schedule contains only opaque group/run identifiers, replicate,
public decoder phase, physical mapping, order labels, and randomized public
ordinal. The sealed private source map binds opaque IDs to condition,
`OrbitState`, response mode, private source phase sequence, polarity inversion,
and source-off dummy mode.

## Process Boundary

The receiver allocates experiment-owned shared banks and forks a source worker.
The source child opens the private source map only after process separation and
is pinned to core 4. The receiver is pinned to core 5 and sends only opaque run
ID plus execution synchronization. The receiver feature extractor never opens or
parses the private source map, source receipts, condition, member, response mode,
q, or work fields.

## Independent Windows

Each mapping leg contains two independent component windows:

1. receiver full baseline
2. pre-sentinels
3. receiver rebaseline
4. source child computes and applies complete OrbitState-derived encoding
5. measure exactly one logical bank
6. receiver restoration
7. post-sentinels

Physical mapping crossover:

```text
map0: positive -> A, negative -> B
map1: positive -> B, negative -> A
```

Logical response:

```text
F_theta = positive_component_change_to_dirty - negative_component_change_to_dirty
```

Physical response:

```text
A_minus_B
```

## Schedule Geometry

Per replicate:

```text
9 conditions * 4 public phases * 2 mapping legs = 72 mapping-leg records
```

Total:

```text
144 mapping-leg records
288 independent component windows
2016 public stage receipts
288 sealed source receipts
```

## Feature Freeze and Unblinding

The receiver-only extractor consumes only the public schedule, raw receiver PMU
records, receiver sentinels, and public stage receipts. It writes
`ORBITSTATE_RECEIVER_FEATURES.json` and freezes
`ORBITSTATE_RECEIVER_FEATURES.sha256`.

Only after that hash is frozen may adjudication join the sealed private source
map, sealed source receipts, and receiver features. Any feature mutation after
unblinding is a hard failure.

## Complex Decoder

For each opaque condition group and mapping:

```text
Z = (2/4) * sum_k(F_theta_k * exp(i * theta_k))
```

The two logical mapping results are averaged only after mapping-invariance checks
pass. Decoder weights are not fit from live evidence.

## Acceptance Laws

Target/fold geometry must pass separately in both fresh replicates and aggregate:

- `Re(Z_d)` and `Re(Z_fold)` have relative error `<= 0.25`.
- `Im(Z_d) * Im(Z_fold) < 0`.
- `abs(Im(Z_d))` and `abs(Im(Z_fold))` agree within `0.25`.
- `min(abs(Im(Z_d)), abs(Im(Z_fold))) > 456`.
- `Z_fold` is componentwise within `0.25` of `conjugate(Z_d)`.

Source polarity:

- `Z_polarity_inversion` is componentwise within `0.25` of `-Z_d`.

Null/control laws:

- The imaginary components of `source_off`, `query_off`, `post_projection`,
  `declaration_sham`, `query_scramble`, and `equal_orbit_odd_zero` are bounded
  by `152`.
- The complete complex magnitudes of `source_off`, `query_off`,
  `declaration_sham`, and `query_scramble` are bounded by `152`.

Phase-level physical transfer:

- For target/fold/polarity phases, `sign(F_theta)` follows `sign(q_theta)`.
- Nonzero `q_theta` requires mapping invariance and physical reversal within
  `0.25`.
- For `abs(q_theta) < 256`, the fixed `152` absolute bound is used.

Custody:

- component bytes unchanged;
- source work restored;
- PMU windows unmultiplexed;
- receiver windows on core 5;
- source operations on core 4;
- event IDs valid;
- process separation receipts valid;
- both fresh process replicates independently pass.

Contradictory replicates cannot be promoted by aggregate averaging.

## Small Wall Promotion

`SMALL_WALL_CROSSED` may be written only after:

1. the fresh target class is `ORBITSTATE_INDEPENDENT_COUPLING_CONFIRMED`;
2. both fresh replicates independently pass;
3. receiver features were frozen before unblinding;
4. no private source field entered receiver feature extraction;
5. restoration and physical mapping controls pass;
6. a GPT-5.6 Sol Extra High read-only claim audit finds no material blocker;
7. GPT-5.5 independently verifies copied source and evidence.

No second private confirmation run is required by this contract when all seven
conditions pass.
