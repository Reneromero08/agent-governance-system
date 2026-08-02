# I2B Second-Generator and Inverse Candidate Audit

## Result

```text
I2B_SECOND_GENERATOR_AND_INVERSE_DESIGN_COMPLETE
NO_PHYSICALLY_SUPPORTED_GENERATOR_PAIR
NO_QUALIFIED_INVERSE
MULTI_DESTINATION_PARTIAL_OVERLAP_GRAMMAR_DESIGN_ONLY
I2C_SYNTHETIC_BIDIRECTIONAL_OWNERSHIP_MODEL_NEXT
NO_LIVE_EXECUTION_AUTHORIZED
FAMILY10H_TRANSPORT_OPERATOR_NOT_ESTABLISHED
FAMILY10H_HOLONOMY_NOT_ESTABLISHED
SMALL_WALL_NOT_CROSSED
```

I2B attacks the obvious ways to turn the accepted same-value-store primitive into two
generators. It binds the decision to four retained physical checkpoints and the I2A
compile-only interface. It does not reactivate any historical package or authorize a new
transaction.

## 1. Retained primitive

The retained coherence checkpoint establishes a byte-preserving remote same-value store
as a controlled ownership-intent operation:

```text
classification = CONTROLLED_COHERENCE_STATE_FOUND
change_to_dirty(remote_store_same_value) = 2104
probe_dirty(remote_store_same_value) = 5728
all operator windows restored logical bytes
```

This is sufficient to justify an extraction target. It does not establish a state map,
a second generator, an inverse, path dependence, or holonomy.

## 2. Rejected pair families

### Remote read versus remote store

The retained read/store path rectangle used remote reads and remote same-value stores on
two line sets. Its normalized areas were:

```text
forward = 3.149513983986e-09
reverse = 3.0244653157e-09
shuffle = 9.150610884539e-10
reverse_shuffle = -1.986661616106e-10
```

The forward and reverse areas had the same sign. The frozen result was:

```text
sign_reversal = false
controls_small = false
path_dependence_pilot = false
PATH_RW_OBSERVE_NOT_ESTABLISHED
```

Remote read remains a control/readout family, not a supported second generator.

### Route 4-to-5 versus route 2-to-3

The remote store was strong on both matched routes:

```text
route 4->5 change_to_dirty = 2021
route 2->3 change_to_dirty = 1904
```

The retained conclusion was route stability rather than route selectivity. Two route
instances of the same accepted primitive are not automatically two independent maps,
and the runs were not two operations on one persistent carrier.

### Simple route-state axis

The route-state pilot found:

```text
direct_route_moved = false
swapped_route_moved = true
route_state_response = false
ROUTE_STATE_NOT_ESTABLISHED
```

The one moved swapped-route distance did not rescue the failed direct route and control
asymmetry. It cannot supply the second carrier axis.

### Same-core store as inverse

A same-core same-value store was a near-identity control in the retained operator runs.
That does not make it the inverse of a remote ownership transfer. A valid inverse must
reverse the physical map on held-out carrier states in both orders:

```text
A_inverse(A(s)) ~ s
A(A_inverse(s)) ~ s
```

No such law was measured.

### Flush or re-preparation as inverse

Flush, prefault, and relation re-preparation destroy, evict, or replace the physical
state. They remain kill controls. They cannot enter an inverse word.

### Disjoint line sets

Two ownership operations on independent disjoint line sets commute in the declared local
line model. A commutator built only from disjoint support would be structurally trivial,
not a useful Small Wall target.

### Identical operator instances

Two labels, amplitudes, or budgets on the same source core, destination core, route, and
line set form at most one parameterized operation family. Independent names do not create
independent generators.

## 3. Strongest design-only grammar

The only surviving ownership-based grammar uses one persistent carrier, one public home
or reclaim core, two distinct remote receiver cores, and partially overlapping line
sets.

```text
A      = remote same-value ownership transfer by remote_core_A on S_A
A^-1 ? = same-value reclaim by home_core on S_A
B      = remote same-value ownership transfer by remote_core_B on S_B
B^-1 ? = same-value reclaim by home_core on S_B
```

Required support geometry:

```text
|S_A| = |S_B|
0 < |S_A intersection S_B| < |S_A|
S_A != S_B
```

A synthetic fixture uses:

```text
S_A = {0, ..., 15}
S_B = {8, ..., 23}
intersection size = 8
```

The fixture proves only that the design is neither disjoint nor identical. It is not a
physical schedule.

The design intends overlap lines to be the only place where two remote ownership
transfers could fail to commute. This is a hypothesis about a future physical backend,
not a conclusion from retained evidence.

## 4. Why the inverse is still open

A home-core same-value reclaim operation may:

- return ownership to the declared home core;
- create another forward coherence transition;
- fail to restore route, timing, replacement, or coherence metadata;
- restore logical bytes while leaving the physical carrier displaced;
- depend on the immediately preceding owner rather than act as a uniform inverse.

Therefore `A^-1` and `B^-1` remain inverse candidates only. Each must pass left and
right inverse laws over held-out starting states, mappings, delays, amplitudes, and line
set placements before H2.

Wrong direction, wrong core, wrong line set, wrong amplitude, and wrong inverse order
must fail under the same equivalence law.

## 5. Topology remains unresolved

The retained route comparison validated two independent route pairs, not one
three-core or multi-destination operation family on a shared persistent carrier. I2B does
not assign concrete cores to `home_core`, `remote_core_A`, or `remote_core_B`.

Before physical freeze, a topology audit must establish:

```text
both remote cores can act on the same experiment-owned carrier
operator receipts distinguish the acting core and route
line-set overlap is exact and stable
source preparation does not preload the receiver word
state tomography can observe the same carrier after every primitive
```

## 6. Alternate second-generator track

The retained read/store and route-state coordinates failed. A genuinely independent
second generator may require a phase-native or timing-coupled operation rather than a
second cache-line ownership instance.

No such operation is currently frozen. This alternate track remains:

```text
PHASE_OR_TIMING_NATIVE_SECOND_GENERATOR_UNRESOLVED
```

## 7. I2B decision

```text
attacked pair families = 7
design grammars retained = 1
physically supported generator pairs = 0
qualified inverses = 0
H1 pair established = false
H2 inverse laws established = false
```

The next gate is:

```text
I2C_SYNTHETIC_BIDIRECTIONAL_OWNERSHIP_MODEL
```

I2C may implement an abstract finite ownership-state model to determine which controls
can distinguish:

- true inverse transport from new forward transfers;
- partial-overlap noncommutativity from per-line additive effects;
- commutator cancellation from idempotent saturation;
- logical-byte return from full carrier restoration.

A synthetic model may validate the protocol and kill matrix. It cannot establish that
the Family 10h substrate realizes the model.
