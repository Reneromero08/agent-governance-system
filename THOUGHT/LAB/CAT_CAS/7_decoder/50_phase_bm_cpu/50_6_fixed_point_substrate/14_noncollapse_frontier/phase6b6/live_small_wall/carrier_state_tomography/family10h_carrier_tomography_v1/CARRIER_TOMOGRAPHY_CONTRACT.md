# Public Family 10h Carrier-State Tomography Contract

Status: `FAMILY10H_CARRIER_TOMOGRAPHY_PACKAGE_FROZEN_AWAITING_AUTHORIZATION`

Science package id: `family10h_carrier_tomography_v1_0`

Transaction/run id: `family10h_carrier_tomography_v1_0`

This is a public substrate-characterization package. It does not claim an
unresolved access path, a private relation, physical memory, borrowing, or a Small
Wall crossing.

## Claim Ceiling

Allowed eventual result classes:

```text
FAMILY10H_POST_SOURCE_STATE_OBSERVED
FAMILY10H_POST_SOURCE_STATE_NOT_OBSERVED
FAMILY10H_CARRIER_TOMOGRAPHY_CANDIDATE
FAMILY10H_CARRIER_TOMOGRAPHY_CUSTODY_INVALID
```

Forbidden classes:

```text
ORBITSTATE_ACCESS_ESTABLISHED
RELATIONAL_CARRIER_ESTABLISHED
PHYSICAL_RELATIONAL_MEMORY_ESTABLISHED
CATALYTIC_BORROWING_ESTABLISHED
SMALL_WALL_CROSSED
```

Even a perfect run establishes only a route-scoped public carrier-state model.
No answer-cache exclusion, query-separated identifiability repair, restoration-R2
closure, or wall promotion follows from this package.

## Causal Sequence

The frozen sequence is:

```text
public preparation selected
source process prepares carrier
source process exits
parent verifies source death with waitpid
all source IPC and descriptors close
fresh receiver query is selected
receiver applies standardized query
receiver measures public observables
public feature packet freezes
analysis reconstructs state/operator
```

The receiver query occurs only after source death. Every state/query/delay cell
uses a fresh preparation because the query may disturb the carrier.

## Public Preparation Grammar

Base work:

```text
M = 2048
total balanced work = 4096
q in {-1536, -1024, -512, 0, 512, 1024, 1536}
```

For each public codeword:

```text
bank_A_work = M + q
bank_B_work = M - q
```

The source-off control sends the same total work to dedicated dummy storage:

```text
bank_A_work = 0
bank_B_work = 0
dummy_work = 4096
```

All preparations are public carrier codewords. There is no private map, hidden
relation, fold branch, target vector, or private adjudication.

## Physical Freeze

The package freezes:

```text
page size = 4096 bytes
pages per lane = 64
line size = 64 bytes
lines per lane = 4096
affine line permutation = (73 * line_index + 19) mod 4096
source core = 4
receiver core = 5
physical mappings = map0, map1
source orders = A_then_B, B_then_A
component order = A, B, dummy, sham
allocation lifetime = before source fork through evidence sealing
dummy storage = dedicated owned storage, never A/B/sham
```

The generated schedule binds the exact layout identity for every tuple.

## Delay Grid

The frozen delay grid is:

```text
0ns
100us
1ms
10ms
100ms
```

Any future live execution must use this grid exactly unless a new package is
frozen before acquisition.

## Receiver Query Family

The public query family is:

```text
query_A
query_B
query_A_then_B
query_B_then_A
query_sham
carrier_off
```

Each query binds the instruction sequence, address sequence, PMU group, query
order, and disturbance ceiling in `CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.json`.
No query is described as topological or relational.

## Measured Observable Vector

Every raw record must include:

```text
Change-to-Dirty
dirty probe response
CPU cycles
duration
time_enabled
time_running
event IDs
PMU read size
receiver CPU before/after
```

Every record also binds:

```text
preparation codeword
mapping
query
delay
replicate
session
source-death receipt
bank identity
address-layout identity
source order
query order
temperature
process custody
policy custody
```

PMU counts are observations, not persistent state variables, until held-out
analysis establishes predictive sufficiency.

## Schedule and Coverage

The schedule is an exact machine-readable tuple multiset and one exact global
execution sequence. Analysis must require:

```text
exact tuple equality
exact cardinality and multiplicity
no missing rows
no extra rows
no duplicate IDs
exact executed order
complete source-death receipts
complete PMU custody
complete source/receiver CPU custody
```

Classification must not iterate observed keys to define the expected set.

Frozen tuple count: `8320`.

The two frozen sessions are:

```text
session_0
session_1
```

## Persistence Matrix

The primary matrix is:

```text
Y[preparation, query, delay, mapping, replicate]
```

The primary question is whether receiver response after source death retains
reproducible information about public preparation. Required tests include:

```text
source-off null
q-sign reversal
q-magnitude ordering
mapping crossover
cross-replicate consistency
delay-dependent persistence or decay
query-order characterization
```

No aggregate rescues a failed replicate.

## Operational Distinguishability

The package reports only operational lower bounds:

```text
cross-validated codeword classification
confusion matrix
held-out replicate prediction
held-out mapping prediction
held-out delay prediction
response-matrix effective rank
singular/eigen spectrum
between-state versus within-state distance
```

These statements are required:

```text
observed distinguishable codewords != total physical preparation capacity
effective response rank != hidden microstate dimension
```

## Operator Identification

The operator ladder is:

```text
S0: scalar preparation amplitude only
S1: amplitude + query + delay + mapping
S2: interactions among preparation, query, delay, mapping, and order
```

The smallest sufficient model is selected only by held-out prediction. In-sample
fit alone cannot support a result class.

Held-out prediction is required in three forms:

```text
held-out replicate
held-out mapping
held-out delay
```

The result adjudicator is exclusive and fail-closed:

```text
invalid packet or custody failure -> FAMILY10H_CARRIER_TOMOGRAPHY_CUSTODY_INVALID
valid packet, source-off null, and no response above the public floor -> FAMILY10H_POST_SOURCE_STATE_NOT_OBSERVED
valid packet, source-off null, q-sign reversal, and q-magnitude ordering -> FAMILY10H_POST_SOURCE_STATE_OBSERVED
valid packet that is neither observed nor not-observed -> FAMILY10H_CARRIER_TOMOGRAPHY_CANDIDATE
```

## Joint-Interaction Characterization

The public factorial block freezes:

```text
both active
A active / B dummy-matched
A dummy-matched / B active
both dummy-matched
```

Every arm must preserve total work, source-loop count, timing envelope, address
population size, query schedule, and receiver work.

The nonadditivity observable is:

```text
J_q = Y_q(A,B) - Y_q(A,dummy) - Y_q(dummy,B) + Y_q(dummy,dummy)
```

`J_q != 0` establishes only observed nonadditivity under matched public arms. It
does not establish unresolved relation, physical memory, or a nonclassical
mechanism. Ordinary nonlinear controls include cache contention, shared-bank
saturation, route interaction, order interaction, and measurement nonlinearity.

## State Lifetime

For every preparation/query pair, the analysis reports response at each delay,
decay or persistence curve, confidence interval, session variation, and mapping
variation. The only allowed lifetime vocabulary is:

```text
vanishes before source death
survives only immediate handoff
survives a bounded delay
persists across the full grid
changes form across delay
```

## Restoration Boundary

This package does not adjudicate R2 restoration. It collects:

```text
baseline public state vector
post-preparation displaced state
post-query state
post-reset/rebaseline state
time-matched natural-relaxation control
```

Allowed statement:

```text
R0 byte/hash return observed or not observed
R2 not adjudicated
```

No R2 equivalence threshold may be retrofitted after acquisition.

## Live Custody Inheritance

Before any future hardware execution:

```text
clean main
HEAD = origin/main = commit authority
exact manifest file authority
exact source-file key/hash/size equality
deterministic source-bundle equality
compile from immutable snapshot
target output root absent
local evidence root absent
coherent nested timeouts
strict platform identity
strict readable policy fields
strict temperature
strict process custody
```

Target results remain provisional until controller copy-back and evidence
verification close. There is no automatic retry.

## Offline Artifacts

The freeze artifacts are:

```text
CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.json
CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.tsv
CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.sha256
CARRIER_TOMOGRAPHY_SELF_TEST.json
CARRIER_TOMOGRAPHY_RUNTIME_SELF_TEST.json
CARRIER_TOMOGRAPHY_TARGET_SELF_TEST.json
CARRIER_TOMOGRAPHY_CONTROLLER_SELF_TEST.json
CARRIER_TOMOGRAPHY_OFFLINE_VALIDATE.json
CARRIER_TOMOGRAPHY_TRANSPORT_SIMULATION.json
CARRIER_TOMOGRAPHY_DEPLOYMENT_LAYOUT_SELF_TEST.json
CARRIER_TOMOGRAPHY_IMPLEMENTATION_MANIFEST.json
CARRIER_TOMOGRAPHY_IMPLEMENTATION_MANIFEST.sha256
CARRIER_TOMOGRAPHY_SOURCE_HASHES.json
CARRIER_TOMOGRAPHY_SOURCE_BUNDLE.tar.gz
```

No live execution is authorized by this contract.
