# Physical Carrier Architectures

Status: `SELECTED_WITH_REJECTIONS`

The design considered three materially different carriers. The selected route is not
chosen because it is easiest to code; it is chosen because it best preserves one
unresolved relational process-object while giving fast no-smuggle falsification.

## Candidate A: Dual-Axis Balanced State

Mechanism: source prepares two balanced public carrier coordinates:

```text
I = relation cosine-like coordinate
Q = relation sine-like coordinate
```

Physical carrier: paired Change-to-Dirty ownership-intent windows or equivalent
balanced carrier lanes.

State variables:

```text
I coordinate
Q coordinate
carrier gain controls
restoration observables
```

Source-visible inputs:

```text
public relation object
public carrier layout
preparation seed
```

Receiver-visible inputs:

```text
post-source query basis
query orientation
PMU/timing windows
```

Query operator: receiver selects axis and sign after source closure.

Observable: gain-normalized vector response.

Expected invariant: relation-basis response changes under true relation mutation and
is invariant under branch-label swap.

Restoration law: at least R2 observable-state equivalence for both axes.

Ordinary scalar explanation: high risk. If source computes axis projections, scalar
replay reconstructs the claimed response.

No-smuggle boundary: source may not receive query axis or projection phase; receiver
may not see source projection formulas.

Hardware feasibility: high.

Required controls: label invariance, relation mutation, geometry null, carrier-off,
scalar-oracle adversary.

Fatal blocker: rejected as selected architecture because it can easily collapse into
sender-authored `q_theta` projections unless the implementation proves the axes are
not precomputed scalar answers.

## Candidate B: Shared Pair Topology

Mechanism: both fold branches load one shared physical topology or ownership relation.
The carrier preparation is symmetric under label swap. The receiver later applies a
public basis/query operation that exposes a fold-odd observable if and only if the
relation survived as a physical process-object.

Physical carrier: shared Family 10h ownership-intent topology over paired cache-line
sets, coherence-state paths, or an equivalent carrier where both branches contribute
to the same physical structure.

State variables:

```text
unordered relation pair
shared carrier topology id
branch-symmetric preparation trace
receiver query basis
bank-resolved PMU/timing response
R2 restoration observables
```

Source-visible inputs:

```text
public unordered relation
carrier topology
preparation seed
relation-basis mutation id
```

Receiver-visible inputs:

```text
post-source query seed
query basis/operator
public schedule skeleton
carrier measurement windows
```

Query operator: receiver applies a public basis operation after source closure,
for example basis-specific ownership-intent probe, bank-resolved reversal, or topology
closure query.

Observable: predeclared fold-odd response that is null under identity relation,
carrier-off, and scalar-oracle replay.

Expected invariant:

```text
label swap = null
relation mutation = predicted non-null contrast
geometry null = fold-odd response disappears
carrier-off = response disappears
fixed-work semantic control = contrast survives
```

Restoration law: target at least R2 using independent measured carrier observables.

Ordinary scalar explanation: rejected only if scalar replay from all source-visible
inputs and receipts cannot reconstruct query-basis response.

No-smuggle boundary: source never sees query; receiver never sees private labels,
source work coordinates, or expected geometry before feature freeze.

Hardware feasibility: medium-high. It reuses carrier families already observed, but
requires stricter topology, order, and bank-resolved evidence than the retired
package.

Required controls:

```text
label invariance
relation mutation
geometry null
carrier-off null
schedule null
query-separation violation
scalar-oracle adversary
fixed-work semantic control
```

Fatal blockers found by independent design review:

```text
physical state encoding is not implementable from this draft
nonseparability criterion is not frozen
finite-query answer-vector replay remains admissible
R2 equivalence law is a checklist, not a concrete contract
control orthogonality is underdefined
receiver capability boundary is not yet frozen at OS level
```

Selection: provisional strongest architecture only. It is not frozen for
implementation.

## Candidate C: Path Or Holonomy Carrier

Mechanism: relational object prepares a path-dependent physical state or sequence.
Receiver later selects a closure/query path. The invariant depends on relational
ordering or composition history rather than marginal source work.

Physical carrier: path-dependent coherence, timing, or multi-window PMU state with a
declared closure operation.

State variables:

```text
ordered path history
composition law
closure path
terminal carrier state
restored carrier state
holonomy candidate
```

Source-visible inputs:

```text
public relation path
preparation seed
carrier route
```

Receiver-visible inputs:

```text
closure query selected after source closure
terminal and restored measurements
```

Query operator: receiver selects closure path or basis after source closure.

Observable: path-order-sensitive invariant.

Expected invariant: reorder changes response; wrong inverse fails; correct closure
restores up to predeclared invariant.

Restoration law: R4 would eventually be required for holonomic memory.

Ordinary scalar explanation: lower scalar replay risk, but higher drift and
restoration false-positive risk.

No-smuggle boundary: source cannot encode the closure answer; receiver cannot see
path truth labels before feature freeze.

Hardware feasibility: medium-low for the next test because R2/R4 restoration must be
established first.

Required controls: wrong inverse, reordered inverse, carrier-off, schedule null,
route/session survival.

Fatal blocker: rejected as first successor because it depends on stronger restoration
qualification than currently established. It remains a downstream route after R2
closure is accepted.

## Selection

Provisional strongest architecture:

```text
SHARED_PAIR_TOPOLOGY_QUERY_SEPARATED_CARRIER
```

This route gives the fastest plausible falsification path for:

```text
relation structure rather than labels
receiver query rather than source projection
shared topology rather than scalar work
R2 carrier restoration rather than byte equality
```

It remains blocked until a repair package freezes:

```text
exact preparation map P(R)
query family Q generated unpredictably after source closure
observable h(P(R), q)
joint-interaction observable J_q
nonseparability gate against f(a)+g(N-a)
all-query answer-cache adversary
crossed relation/label/query/mapping/bank/route/order controls
machine-readable tuple multiset and execution sequence
```
