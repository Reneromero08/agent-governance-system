# Control And Kill Matrix

Status: `FROZEN_DESIGN_CONSTRAINT`

The successor controls operate on relational structure, not merely source magnitude.

| Control | Frozen Law | Kills |
| --- | --- | --- |
| Label invariance | Swap branch/candidate labels while preserving the complete relation. Preparation and response must remain equivalent. | label leakage, opaque-ID dictionary, candidate truth labels |
| Relation mutation | Change actual relation basis while preserving labels and marginal workload. Response must change in the predeclared direction. | total work, scalar q replay, route/order lookup |
| Geometry null | Replace fold relation with identity or neutral relation. Fold-odd response must disappear. | baseline drift, nonrelational PMU curvature |
| Carrier-off null | Disable the shared physical carrier path. Coupled response must disappear. | receiver-only artifact, software classification artifact |
| Schedule null | Preserve marginal operations while scrambling relational ordering. Path response must follow relation order, not workload. | source order, subcapture order, thermal drift |
| Query-separation violation | Give source access to receiver query. Package must classify custody-invalid, not positive. | source-side query oracle |
| Scalar-oracle adversary | Supply perfect source-side q projections through perfect linear PMU channel. Must not satisfy successor claim. | scalar projection transduction |
| Fixed-work semantic control | Hold work traces, mappings, bank identities, and execution order fixed while changing only relation structure. Contrast must survive. | marginal workload and bank label artifacts |

## Blocked Control Repair

Independent review found this matrix not yet freeze-ready. A future repair must make
the manipulated physical variable explicit. It must not demand "change only relation
structure" while holding every physical degree of freedom fixed.

Required repair:

```text
freeze relation object R
freeze source encoder P(R)
freeze query space Q
freeze observable h(P(R), q)
freeze label action
freeze relation mutation
freeze branch-local marginals
freeze pair-level topology mutation
freeze joint-interaction observable J_q
```

The joint-interaction observable should have this form or an explicitly equivalent
predeclared form:

```text
J_q = Y_q(a,b) - Y_q(a,empty) - Y_q(empty,b) + Y_q(empty,empty)
```

The design remains blocked until scalar, additive, nonlinear, bank, route, timing,
and order models fail against that frozen observable on held-out relation/query data.

## Strong-Signal Mapping And Reversal

Strong-signal pair cells must have:

```text
logical mapping pass
independent bank-resolved physical reversal pass
sign pass
```

Logical mapping and physical reversal must not be duplicate calculations over the
same signed numbers. Bank labels and measured bank assignment are hard-gated.

## Near-Zero Absolute Law

The future near-zero law must:

```text
use a fixed bound frozen before the run
use the same bound for algebraically identical logical and physical pair quantities
not compare one-leg and two-leg quantities to the same ceiling
not derive a held-out ceiling from a same-sized sample maximum
```

Near-zero raw legs may be diagnostic if they are stochastic single-window estimates.
Hard gates must use the estimator they were calibrated for.

## Exact Failure Semantics

A control failure returns:

```text
QUERY_SEPARATED_RELATIONAL_CARRIER_NOT_ESTABLISHED
```

or custody-invalid, depending on the failure.

It never returns:

```text
SMALL_WALL_CROSSED
```
