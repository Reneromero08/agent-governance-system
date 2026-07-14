# Scalar Replay Adversary Spec

Status: `ADVERSARY_DEFINED__NOT_YET_DEFEATED`

## Scope

This adversary is not a synonym for every conventional explanation. It is a suite of
ordinary old-boundary predictors with explicit inputs, capacities, metrics, and
held-out tests.

## Inputs

The adversary receives:

```text
all source-visible inputs
all source receipts
public schedule skeleton
public carrier layout
public source binary
public receiver binary
source operation counts
marginal branch traces
total work
bank/address/route metadata
order and timing
public relation metadata
realized receiver query at scoring time
```

## Families

```text
linear marginal model
additive branch model
nonlinear marginal-workload model
finite response table
bank/route/order lookup
timing model
bounded-capacity answer cache
```

## Unbounded Lookup Table

An arbitrary table indexed by `(R, q)` is observationally indistinguishable on a
closed finite set. Finite evidence cannot reject it. The claim can only be made
against the frozen bounded class.

## Bounded Answer Cache

For a cache that can store at most `k` query responses:

```text
pass on closed training set if |Q_train| <= k
fail held-out query if q_heldout not stored
fail capacity gate if |Q_total| * response_bits > C_prep
```

The current package cannot run the capacity gate because `C_prep` is not measured.

## Rejection Rule

A future relational-carrier result fails if any permitted old-boundary family predicts
the held-out response within the frozen tolerance. The realized query must always be
available to the adversary at scoring time.

## Current Result

The adversary is specified but not defeated. `SCALAR-REPLAY-01` remains unresolved.

