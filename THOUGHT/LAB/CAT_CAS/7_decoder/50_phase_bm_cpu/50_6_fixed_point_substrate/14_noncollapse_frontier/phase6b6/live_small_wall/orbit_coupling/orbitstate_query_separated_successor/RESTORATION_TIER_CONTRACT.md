# Restoration Tier Contract

Status: `FROZEN_DESIGN_CONSTRAINT`

Do not call byte equality physical restoration.

## Tiers

```text
R0 = bytes/hash return
R1 = measured-output return
R2 = accepted observable-state equivalence
R3 = multi-instrument carrier return
R4 = closure up to predeclared invariant
```

The successor targets at least:

```text
R2
```

before any physical relational-memory or Small Wall claim.

## Required Restoration Specification

Every implementation contract must freeze:

```text
baseline physical observables
forward/preparation observables
terminal observables
restoration operation
post-restoration observables
metric
uncertainty/tolerance
no-restoration control
wrong-inverse control
reordered-inverse control
carrier-off restoration control
```

## R2 Observable-State Equivalence

R2 requires an accepted measured state representation, not only bytes. The state may
include:

```text
Change-to-Dirty counts
dirty-probe counts
duration or cycle measurements
bank-resolved response
route/core identifiers
predeclared uncertainty estimates
```

The exact state representation must be frozen before live evidence.

R2 is two-sided:

```text
forward displacement from baseline must be demonstrated
post-restoration equivalence to baseline must be demonstrated
```

An insensitive observable, natural relaxation, global reset, destructive probe, or
carrier-off equality cannot count as restoration.

## Required Future R2 Contract

The future implementation must freeze:

```text
state vector
standardized probe operation
measurement disturbance law
baseline distribution
terminal displacement threshold
restored equivalence region
confidence rule
multiplicity correction
time-matched no-op control
natural-relaxation control
wrong-inverse control
reordered-inverse control
carrier-off restoration control
```

## Restoration Metric

The restoration metric must use independent physical observables:

```text
physical_restoration_distance(baseline_state, restored_state)
```

It must define:

```text
units
domain
denominator
zero behavior
threshold source
which observable it gates
```

No threshold may be fitted after live evidence.

## Controls

No-restoration control: forward path without restoration must fail the R2 gate.

Wrong-inverse control: an inverse for a different relation or route must fail.

Reordered-inverse control: path order changes must fail when order is part of the
claimed carrier.

Carrier-off restoration control: carrier-off must not produce a false restoration
claim.
