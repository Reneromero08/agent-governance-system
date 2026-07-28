# Rematerialized Stokes BCH Phase Format

## Scope

This bounded software experiment composes two noncommuting normalized
quadratic Stokes Hamiltonians through the exact signature

```text
log(exp(A) exp(B)).
```

The tensor logarithm is public topology. Its homogeneous primitive components
are converted to right-nested Poisson brackets by the
Dynkin-Specht-Wever projection. No coefficient answer table is supplied to
the phase backend.

## Native state and scheduling

Each modular coefficient is held as its complete unit-phase character orbit
over `F17` and `F19`. Coefficient addition is componentwise phase
multiplication; public nonzero scalar action is character-index permutation.

For each public Lie word, the backend:

```text
seal final generator
-> build the right-nested bracket in reusable scratch blocks
-> add its weighted character orbit to the declared BCH grade
-> apply the actual inverse bracket chain
-> unseal the generator
```

The inverse transaction repeats this public rematerialization in reverse and
subtracts the actual resident contribution. At word grade six, at most six
scratch blocks are live and no per-word value or inverse-history tape is
retained.

## Boundary and controls

Only the complete grade-one-through-six coefficient signature is final.
Scratch projection is denied. A non-emitting verifier compares every final
dual-prime coefficient against an independent exact rational construction.

Missing and wrong inverses must leave a residual. Swapping the noncommuting
modules must change the boundary. Reordering the final component
subtractions is not an applicable failure control because those additions
commute; the intra-word bracket dependency order is fixed by topology.
A snapshot sham executes the same primary forward transaction, reloads a
blank carrier image, and then runs the reuse program. Its creation and reload
traffic are counted separately, and it cannot mint a restoration receipt.
An explicit null-carrier path fails closed.

## Claim boundary

The accepted claim is bounded to normalized two-module Stokes software,
phase execution through BCH word grade six, and an exact rational diagnostic
through grade ten. It does not establish fixed-rank closure, an arbitrary
representation lower bound, unbounded rank growth, a distinct phase
resource, advantage, Small Wall crossing, or physical waveform execution.
