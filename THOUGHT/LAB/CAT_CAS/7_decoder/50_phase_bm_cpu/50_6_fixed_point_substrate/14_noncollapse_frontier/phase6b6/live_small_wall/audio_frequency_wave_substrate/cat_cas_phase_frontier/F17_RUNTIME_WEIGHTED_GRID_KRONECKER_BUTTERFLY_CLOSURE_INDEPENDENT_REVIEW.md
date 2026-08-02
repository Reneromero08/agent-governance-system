# M120 independent review

## Decision

```text
classification       INDEPENDENTLY_VERIFIED_STRICT_SCOPE
verification level   INDEPENDENT_ORACLE_REEXECUTION
factor restoration   EXACT_ALGEBRAIC_RESTORATION
transient buffers     NO_RESTORATION_CLAIM
```

The reviewed package was reexecuted after the focused control repair. The
production program, independent oracle, and benchmark all exited successfully.
Every `n=2,3,4` MSB-indexed resident butterfly interface was also compared
against an explicit dense source/target contraction on an independently chosen
nontrivial exact vector; all values agreed exactly.

## Verified strict scope

For the two deterministic public runtime-weight families on binary square
grids `n=2,3,4`, the actual resident vertical factor has the checked form

```text
K(J) = [[1, 1], [1, zeta17^J]].
```

The production path reads the actual resident `zeta17^J` cell, checks the
other three resident unit cells, and applies the MSB-indexed Kronecker
butterfly without a `2^n x 2^n` matrix or source/target enumeration. The exact
kernel counts across all row interfaces are:

```text
n                         2    3     4
nontrivial root actions   4   24    96
field additions           8   48   192
```

The repair preserves the six frozen M119 boundaries. The independent
16-integer power-basis oracle reconstructs them from a Gray-code global
character histogram. Explicit matrices over `F103` reproduce interface ranks
`4,8,16`; setting one separator weight to zero halves each rank. Only the
final scalar receives a full-basis lift.

The actual borrowed factor carrier is reversed and seed-unloaded to its
original zero backing, then reused on the same backing by the held-out family.
This supports `EXACT_ALGEBRAIC_RESTORATION` only for that factor carrier. The
butterfly frontiers are transient projection work buffers, are not retained,
and carry `NO_RESTORATION_CLAIM`. A resident interface bank and its inverse
would be required before claiming catalytic custody or restoration of the
frontier itself.

## Controls and accounting

The hard gate covers premature projection, missing and wrong inverse,
applicably reordered inverse, resident mutation, null carrier, wrong family,
wrong topology fingerprint, public-plan answer exclusion, omitted butterfly
stage, resident vertical-factor mutation, zero-weight rank halving, false rank
cap, snapshot absence, exact restoration, and fresh-versus-restored reuse.
Butterfly column stages commute, so failure under their reordering is not
required.

The package reports the actual factor cells, `2^n` frontier width, conservative
alias-inclusive live cell and payload counts, coordinate height, topology,
runtime descriptor, root table, projection, restoration, and reuse. Python
object allocator, bigint internals, and native-library memory remain excluded;
benchmark RSS is process-wide. The resident-factor and public-descriptor paths
share the identical interface recurrence, but their row-diagonal generation is
not operation matched. Timing is observational and supports no advantage
claim.

## Claim ceiling

This is a bounded Linux direct-process repair for `n=2,3,4`. It reduces the
accepted interface application from M119's `(n-1)4^n` source/target
transitions to `(n-1)n2^(n-1)` nontrivial root actions while retaining the
full `2^n` exact interface rank and message bank. The identical compact
classical butterfly has the same recurrence. The evaluated comparison set is
not proved exhaustive or Pareto optimal.

The result does not establish a separator quotient below `2^n`, a distinct
phase resource, computational advantage, Small Wall crossing, CATVM custody,
catalytic inference, physical waveform or silicon execution, replacement of
physical bits with pi, or unbounded computation.
