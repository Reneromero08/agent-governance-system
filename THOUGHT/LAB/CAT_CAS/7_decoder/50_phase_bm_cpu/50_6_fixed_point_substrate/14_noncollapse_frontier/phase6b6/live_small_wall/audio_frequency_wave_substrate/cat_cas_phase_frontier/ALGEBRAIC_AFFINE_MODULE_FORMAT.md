# Width-Parametric Mixed Affine Phase Module

## Status

This mutable successor uses the accepted coefficient-oblivious affine kernel
as the native backend for one public mixed series/parallel module:

```text
R    = COMPOSE(F, G)
I    = INTERSECT(actual R, P)
ROOT = COMPOSE(actual I, K)
project only ROOT
```

The accepted claim is:

```text
BOUNDED_WIDTH_PARAMETRIC_AFFINE_COMPOSITION_AND_INTERSECTION_MODULE_CLOSURE
```

with ceiling:

```text
BOUNDED_WIDTH16_SOFTWARE_GF2_AFFINE_MIXED_MODULE_REFERENCE_ONLY
```

## Typed public geometry

The public nominal signatures are:

```text
F : X -> U
G : U -> B
P : X -> B
K : B -> C
R : X -> B
I : X -> B
ROOT : X -> C
```

`COMPOSE` requires the left codomain to equal the right domain.
`INTERSECT` requires both ordered signatures to match exactly. The output
signature is checked against the operator law before execution; a physically
width-compatible but nominally invalid composition rejects.

Four leaf relations, two resident intermediates, the root, and the surviving
boundary use eight canonical affine blocks. The actual address of `R` is
passed to the intersection descriptor, and the actual address of `I` is
passed to the root composition descriptor. There is no export copy,
controller-side reconstruction, or decoded handoff.

## Native intersection

For `R(X,B) ∩ P(X,B)`, the existing `4w`-row workspace retains its public
column order:

```text
[unused shared columns: w | X: w | B: w | constant]
```

Both input equation systems are concatenated into the external `X,B`
columns. The first `w` public pivot stages see zero columns; their owner rows
remain eligible for the later external stages. The unchanged fixed reduction
then produces canonical RREF rows by `X,B` pivot column. Empty inputs,
contradictory rows, universal relations, duplicates, dependencies, and rank
`2w` outputs use the same resident empty/pivot/activity controls as
composition.

No new workspace or phase-control cells are required. Intersection has the
same native schedule as composition; only the public static coefficient
address map differs.

## Primary and reuse transactions

The width-three primary uses four public affine functions whose intersection
imposes `x0=1`. Its final relation is:

```text
x0 = 1
c0 = 1
c1 = x2
c2 = x1 XOR x2 XOR 1
```

The complete canonical boundary has four active rows and hash:

```text
594d12aa095dab79
```

The actual restored carrier then runs an unrelated transaction whose
intersection imposes `x2=0`:

```text
x2 = 0
c0 = 1
c1 = x0
c2 = x1 XOR 1
```

Its complete boundary hash is:

```text
4057ecbc8bf19d29
```

Both 49-cell boundaries match a separately compiled coefficient-aware GF(2)
reference exactly.

## Restoration law

The accepted in-place path is:

```text
encode F,G,P,K
-> derive resident R
-> derive resident I from actual R and P
-> derive resident ROOT from actual I and K
-> copy and decode only ROOT
-> remove boundary copy
-> recompute and inverse ROOT
-> recompute and inverse I
-> recompute and inverse R
-> remove K,P,G,F
-> verify restoration
-> reuse the same actual carrier
```

Wrong root inverse, missing root inverse, and the prospectively noncommuting
order `I^-1` before `ROOT^-1` each leave one full F3-symbol restoration error
of `1.73205080757`. Independent siblings are not used as a reordered-inverse
failure requirement.

Snapshot reload is a separately labeled weaker baseline. It performs zero
inverse-factor recomputations and copies the working snapshot back; it is not
credited as the accepted restoration path.

## Semantic and algebra controls

Six complete boundaries per width cover:

```text
primary mixed module
unrelated restored-carrier reuse
explicit empty propagation
universal intersection identity
cross-branch contradiction to empty
duplicate intersection idempotence
right-operand highest-column intersection coverage
```

Three altered algebra paths execute and restore their own histories:

```text
bypass intersection with its left input
bypass intersection with its right input
substitute composition for intersection
```

All three change the final complete boundary. This discriminates the native
intersection law from mere routing and from a physically width-compatible
but nominally invalid operator substitution.

The duplicate/idempotence fixture also adds an asymmetric equation only to
the intersection's right operand on `X[w-1],B[w-1]`. It therefore exercises
the right-side encoder at the highest external coefficient columns for every
scaled width.

## Exact resource laws

The inherited canonical relation and workspace sizes are:

```text
B(w) = 4w^2 + 4w + 1
W(w) = 36w^2 + 11w + 2
```

The eight-block mixed module carrier is:

```text
Carrier(w) = 8B(w) + W(w) = 68w^2 + 43w + 10
```

One accepted forward-and-actual-inverse transaction executes:

```text
AND         864w^3 + 675w^2 - 27w
XOR        1728w^3 + 576w^2 - 252w
OR                         72w^2 + 24w + 12
NOT                       147w^2 + 33w + 6
swaps                            144w^2
row adds                   144w^2 - 36w
updates      1296w^3 + 892w^2 - 2w + 40
native reads 1728w^3 + 1235w^2 - 55w + 32
workspace scan passes                           13
workspace cell checks                       13W(w)
boundary decodes                              B(w)
restoration checks                        Carrier(w)
integrity checks                          Carrier(w)
```

At width three:

```text
relation cells                                49
workspace cells                              359
carrier cells                                751
live carrier bytes                        24,032
carrier plus comparison heap bytes         48,064
compiler-measured nested stack chain        8,432
conservative current-ABI accounted bytes   56,496
same-carrier accepted transactions              54
maximum restoration error       1.57009245868e-16
```

At width sixteen:

```text
relation cells                             1,089
dense membership entries          4,294,967,296
workspace cells                           9,394
carrier cells                            18,106
live carrier bytes                      579,392
phase ANDs                             3,711,312
phase XORs                             7,221,312
phase-cell updates                     5,536,776
native reads                           7,393,200
logical carrier cell inspections       7,552,623
compiler-measured nested stack chain      98,320
conservative current-ABI accounted bytes 1,257,104
```

Widths `3,4,8,12,16` match the independent reference for all six complete
boundaries. These are fixed-schedule software operation counts, not optimized
Gaussian elimination and not evidence of performance advantage.

## No-smuggle boundary

Neither resident `R` nor resident `I` is decoded, serialized, hashed,
committed, inspected by a diagnostic aggregate, or copied outside the
carrier. The output contains no intermediate coefficient, rank, empty flag,
pivot pattern, row activity, or carrier displacement. The only decoder is the
accepted final boundary latch.

Strict GCC, analyzer, ASan, UBSan, leak checks, deterministic replay, exact
JSON allowlists, source/result hashes, separate reference linkage, rejected
intermediate/control projection, and mandatory file/network/write/send
tracing pass.

Focused review closed three evidence defects: the compiler stack chain now
includes the separately emitted reduction frame; scaled fixtures exercise the
right intersection operand's highest coefficient columns; and those same
high-index equations are genuinely duplicated through `R` and `P`, preserving
the idempotence claim. Closure review reproduced all 30 complete
phase/reference boundaries and found no remaining finding.

## Claim boundary

This establishes native composition and native intersection on actual
resident canonical affine messages in one bounded public mixed module
topology through width sixteen. It does not establish:

```text
an arbitrary mixed-module parser or compiler
unbounded module depth or topology size
arbitrary graph topology or treewidth
runtime-unbounded width or equation capacity
nonlinear or arbitrary Boolean relation closure
CATVM custody for this mixed module
performance or computational advantage
physical waveform or silicon execution
Small Wall crossing
unlimited catalytic computation
```

The next relational step is a public recursive mixed affine module compiler:
allocate canonical resident messages by topology, execute arbitrary bounded
series/intersection trees in dependency order, reverse them in exact reverse
dependency order, and measure carrier growth against live separator count
rather than hardcoded topology size.
