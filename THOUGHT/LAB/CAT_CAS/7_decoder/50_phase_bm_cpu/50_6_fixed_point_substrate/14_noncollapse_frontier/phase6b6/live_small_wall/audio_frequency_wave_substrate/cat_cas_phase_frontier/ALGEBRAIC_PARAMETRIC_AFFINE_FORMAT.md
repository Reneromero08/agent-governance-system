# Width-Parametric General Affine Phase Closure

## Status

This mutable successor extends the fixed-width-two calibration to compile-time
interface widths:

```text
w = 2, 3, 4, 8, 12, 16
```

The accepted claim is:

```text
BOUNDED_WIDTH_PARAMETRIC_COEFFICIENT_OBLIVIOUS_GENERAL_AFFINE_PHASE_RELATION_COMPOSITION
```

with ceiling:

```text
BOUNDED_WIDTH16_SOFTWARE_GF2_AFFINE_RELATION_SYSTEM_REFERENCE_ONLY
```

## Complete affine relation family

Every affine relation between two `w`-bit ports is an affine subset of
`F2^(2w)` and therefore has rank at most `2w`. The canonical descriptor has
`2w` equation slots:

```text
[active, 2w coefficients, constant]
```

plus an explicit empty flag. Its exact size is:

```text
B(w) = 2w(2w+2)+1 = (2w+1)^2 = 4w^2+4w+1
```

Universal is all zero. Empty has all equation rows zero and only the final
flag set. Nonempty output rows are RREF rows stored by external pivot column,
so the descriptor is independent of input row order and duplicates.

At width two, `B(2)=25` is larger than the 16-entry dense membership vector.
At width three, `B(3)=49` is already smaller than 64 dense entries. At the
largest accepted width:

```text
w                              16
complete affine coefficients 1,089
dense membership entries     2^32
```

This is compact closure for the affine relation family, not a compact encoding
of arbitrary Boolean relations.

## Fixed phase projection

Composing `F(X,Y)` and `G(Y,Z)` uses:

```text
joint rows                 M = 4w
augmented columns          Q = 3w+1
public reduction stages    L = 3w
column order               Y, X, Z
```

Every coefficient column owns one public pivot row. A successful owner is
excluded from later selection by its resident phase pivot flag. A failed
owner remains eligible and can be swapped into a later owner. Because
`L=3w < M=4w`, every possible coefficient pivot has a public owner.

All `M` pivot candidates and all non-owner row additions execute at every
stage. Selection, active-row, pivot, elimination, input-empty, and
inconsistency controls stay as Boolean-subset F3 phase symbols. Coefficients,
rank, and pivot positions do not select a host pivot, loop bound, memory
address, or native-operation count.

After the `Y` stages, every remaining active non-owner row has zero shared
coefficients. The `X,Z` stages reduce exactly that projected row space. The
external pivot for column `d` remains in public row `w+d`, which supplies
canonical output slot `d`.

## Exact resource laws

The resident joint matrix and controls use:

```text
D(w)   = 4w(3w+1)             = 12w^2+4w
Ctl(w) = 4w + 3w + 2(3w)(4w) + 2
       = 24w^2+7w+2
W(w)   = D(w)+Ctl(w)          = 36w^2+11w+2
```

The existing chained transaction retains `F,G,K,H,Z,boundary`, so:

```text
Carrier(w) = 6B(w)+W(w) = 60w^2+35w+8
```

The carrier implementation stores baseline and working complex doubles:

```text
live carrier bytes       = 32*Carrier(w)
carrier+snapshot subtotal = 64*Carrier(w)
```

For one accepted child/parent forward-and-inverse transaction, exact native
counters are:

```text
AND         576w^3 + 450w^2 - 18w
XOR        1152w^3 + 384w^2 - 168w
OR                         48w^2 + 16w + 8
NOT                        98w^2 + 22w + 4
swaps                              96w^2
row adds                     96w^2 - 24w
updates                    864w^3 + 600w^2 + 4w + 28
native kernel reads       1152w^3 + 826w^2 - 34w + 22
boundary decodes                                 B(w)
workspace scan passes                               9
workspace cell checks                           9W(w)
restoration cell checks                       Carrier(w)
integrity cell checks                         Carrier(w)
```

These are fixed-schedule operation counts, not optimized Gaussian-elimination
costs and not evidence of performance advantage. Snapshot creation separately
copies `2*Carrier(w)` complex values (baseline and working state); snapshot
reload copies another `Carrier(w)` working values only on the weaker snapshot
path. These copy counts are not hidden inside the native-kernel read counter.

## Scaling qualification

Each accepted width compiles the phase engine and a separate conventional
rank-pointer GF(2) reference. Nine complete boundaries match exactly at every
width, covering:

```text
rank-deficient many-to-many relation
unrelated reuse relation
input row permutation
duplicate equations
rank-zero universal
rank-one universal
explicit input empty
two simultaneous contradiction rows
maximum-rank 2w-row output
```

No assignment, relation tuple, candidate, or witness expansion is used.

The largest accepted transaction reports:

```text
width                                      16
equation capacity                           32
relation cells                           1,089
workspace cells                          9,394
carrier cells                           15,928
live carrier bytes                     509,696
phase ANDs                            2,474,208
phase XORs                            4,814,208
phase-cell updates                    3,692,636
native kernel reads                   4,929,526
logical carrier cell inspections      5,047,017
maximum restoration error  2.00148302124e-16
predeclared tolerance                      2e-12
```

GCC `-O2 -fstack-usage` reports a 105,840-byte conservative nested call chain
for the width-16 qualification process. Carrier plus comparison heap is
1,019,392 bytes, giving a conservative current-ABI total of 1,125,232 bytes.
The stack contains the qualification suite's coexisting programs and results;
it is not a minimal transaction-stack claim.

## Catalytic transaction and controls

At every width:

```text
H = PROJECT_COMPOSE(F,G)
Z = PROJECT_COMPOSE(actual H,K)
decode only complete final Z
remove the boundary using actual Z
recompute and inverse parent
recompute and inverse child
remove public inputs
verify restoration
reuse the actual restored carrier
```

Wrong, missing, and applicable reordered inverses fail. Snapshot reload
remains a separately labeled weaker baseline and performs zero inverse-factor
recomputations. Intermediate projection, phase-control projection, null
carrier, and unknown commands reject.

The phase source retains one decoder in the final boundary latch. No
pre-inverse aggregate of the carrier is calculated or emitted; in particular,
the earlier whole-carrier displacement diagnostic was removed because it
could reveal a function of the resident intermediate. Mandatory syscall
tracing covers file, network, write, writev, pwrite, and send channels and
finds no extra output path. The conventional reference is not linked into the
phase executable.

Focused review independently matched all nine boundaries at every integer
width from two through sixteen and closed four findings. It removed the
pre-inverse displacement leak, separated native kernel reads from all
verification/projection scans and snapshot copies, accounted for all nine
coexisting semantic programs, and repaired the stack-chain matcher so the
`ga_apply_compose` frame is included. Closure review found no remaining
finding within the bounded software claim.

## Claim boundary

This establishes complete polynomial descriptor closure and polynomial
carrier storage for general affine GF(2) relations through width 16. It does
not establish:

```text
runtime-unbounded width or equation capacity
nonlinear or arbitrary Boolean relation closure
arbitrary graph topology or treewidth
automatic elimination-order discovery
CATVM custody for this wider affine kernel
performance or computational advantage
physical waveform or silicon execution
Small Wall crossing
unlimited catalytic computation
```

The next relational step is compact recursive geometry over these wider
affine interfaces: compose public module trees or bounded-treewidth graphs
while retaining canonical affine messages, actual inverse restoration, and
restored-carrier reuse.
