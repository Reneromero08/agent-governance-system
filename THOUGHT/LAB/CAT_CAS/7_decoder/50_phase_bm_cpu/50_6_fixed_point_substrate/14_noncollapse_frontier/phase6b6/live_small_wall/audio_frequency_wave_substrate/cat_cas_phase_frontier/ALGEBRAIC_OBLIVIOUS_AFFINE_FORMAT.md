# Coefficient-Oblivious General Affine Relation Closure

## Status

This mutable checkpoint establishes:

```text
BOUNDED_WIDTH2_COEFFICIENT_OBLIVIOUS_GENERAL_AFFINE_PHASE_RELATION_COMPOSITION
```

for complete GF(2) affine equation relations over two-bit typed interfaces.
The claim ceiling is:

```text
FIXED_WIDTH2_SOFTWARE_GF2_AFFINE_RELATION_SYSTEM_REFERENCE_ONLY
```

## Complete relation format

A nonempty relation between two two-bit ports is an affine subset of
`F2^2 x F2^2`. It has rank at most four and is stored in four fixed equation
slots:

```text
[active, left0, left1, right0, right1, constant]
```

An active row denotes:

```text
left_coefficient * left XOR right_coefficient * right = constant
```

The 24 row cells plus one empty flag form a complete 25-cell descriptor.
Inactive rows are zero. The canonical output is reduced with one public row
slot per external pivot column. Universal is 25 zeros; empty has zero rows
and only the final flag set.

This descriptor is polynomial in a generalized width, but at width two its
25 cells exceed the 16 entries of a dense membership table. No concrete
storage advantage is claimed for this fixed-width calibration.

## Native existential projection

To compose relations `F(X,Y)` and `G(Y,Z)`, the phase engine embeds their
equations into a fixed eight-row, seven-column augmented system ordered as:

```text
Y0, Y1, X0, X1, Z0, Z1, constant
```

It executes six fixed Gauss-Jordan stages in that public column order.
Selection, pivot, row-active, elimination, input-empty, and inconsistency
values remain Boolean-subset F3 phase symbols. Loop bounds and addresses
depend only on the public capacities, never on coefficient values, pivot
positions, rank, emptiness, or inconsistency.

The native Boolean phase laws are:

```text
AND(a,b) = symbol_product(a,b)
XOR(a,b) = lock(a*b*symbol_product(a,b))
OR(a,b)  = lock(a*b*symbol_product(a,b)^2)
NOT(a)   = lock(omega*conj(a))
```

Every scheduled pivot candidate performs a controlled phase swap. Every
scheduled non-pivot row performs a controlled row addition. The accepted
operation topology is identical for rank-zero, rank-one, full-rank,
duplicated, permuted, input-empty, and composed-empty fixtures.

After the two shared `Y` columns are eliminated, the remaining four stages
produce the canonical external relation. No `Y` coefficient, pivot choice,
rank value, witness, assignment, or candidate set is decoded.

## Resident chained transaction

The 318-cell carrier layout contains six 25-cell relation blocks and 168
workspace cells:

```text
F, G, K                  75 cells
actual resident H        25 cells
actual resident Z        25 cells
public boundary          25 cells
joint augmented matrix   56 cells
resident controls       112 cells
```

The transaction is:

```text
H = PROJECT_COMPOSE(F,G)
Z = PROJECT_COMPOSE(actual H,K)
copy actual Z to the final boundary
decode only the 25 final coefficients
remove the boundary using actual Z
recompute and inverse the parent from actual H and K
recompute and inverse the child from actual F and G
remove K, G, and F
verify restoration
reuse the same restored carrier for an unrelated program
```

Each composition reverses its fixed row operations using its actual resident
phase controls and clears the 168-cell workspace before returning. Inverse
composition recomputes the output factor from the actual resident operands;
it does not retain an output copy or inverse-factor table.

## Discriminating semantics

The primary first composition is:

```text
F: x0 XOR y0 XOR y1 = 0
G: y0 XOR y1 XOR z0 = 1
H: x0 XOR z0 = 1
```

Every admissible `(X,Z)` pair leaves two unresolved `Y` witnesses. No witness
is selected or counted during execution. The parent relation makes the final
boundary:

```text
x0 XOR w0 XOR w1 = 1
```

It contains eight of the sixteen possible `(X,W)` pairs and is many-to-many:
each `X` has two `W` values and each `W` has two `X` values.

The accepted semantic suite also includes:

```text
late versus early pivot row permutation
duplicated equations
rank-zero universal projection
rank-one universal projection
explicit input empty
two simultaneous contradiction rows
full-rank four-equation relation
unrelated restored-carrier reuse
```

The multi-contradiction case specifically distinguishes Boolean OR from an
incorrect XOR accumulation of inconsistency flags.

## Restoration and controls

One carrier executes 73 actual-inverse transactions with maximum restoration
error:

```text
2.00148302124e-16
```

The predeclared complex-state tolerance is `2e-12`. Wrong parent inverse,
missing parent inverse, and applicable child-before-parent inversion each
leave `1.73205080757` error. Intermediate projection, control projection,
null-carrier execution, and unknown commands reject before ordinary output.
Snapshot reload is separately labeled and is not credited as actual inverse.

## No-smuggle boundary

The phase source contains one `decode_root` call, inside the final boundary
latch. The intermediate `H`, augmented system, pivot controls, and elimination
controls are neither decoded nor serialized. The phase and conventional GF(2)
references are separate binaries. Neither source materializes relation truth
tables, assignment expansions, witnesses, or candidate sets.

Qualification traces file and write syscalls. The accepted executable opens
no writable side channel and writes only its declared final-boundary and
control JSON to stdout. Projection rejection diagnostics contain no
intermediate state.

## Independent reference

The reference executable uses a conventional coefficient-aware rank-pointer
Gaussian eliminator. It is intentionally algorithmically different from the
phase engine's fixed column-owned pivot rows. All nine complete 25-cell
boundaries and hashes match exactly.

## Resource accounting

For each accepted actual-inverse transaction:

```text
public program coefficients                       75
carrier cells                                    318
joint workspace cells                             56
resident phase-control cells                     112
total reusable workspace cells                   168
baseline plus working complex values             636
live carrier bytes                            10,176
comparison snapshot bytes                     10,176
carrier plus comparison heap subtotal         20,352
two coexisting program structs                    150
two compiled composition descriptors               48
execution result struct                            240
native composition calls                             4
phase ANDs                                       6,372
phase XORs                                      10,416
phase ORs                                         232
phase NOTs                                        440
scheduled conditional swaps                       384
scheduled conditional row additions               336
phase-cell updates                               9,348
carrier reads                                  12,474
boundary decodes                                    25
restoration-cell checks                            318
```

GCC `-O2 -fstack-usage` measures a conservative nested active call chain of
6,000 bytes. Adding it to the carrier-plus-comparison heap subtotal gives a
26,352-byte current-ABI accounting value. The stack figure includes the
qualification suite's coexisting fixtures and results; it is a conservative
process-level bound, not a minimal transaction stack.

No performance advantage is claimed. The matched conventional baseline is
ordinary compact binary Gaussian elimination, not dense relation-table
enumeration.

## Claim boundary

This result removes the preceding quotient-only restriction at fixed width
two: singular, kernel-sensitive, rank-deficient, universal, and empty affine
relations now close through a coefficient-oblivious phase schedule.

It does not establish:

```text
arbitrary interface width
arbitrary equation capacity
nonlinear or general Boolean relation elimination
arbitrary graph topology or treewidth
automatic elimination-order discovery
CATVM custody for this affine kernel
compactness for arbitrary relations
performance or computational advantage
physical waveform or silicon execution
Small Wall crossing
unlimited catalytic computation
```

The next representation experiment is a width-parametric version with fixed
public row capacity, complete polynomial affine boundaries, and measured
resource growth across wider shared interfaces.
