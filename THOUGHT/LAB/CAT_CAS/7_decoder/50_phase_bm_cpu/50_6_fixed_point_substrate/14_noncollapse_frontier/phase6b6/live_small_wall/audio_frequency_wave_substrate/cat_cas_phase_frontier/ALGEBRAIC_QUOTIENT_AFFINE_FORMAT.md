# Projected-Affine Compact Full Relation Closure

## Status

This mutable checkpoint establishes:

```text
BOUNDED_PROJECTED_AFFINE_PHASE_RELATION_COMPOSITION_WITH_COMPACT_FULL_BOUNDARY
```

for a typed projected-affine Boolean relation family on the software phase
carrier. Unlike the preceding lazy tensor-train result, the complete surviving
relation is retained and projected; this is not a sparse-coefficient query.

## Exact relation family

A typed port value has `q` quotient bits and `f` free fibre bits. Its public
projection is:

```text
pi : F2^(q+f) -> F2^q
```

For an invertible binary matrix `A` and binary offset `a`, define:

```text
R(A,a) = {(x,y) : pi(y) = A pi(x) XOR a}
```

The complete membership relation factorizes into `q` affine parity
constraints. It needs:

```text
q*q matrix bits + q offset bits = q*(q+1) phase coefficients
```

rather than a dense table or multiaffine coefficient tensor over all
`2*(q+f)` boundary variables.

Composition closes exactly:

```text
R(B,b) o R(A,a) = R(B*A, B*a XOR b)
```

The middle quotient is fixed by the left relation, while the middle fibre is
free. Every composable pair has exactly `2^f` middle witnesses, but no witness
is selected, counted during execution, or materialized.

For full-rank `A` and `f > 0`, each complete source value relates to `2^f`
targets and each target relates to `2^f` sources. The accepted `q=8, f=4`
fixtures therefore have degree 16 in both directions. Their relation is
genuinely many-to-many even though the quotient action is bijective.

## Native phase law

Each descriptor bit is a resident Boolean-subset F3 phase symbol. Existing
branch-native interpolation computes:

```text
AND(a,b) = a*b
XOR(a,b) = a + b + a*b  mod 3
```

For phase roots `x=omega^a`, `y=omega^b`, `symbol_product(x,y)` supplies
`omega^(a*b)`. The XOR accumulator is:

```text
lock(x * y * symbol_product(x,y))
```

It remains in the Boolean `0/1` phase subset and never decodes either input.
Matrix multiplication and offset propagation use only these fixed native
phase operations.

The phase executable and independent compact GF2 reference are separate
binaries. The reference contains no phase code and neither executable
materializes tuples, assignments, fibres, witnesses, or relation tables.

## Resident two-stage transaction

For `d=q*(q+1)`, the carrier layout is:

```text
F input descriptor       cells 0d .. 1d-1
G input descriptor       cells 1d .. 2d-1
K input descriptor       cells 2d .. 3d-1
actual resident H        cells 3d .. 4d-1
actual resident Z        cells 4d .. 5d-1
public boundary          cells 5d .. 6d-1
```

The transaction is:

```text
H = COMPOSE(F,G)
Z = COMPOSE(actual H,K)
copy actual Z to the boundary
decode every final descriptor coefficient
remove the boundary using actual Z
inverse the parent by recomputing from actual H and K
inverse the child by recomputing from actual F and G
remove K, G, and F
verify restoration
reuse the actual restored carrier with another program
```

`H` is never decoded or serialized. The only `decode_root` call is in the
final-boundary latch. The 72 ordinary result coefficients survive outside the
carrier while the actual forward history is reversed.

The accepted primary and unrelated reuse boundaries have hashes:

```text
primary  33f962d84bbdd8db
reuse    1b6d1f9d91bbe788
```

Both complete 72-coefficient vectors exactly match the separate compact GF2
reference.

## Minimum nonabelian exhaustive calibration

At `q=2, f=1`, the invertible projected-affine family contains:

```text
6 invertible 2x2 matrices * 4 offsets = 24 relations
```

Each relation connects 16 of the 64 possible pairs and has degree two in both
directions. The phase engine generates one compact descriptor at a time and
passes:

```text
ordered pair closures       576 / 576
associativity triples    13,824 / 13,824
maximum restoration error  1.66533453694e-16
retained relation lookups   0
tuple/assignment/witness    0 / 0 / 0
```

The two parenthesizations use different actual resident intermediates and
produce identical complete final descriptors. This is the smallest
noncommutative case; rank one would not provide a meaningful order
discriminator.

## Scaling evidence

Strict phase/reference builds at quotient ranks `2,4,8,16,32,64` all match.
The implemented laws are:

```text
descriptor coefficients          q*(q+1)
six-block carrier cells        6*q*(q+1)
native ANDs per transaction    4*q*q*(q+1)
native XORs per transaction    4*q*q*(q+1)
symbol products per transaction 8*q*q*(q+1)
```

The largest bounded run uses:

```text
quotient bits                         64
free fibre bits                       20
port bits                             84
source and target degree            2^20
complete factor coefficients        4,160
carrier cells                       24,960
live carrier bytes                 798,720
dense pair/coefficient exponent       168
symbol products per transaction  2,129,920
maximum root error       3.72380122987e-16
restoration error        2.22044604925e-16
```

This establishes an exact polynomial representation and operator law for this
subcategory. It is not a runtime or asymptotic performance advantage over
ordinary binary matrix arithmetic.

## Restoration and controls

One `q=8` carrier allocation executes:

```text
primary transaction                 1
unrelated reuse transaction         1
alternating reuse transactions    128
maximum restoration error  2.48253415325e-16
predeclared tolerance              2e-12
```

Wrong parent inverse, missing parent inverse, and applicable child-before-
parent inversion all fail restoration. The selected matrices are
noncommuting. Two altered native laws also discriminate:

```text
ordinary F3 addition in place of Boolean XOR
reversed matrix-composition order
```

Both change the complete boundary and restore their own actual altered
histories. Intermediate projection and null-carrier paths fail before
ordinary output. Snapshot reload is labeled separately and is not credited as
the actual inverse.

Qualification also requires deterministic replay, exact JSON key allowlists,
one final decode path, phase/reference binary separation, static expanded-
relation exclusions, strict GCC, `-fanalyzer`, ASan, UBSan, leak detection,
optional Clang, source/result hashes, all-rank reference parity, and one
focused independent review.

## Resource accounting

For the accepted `q=8` in-place transaction:

```text
factor coefficients                         72
public input coefficients                   216
carrier cells                               432
baseline + working complex values           864
live carrier bytes                       13,824
comparison snapshot bytes                13,824
carrier plus comparison heap subtotal    27,648
two coexisting program structs              448
two compiled descriptor structs              48
one execution-result struct                  408
native compose calls                           4
native AND products                        2,304
native XOR accumulations                   2,304
symbol products                            4,608
carrier reads                              4,784
phase-cell updates                           864
inverse-factor recomputations                  2
boundary decodes                              72
restoration-cell checks                      432
```

GCC `-O2 -fstack-usage` measures a conservative nested active call chain of
5,776 bytes. Adding that chain to the carrier-plus-comparison heap subtotal
gives a deliberately conservative 33,424-byte current-ABI accounting value.
The separately listed program, descriptor, and result structures are already
components of that compiler-measured stack chain and are not added again.
The byte accounting applies to the accepted `q=8` build; scaling claims are
limited to the explicitly measured factor, carrier, live-byte, and native-
operation laws.

## Claim boundary

The claim ceiling is:

```text
BOUNDED_PROJECTED_AFFINE_F2_RELATION_SUBCATEGORY_REFERENCE_ONLY
```

Compact closure holds because every relation ignores the free kernel
coordinates and acts affinely and invertibly on the public quotient. This
does not establish compact closure for general affine equation systems,
kernel-sensitive constraints, singular projected maps in the accepted
bi-many family, nonlinear relations, general intersection or union,
arbitrary runtime types or topology, a compact representation for arbitrary
relations, computational advantage, CATVM enforcement for this new family,
physical execution, Small Wall crossing, or unlimited catalytic computation.

## Reproduction

```bash
evidence_dir=$(mktemp -d /tmp/quotient-affine-qual.XXXXXX)
./qualify_algebraic_quotient_affine_phase.sh . "$evidence_dir" \
    >"$evidence_dir.summary.json"
```
