# Native Width-Two Relational Phase Contraction

## Status

This mutable checkpoint establishes:

```text
NATIVE_WIDTH2_TYPED_RELATIONAL_PHASE_CONTRACTION
```

for the bounded tested Boolean/F3 chain. It is the first branch-native
relation interface that closes two unresolved Boolean ports in one native
operator body.

## Public source

```text
CATCAS_WIDE2_RELATION_CHAIN 1
RELATION F C0 ... C15
RELATION G C0 ... C15
RELATION K C0 ... C15
END
```

Every coefficient is in `{0,1,2}`. Coefficient index is the monomial subset
mask for the ordered four-port signature:

```text
[LEFT_0, LEFT_1] -> [RIGHT_0, RIGHT_1]
```

The format therefore supplies public relation algebra, not an answer,
witness, candidate set, or truth table.

## Native operator

A four-port relation is a multiaffine polynomial over:

```text
F3[q0,...,qn] / (qi^2 - qi)
```

Its 16 coefficients occupy 16 relative cube-root phase cells. For resident
relations `F(X,Y)` and `G(Y,Z)`, where each of `X`, `Y`, and `Z` is a pair of
Boolean variables, the native body computes:

```text
J = F^2 + G^2
N_t(P) = P|t=0 * P|t=1
CONTRACT2(F,G) = N_Y1(N_Y0(J))
```

Over F3, `F^2 + G^2 = 0` exactly when both `F = 0` and `G = 0`. For a Boolean
variable, the norm product is zero exactly when at least one of its two
fibers is zero. The two norms therefore implement existential closure across
both shared ports.

All algebra remains in cube-root phase symbols. Boolean-quotient
multiplication uses OR-index convolution. Restriction at one uses coefficient
addition, implemented as phase multiplication. `symbol_product` supplies
coefficient multiplication. No phase coefficient is decoded inside
`CONTRACT2`, and there is no loop over shared assignments.

## Resident chain and reversal

The accepted transaction uses:

```text
F          cells  0..15
G          cells 16..31
K          cells 32..47
resident H cells 48..63 = CONTRACT2(F,G)
resident Z cells 64..79 = CONTRACT2(actual H,K)
public     cells 80..95
```

Only the public cells are decoded. The controller-facing CLI rejects an
attempted intermediate projection before reading a program or carrier.

Reversal removes the public boundary, recomputes and inverses
`CONTRACT2(H,K)` while `H` is still resident, then recomputes and inverses
`CONTRACT2(F,G)`, and finally inverses the public relation encodings. A
comparison snapshot is never loaded. The unrelated reuse source then
executes on the same restored 96-cell carrier allocation.

## Demonstrated result

```text
primary boundary [1,0,2,1,2,1,2,1,0,0,1,1,1,1,1,1]
reuse boundary   [1,0,2,1,0,0,1,1,2,1,2,1,1,1,1,1]
maximum correct restoration 1.66533453694e-16
predeclared restoration tolerance 2e-12
```

The decoded final boundary remains in ordinary result state while inverse
execution restores the carrier, so the result survives outside inverse
history.

## Controls

Qualification requires:

```text
wrong public-boundary inverse
omitted parent contraction inverse
reordered noncommuting child-before-parent inverse
bypassed first Boolean norm
ordinary polynomial sum in place of the first norm product
swapped shared-port order
attempted intermediate projection rejection
null-carrier rejection
malformed coefficient and relation-order rejection
exact output-key allowlist
preprojection output-sink rejection
fresh-process byte determinism
strict GCC and static analysis
ASan, UBSan, and leak detection
source, fixture, and result hashes
focused independent review
```

Wrong, omitted, and reordered inverses each leave restoration error
`1.73205080757`. The three prospective operator controls all change the final
boundary and reverse their actual altered histories cleanly.

The final coefficient vectors are predeclared in the qualifier. They are not
computed by a controller-side oracle during the accepted run. The evidence is
therefore exact fixture comparison plus independent algebra/code inspection,
not differential validation against a second implementation that would
materialize the unresolved intermediate outside the phase carrier.

## Resource law

At width two, one native contraction accounts for:

```text
relation/message cells                            16
transient union polynomial coefficients           64
maximum CONTRACT2 complex workspace              240
symbol products                                 1,792
coefficient accumulation additions              1,792
restriction and intersection additions            112
```

The accepted primary-plus-reuse path reports:

```text
carrier creation count                              1
transactions on that carrier                        2
carrier cells                                      96
live carrier complex values                       192
comparison snapshot complex values                192
two current-ABI descriptors                        48 bytes
two parsed program structures                     400 bytes
accounted execution upper bound                   656 complex values
accounted execution upper bound                10,496 bytes
```

The upper bound conservatively adds resources that do not all have overlapping
C abstract lifetimes. It is an accounting ceiling, not a measured exact RSS
peak. Descriptor byte counts are the current x86_64 ABI result.

For separator width `w`, the dense construction has:

```text
persistent relation/message cells  2^(2w)
transient union coefficients       2^(3w)
successive norm widths             2^(3w-1) ... 2^(2w)
```

This exponential law is a disclosed blocker, not hidden leverage.

## Claim boundary

The accepted ceiling is:

```text
SOFTWARE_COMPACT_WIDTH2_TYPED_RELATIONAL_PHASE_CONTRACTION_AND_RESTORATION_REFERENCE_ONLY
```

`compact` refers to one reusable `CONTRACT2` body and two descriptors instead
of expansion into four shared assignments. It does not refer to separator
storage: the dense coefficient representation is exponential in width.

This does not establish arbitrary width, relation arity or domain, arbitrary
topology or treewidth, compact separator storage, computational advantage,
CATVM enforcement for the wider operator, physical phase execution, Small
Wall crossing, or unlimited catalytic computation.

The next blocker is a factorized wider-interface representation whose native
closure preserves useful structure without materializing `2^(2w)` dense
relation coefficients.

## Reproduction

```bash
evidence_dir=$(mktemp -d /tmp/wide2-qual.XXXXXX)
./qualify_algebraic_wide_interface_phase.sh . "$evidence_dir" \
    >"$evidence_dir.summary.json"
```
