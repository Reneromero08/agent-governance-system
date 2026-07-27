# Width-Parametric Boolean Relation Tensor Trains

## Scope

This mutable experiment replaces the fixed QANF schema with a relation family
whose public interface width grows. It uses Boolean-semiring tensor trains
(TTs) as compact many-to-many relation signatures and the existing
three-phase carrier cells as their resident storage.

It is a bounded Linux software reference. It does not establish rank
minimization, fixed-rank closure at unbounded composition depth, arbitrary
QANF compactness, arbitrary topology, advantage, physical execution, Small
Wall crossing, or unlimited catalytic computation.

## Relation semantics

For binary words `X=(x_1,...,x_w)` and `Y=(y_1,...,y_w)`, a TT with Boolean
cores `A_i` denotes:

```text
chi_A(X,Y)
  = OR over internal bond states
      AND_i A_i[left_bond, x_i, y_i, right_bond]
```

Core coordinates use public lexicographic order:

```text
(((left * 2 + x) * 2 + y) * right_rank) + right
```

For uniform internal rank `r`, the compact core count is:

```text
N_r(w) = 8r + 4(w-2)r^2
```

The tested rank laws are:

```text
N_2(w) =  16w - 16
N_4(w) =  64w - 96
N_8(w) = 256w - 448
```

No `4^w` relation table is allocated.

## Native composition

For `A:X<->Y` and `B:Y<->Z`, each output core cell is:

```text
C_i[x,z,(a,c),(b,d)]
  = OR_y (
      A_i[x,y,a,b] AND B_i[y,z,c,d]
    )
```

Only the two local Boolean values of `y_i` are contracted. Boolean
distributivity proves that the resulting TT is the exact existential
composition over the complete word `Y`; the implementation never loops over
the `2^w` shared assignments.

Ranks multiply. The accepted transaction uses:

```text
F, G, J  rank 2
H=F;G    rank 4
Z=H;J    rank 8
```

The union over finite TT ranks is closed under this product-rank
construction. A fixed rank cap is not closed under unbounded composition
depth. No recompression or gauge canonicalization is claimed.

## Non-affine many-to-many witness family

The primary rank-two leaf relation `N_w`, for `w>=4`, is:

```text
y_i = x_i AND x_(i+1)  for i < w
y_w is free
```

Its bond state carries the required next input bit. Every leaf input has
exactly two outputs. The root `N_w^3` has at least two outputs per input, and
the all-zero root output has at least `Fibonacci(w+2)` inputs. These are lower
bounds for the composed root, not exact root multiplicities.

For every `i<=w-3`, the root enforces:

```text
z_i = x_i*x_(i+1)*x_(i+2)*x_(i+3)
```

The fourth Boolean derivative is one. Projection onto those four input bits
and `z_i` is therefore the degree-four AND graph, which cannot be an affine
GF(2) relation. Width four embeds the prior fixed QANF `d=abce` witness;
larger widths carry `w-3` overlapping degree-four windows. The unrelated
reuse program uses the analogous neighbor-NAND leaves.

## Carrier transaction

One accepted direct-backend transaction is:

```text
encode F/G/J
-> compose resident H from actual F/G
-> compose resident Z directly from actual H/J
-> copy actual Z into the final boundary block
-> decode only final Z cores
-> remove final boundary copy
-> inverse Z using actual H/J
-> inverse H using actual F/G
-> inverse-encode J/G/F
-> verify restoration
-> reuse the actual restored carrier
```

H is never decoded, serialized, hashed, or materialized in a second block.
Its cells are read directly as counted carrier operands while producing Z.
The final projected core bytes survive outside inverse history. State
restoration uses the predeclared
complex tolerance `2e-12`; discrete width, ranks, topology, program variant,
operation order, and restoration generation are exact.

The snapshot control reloads the verification image and keeps restoration
generation zero. It is not an accepted inverse path.

## Exact resources

With three rank-two leaves, resident H, resident Z, and a copied final
boundary:

```text
carrier cells = 3N_2 + N_4 + 2N_8 = 624w - 1040
```

An accepted forward/inverse transaction uses:

```text
phase-cell updates = 6N_2 + 2N_4 + 4N_8
logical phase ANDs = 4(N_4 + N_8)
logical phase ORs  = 2(N_4 + N_8)
carrier reads      = 8N_4 + 11N_8
final decodes      = N_8
```

The best matched generic classical TT evaluator stores `F/G/J/H/Z` in:

```text
3N_2 + N_4 + N_8 = 368w - 592 bits
```

and performs linear local Boolean contraction. The stronger
fixture-specialized baseline can emit the known rank-eight neighbor-AND-cubed
cores directly in `O(N_8)`. Dense `4^w` storage is only a nonallocated
counterfactual; rank-eight storage first becomes smaller at width five.

## Controls and ceiling

Qualification covers widths `4,5,8,12,16`, independent compact-reference
parity for every final core, deterministic replay, strict compilation,
GCC analyzer, ASan/UBSan, one-write output tracing, wrong/missing/reordered
inverse failures, snapshot separation, restored-carrier reuse, null carrier,
hidden-H projection denial, dense-request denial, and prospective rank-cap
rejection.

The claim is:

```text
BOUNDED_WIDTH_PARAMETRIC_BOOLEAN_TT_MANY_TO_MANY_RELATION_COMPOSITION_WITH_PRODUCT_RANK_NATIVE_PHASE_CLOSURE_AND_RESIDENT_INTERMEDIATE
```

within:

```text
BOUNDED_LINUX_SOFTWARE_WIDTHS4_5_8_12_16_BOOLEAN_SEMIRING_TT_RANK2_TO_RANK4_TO_RANK8_REFERENCE_ONLY
```
