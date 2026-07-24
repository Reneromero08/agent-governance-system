# Exact Algebraic Cycle Phase Format

Status: mutable CAT_CAS frontier checkpoint

## Purpose

This checkpoint is the first native relational loop in the lane. It closes
the fixed public diamond:

```text
A -- U -- W -- V -- B
     `-- Z --'
```

The two `U--V` paths are independently composed, intersected, and then
composed with the two open leaf relations. Empty and partial intermediate
relations are allowed. Four complex phases carry every relation; no
intermediate coefficient or witness is decoded.

## Canonical grammar

```text
CATCAS_ALGEBRAIC_CYCLE 1
TYPE BOOLEAN_F3
TOPOLOGY DIAMOND_U_W_Z_V
RELATION LEAF_A A U c00 c10 c01 c11
RELATION UPPER_LEFT U W c00 c10 c01 c11
RELATION UPPER_RIGHT W V c00 c10 c01 c11
RELATION LOWER_LEFT U Z c00 c10 c01 c11
RELATION LOWER_RIGHT Z V c00 c10 c01 c11
RELATION LEAF_B B V c00 c10 c01 c11
END
```

Relation records may appear in any order or endpoint direction. Each role
appears exactly once. Coefficients are one digit in `{0,1,2}`. Records end
with LF. Blank records, CR, records after `END`, missing roles, duplicate
roles, noncanonical coefficients, wrong endpoints, and the wrong topology
reject.

Unlike the earlier tree engine, this exact operator does not require
bi-total input relations.

## Exact polynomial algebra

Relations are zero sets of multiaffine polynomials in

```text
B = F3[x,y] / (x^2-x, y^2-y).
```

The four coefficient phases correspond to basis `{1,x,y,xy}`. Polynomial
multiplication is a fixed phase operator: basis indices combine by bitwise
OR, coefficient multiplication uses the roots-of-unity product polynomial,
and coefficient addition uses phase multiplication.

For two relations on the same ports, exact intersection is:

```text
INTERSECT(f,g) = f^2 + g^2.
```

Every nonzero element of `F3` squares to one. Therefore:

```text
INTERSECT(f,g) = 0  iff  f = 0 and g = 0.
```

For `f(x,u)` and `g(u,y)`, exact Boolean existential composition is:

```text
K0(x,y) = f(x,0)^2 + g(0,y)^2
K1(x,y) = f(x,1)^2 + g(1,y)^2
COMPOSE(f,g) = K0 * K1.
```

The product is zero iff either Boolean witness makes both input relations
zero. This is an algebraic identity, not a host witness loop. The native
engine evaluates the fixed phase polynomial directly and writes its four
output phases to the carrier.

The separate complete survey exhausts all 81 multiaffine F3 polynomials in
both operand positions:

```text
ordered pairs                 6,561
composition boundary rows    26,244
composition mismatches            0
intersection boundary rows   26,244
intersection mismatches           0
```

## Native lifecycle

Carrier layout:

```text
six public edge relations     24 cells
upper path relation            4 cells
lower path relation            4 cells
cycle intersection relation    4 cells
A-to-V relation                4 cells
final A-to-B boundary           4 cells
total                          44 cells
tuple/witness/truth slots       0
retained inverse factors        0
```

Execution is:

```text
borrow carrier
encode six public relations
compose U-W-V
compose U-Z-V
intersect the two U-V path relations
compose A-U with the cycle core
compose A-V with B-V
latch A-B boundary
reverse the exact five native operators
reverse six inputs
restore and reuse the actual carrier
```

## Decisive discriminator

The primary fixture is chosen prospectively by the complete operator survey.
Exact cycle closure has zero mask `0001`, accepting only boundary `00`.

```text
exact cycle                 {00}
bypass lower cycle path     {00,01}
ordinary phase addition     {00,11}
```

Both wrong forward mechanisms reverse their own actual histories and restore
cleanly. They therefore discriminate the relational operator rather than
generic corruption.

Rotating the boundary coefficient inverse and omitting the cycle-intersection
inverse each leave restoration error `1.73205080757`.

## Reference separation

`algebraic_cycle_reference.c` independently parses the process, reproduces
the scalar polynomial identities, and enumerates all 64 assignments of
`A,B,U,V,W,Z`. It is not linked into the native binary.

`algebraic_exact_operator_survey.c` independently exhausts the complete
81-by-81 polynomial operand class. Neither scalar executable participates in
native recurrence or result selection.

## Claim boundary

A passing checkpoint establishes only:

```text
SOFTWARE_FIXED_CYCLE_EXACT_RELATIONAL_PHASE_CLOSURE_REFERENCE_ONLY
```

It establishes one fixed cyclic relational graph, exact phase-native
intersection, exact phase-native Boolean existential composition, a
loop-dependent boundary, actual reversal, restoration, and reuse.

It does not establish a generic cyclic graph language, arbitrary arity,
compact handling of growing graph treewidth, computational advantage,
physical phase computation, or unlimited catalytic phase computation.
