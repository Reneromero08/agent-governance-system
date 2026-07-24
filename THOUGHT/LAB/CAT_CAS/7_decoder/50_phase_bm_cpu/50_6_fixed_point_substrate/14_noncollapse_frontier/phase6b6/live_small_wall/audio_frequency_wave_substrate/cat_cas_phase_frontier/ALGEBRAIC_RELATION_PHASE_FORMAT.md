# Algebraic Relation Phase Format

**Status:** mutable development format

**Schema token:** `CATCAS_ALGEBRAIC_RELATION_PROCESS 1`

## Typed ports

```text
PORT <name> BOOLEAN_F3
```

The semantic port domain is the Boolean subset `{0,1}` of `F3`. The native
carrier does not instantiate either value. Port identifiers are unique.

## Algebraic relation signature

```text
RELATION <name> <first-port> <second-port> c00 c10 c01 c11 ZEROSET
```

Each coefficient is canonical `0`, `1`, or `2`. The relation is the
Boolean-domain zero set of the multiaffine polynomial

```text
c00 + c10*x + c01*y + c11*x*y = 0 mod 3
```

This is an algebraic signature, not a list of accepted pairs. Reversing the
two port names swaps `c10` and `c01`. Relation identifiers are unique.

After orienting the chain, each local signature must be total toward the
closed Boolean port:

```text
for every open-port value, at least one internal value in {0,1}
satisfies the local polynomial
```

The parser enforces this prospectively from the public coefficients. It does
not inspect or select a boundary result.

## Composition and internal closure

Version 1 contains three compatible ports and two relations joined through
one closed internal port:

```text
CLOSE <internal-port>
BOUNDARY <first-open-port> <second-open-port>
END
```

For normalized relations

```text
f(x,y) = A(x)y + B(x)
g(y,z) = C(z)y + D(z)
```

the native boundary signature is the linear resultant

```text
R(x,z) = B(x)C(z) - A(x)D(z).
```

The four boundary coefficients are roots-of-unity phase relations. Each is
formed by two phase-native field products and a conjugate subtraction. The
native operator has no loop over port values, tuples, internal assignments,
or witnesses.

## Boundary, inverse, and reuse

The boundary coefficient phases are decoded only after elimination. The
decoded signature survives outside the reversed carrier history. The inverse
recomputes the resultant factors from the still-resident input signatures,
then removes the public input encodings. The same restored carrier is reused
for the next process.

## Exactness boundary

A polynomial resultant is not universally identical to existential
projection over a restricted Boolean subset. Version 1 therefore admits only
relations total toward the closed Boolean port. For each fixed open value,
the local affine equation in the internal variable then denotes either
`{0}`, `{1}`, or `{0,1}`—never the empty set. For two such affine fibers,
the resultant determinant is zero exactly when the two nonempty Boolean root
sets intersect. This makes the resultant exact on the admitted class.

An exhaustive independent scalar survey covers all `81 x 81 = 6,561`
coefficient-signature pairs. Without the law, only 3,217 pairs are exact.
The law admits 25 left signatures and 25 right signatures, and all 625
admitted pairs are exact.

`LEQ(x,y)`, encoded as `x(1-y)=0`, is the primary non-functional calibration:

```text
LEQ o LEQ = LEQ
```

The degenerate `EMPTY o ANY` fixture is retained as a decisive admission
negative. Its zero resultant would denote every boundary pair while the
Boolean existential composition is empty, but both native and reference
parsers now reject it because the left relation is not total toward the
closed port.

The mechanism is still a restricted total-fiber, bivariate,
single-internal-port resultant. It does not establish empty internal fibers,
arbitrary arity, multiple simultaneous internal variables, general relational
trace, holographic advantage, physical computation, or unlimited catalytic
computation.
